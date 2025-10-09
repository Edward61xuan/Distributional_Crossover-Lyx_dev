"""dc_vectorized_optimized.py

Fully vectorized and JIT-optimized version of Distributed Crossover (DC).
This version eliminates Python loops and maximizes JAX performance.

Key optimizations:
1. Complete vectorization using jax.vmap
2. Template-based parameter management
3. JIT-compiled core algorithms
4. Efficient multi-GPU support via jax.pmap
5. Minimal host-device communication
"""

import os
# os.environ["WANDB_MODE"] = "online"

from dataclasses import dataclass
from typing import Callable, Any, Dict
import time

import jax
import jax.numpy as jnp
from jax import random, vmap, pmap
from functools import partial
import numpy as np
import math

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from brax import envs
import flax

import ec
from ec import ESConfig, _runner_init, _evaluate_step
from networks import NETWORKS
import optax
from utils.functions import save_obj_to_file


@dataclass
class DCConfig:
    # DC algorithm hyper-params
    seed: int = 0
    dims: int = 64
    population_size: int = 64
    sample_size: int = 8
    crossover_ratio: int = 2
    local_lr: float = 0.01
    eps: float = 1e-3
    total_generations: int = 100
    project_name: str = "DC-vectorized"
    run_name: str = "DC-run"
    warmup_steps: int = 0
    save_every: int = 50


def setup_devices():
    """Setup device configuration."""
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Found {n_devices} devices: {devices}")
    
    if n_devices == 1:
        print("Single device mode")
        return devices, 1, False
    
    print(f"Multi-device mode with {n_devices} devices")
    return devices, n_devices, True


def create_params_template(params):
    """Create a template containing only shapes and dtypes for JIT compilation."""
    def extract_template(x):
        return x.shape, x.dtype
    
    return jax.tree_map(extract_template, params)


def make_runner_arrays(runner: ec.RunnerState) -> Dict[str, Any]:
    """Extract only the array-like parts from RunnerState so they can be
    passed safely into jitted/pmaped functions.

    Keeps: params, fixed_weights, normalizer_state, env_reset_pool
    """
    ra = {}
    # key
    ra['key'] = jnp.asarray(runner.key)
    # params and fixed_weights are pytrees of arrays
    ra['params'] = jax.tree_map(lambda x: jnp.asarray(x), runner.params)
    ra['fixed_weights'] = jax.tree_map(lambda x: jnp.asarray(x), runner.fixed_weights)
    # optimizer state (may be pytree of arrays)
    try:
        ra['opt_state'] = jax.tree_map(lambda x: jnp.asarray(x), runner.opt_state)
    except Exception:
        # Some opt_state entries may be non-array (e.g., None or scalars); keep as-is
        ra['opt_state'] = runner.opt_state
    # normalizer_state may contain arrays + scalars
    try:
        ra['normalizer_state'] = jax.tree_map(lambda x: jnp.asarray(x), runner.normalizer_state)
    except Exception:
        ra['normalizer_state'] = runner.normalizer_state
    # env_reset_pool is the vmap'd reset state for the env; keep as-is (pytree of arrays)
    try:
        ra['env_reset_pool'] = jax.tree_map(lambda x: jnp.asarray(x), runner.env_reset_pool)
    except Exception:
        ra['env_reset_pool'] = runner.env_reset_pool
    return ra


def make_conf_static(es_conf: ESConfig) -> Dict[str, Any]:
    """Create a small 'static' conf object containing only values that are safe
    to treat as static args to jitted/pmaped functions.

    Note: callables (network.apply, env.step/reset functions) are included here
    but must be passed as static_argnums to jitted functions.
    """
    # Create a small flax.struct dataclass to hold static conf (hashable)
    @flax.struct.dataclass
    class ConfStatic:
        pop_size: Any = None
        lr: Any = None
        eps: Any = None
        weight_decay: Any = None
        warmup_steps: Any = None
        eval_size: Any = None

        action_dtype: Any = None
        p_dtype: Any = None
        network_dtype: Any = None

        network_cls: Any = None
        env_cls: Any = None

        observation_size: Any = None
        action_size: Any = None

    scalar_fields = ['pop_size', 'lr', 'eps', 'weight_decay', 'warmup_steps', 'eval_size']
    kwargs = {}
    for f in scalar_fields:
        kwargs[f] = getattr(es_conf, f, None)
    kwargs['action_dtype'] = getattr(es_conf, 'action_dtype', jnp.float32)
    kwargs['p_dtype'] = getattr(es_conf, 'p_dtype', jnp.float32)
    kwargs['network_dtype'] = getattr(es_conf, 'network_dtype', jnp.float32)
    kwargs['network_cls'] = getattr(es_conf, 'network_cls', None)
    kwargs['env_cls'] = getattr(es_conf, 'env_cls', None)
    if getattr(es_conf, 'env_cls', None) is not None:
        kwargs['observation_size'] = getattr(es_conf.env_cls, 'observation_size', None)
        kwargs['action_size'] = getattr(es_conf.env_cls, 'action_size', None)
    else:
        kwargs['observation_size'] = None
        kwargs['action_size'] = None

    return ConfStatic(**kwargs)

def make_flat_masks_to_pytree_jit(template_treedef):
    """
    返回一个 jitted 函数，该函数把 flat masks (batch, total_dim) -> pytree。
    template_shapes 和 template_dtypes 仍作为静态参数传入（必须是 hashable），
    template_treedef 通过闭包捕获（不作为参数）。
    """
    @partial(jax.jit, static_argnums=(1,2))
    def _flat_masks_to_pytree(masks_flat: jnp.ndarray, template_shapes, template_dtypes):
        # sizes: list of Python ints
        sizes = [math.prod(shape) for shape in template_shapes]
        cuts = np.cumsum([0] + sizes)  # numpy, not jax
        cuts = cuts.tolist()

        batch = masks_flat.shape[0]
        out_leaves = []

        for i, (shape, dtype_str) in enumerate(zip(template_shapes, template_dtypes)):
            start, end = cuts[i], cuts[i + 1]
            piece = masks_flat[:, start:end]
            reshaped = jnp.reshape(piece, (batch, *shape))
            out_leaves.append(reshaped.astype(jnp.bool_))

        return jax.tree_util.tree_unflatten(template_treedef, out_leaves)
    return _flat_masks_to_pytree


@partial(jax.jit, static_argnums=(2,))
def sample_from_rho_vectorized(key, rho: jnp.ndarray, n: int) -> jnp.ndarray:
    """Vectorized JIT-compiled sampling from Bernoulli(rho)."""
    u = random.uniform(key, (n, rho.shape[0]))
    return (u < rho).astype(jnp.float32)


@jax.jit
def centered_rank_transform_jit(x: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled centered rank transform."""
    x = x.ravel()
    ranks = jnp.argsort(jnp.argsort(x))
    return ranks / (x.size - 1) - 0.5


@jax.jit
def estimate_gradient_vectorized(rho: jnp.ndarray, samples: jnp.ndarray, rewards: jnp.ndarray) -> jnp.ndarray:
    """Vectorized JIT-compiled gradient estimation."""
    shaped = centered_rank_transform_jit(rewards)
    diffs = samples - rho
    grad = jnp.mean(diffs * shaped[:, None], axis=0)
    return grad


@partial(jax.jit, static_argnums=(2,))
def generate_offspring_vectorized(key, population: jnp.ndarray, ratio: int) -> jnp.ndarray:
    """Vectorized JIT-compiled offspring generation."""
    M = population.shape[0]
    num_off = M * ratio
    pk, bk = random.split(key)
    parent_idx = random.randint(pk, (num_off, 2), 0, M)
    betas = random.uniform(bk, (num_off, 1))
    parentA = population[parent_idx[:, 0]]
    parentB = population[parent_idx[:, 1]]
    offspring = betas * parentA + (1.0 - betas) * parentB
    return offspring


@jax.jit
def select_next_generation_vectorized(population: jnp.ndarray, pop_fitness: jnp.ndarray, 
                                    offspring: jnp.ndarray, off_fitness: jnp.ndarray) -> jnp.ndarray:
    """Vectorized JIT-compiled selection."""
    combined = jnp.concatenate([population, offspring], axis=0)
    combined_fitness = jnp.concatenate([pop_fitness, off_fitness], axis=0)
    idx = jnp.argsort(combined_fitness)[::-1]
    selected = combined[idx[:population.shape[0]]]
    return selected


def evaluate_population_batch(key: jnp.ndarray, masks_flat: jnp.ndarray, 
                            runner: ec.RunnerState, conf: ESConfig,
                            template_shapes, template_dtypes, template_treedef) -> jnp.ndarray:
    """Backward-compatible wrapper kept for legacy calls. Delegates to
    evaluate_population_batch_jit when runner_arrays/conf_static are not used.
    """
    # Fallback to original implementation for backward compatibility
    params_pytree = flat_masks_to_pytree_jit(masks_flat, template_shapes, template_dtypes)
    B = masks_flat.shape[0]
    carry_key, reset_key = random.split(key)
    network_states = conf.network_cls.initial_carry(carry_key, B)
    env_reset = conf.env_cls.reset(random.split(reset_key, B))
    pop = ec.PopulationState(
        network_params=params_pytree,
        network_states=network_states,
        env_states=env_reset,
        fitness_totrew=jnp.zeros(B),
        fitness_sum=jnp.zeros(B),
        fitness_n=jnp.zeros(B, dtype=jnp.int32)
    )
    local_runner = runner.replace(env_reset_pool=env_reset)
    def _cond(p):
        return jnp.any(p.fitness_n == 0)
    pop = jax.lax.while_loop(_cond, (lambda p: _evaluate_step(p, local_runner, conf)), pop)
    fitness = pop.fitness_sum / pop.fitness_n
    return fitness


# JIT-able batch evaluator which accepts only array-like runner_arrays and conf_static
@partial(jax.jit, static_argnums=(3,4,5,6))
def evaluate_population_batch_jit(key: jnp.ndarray, masks_flat: jnp.ndarray,
                                 runner_arrays: Dict[str, Any], conf_static: Dict[str, Any],
                                 template_shapes, template_dtypes, template_treedef) -> jnp.ndarray:
    """JIT-friendly batch evaluation. Expects runner_arrays to contain array-only
    pytrees for params, fixed_weights, normalizer_state and env_reset_pool. conf_static
    must include network_cls and env_cls as static values (passed via static_argnums).
    """
    # reconstruct params pytree from flat masks
    params_pytree = flat_masks_to_pytree_jit(masks_flat, template_shapes, template_dtypes)

    B = masks_flat.shape[0]

    # network initial carry and env reset are provided as static callables via conf_static
    network_initial_carry = conf_static.network_cls.initial_carry
    env_reset_fn = conf_static.env_cls.reset

    carry_key, reset_key = random.split(key)
    network_states = network_initial_carry(carry_key, B)
    env_reset = env_reset_fn(random.split(reset_key, B))

    pop = ec.PopulationState(
        network_params=params_pytree,
        network_states=network_states,
        env_states=env_reset,
        fitness_totrew=jnp.zeros(B),
        fitness_sum=jnp.zeros(B),
        fitness_n=jnp.zeros(B, dtype=jnp.int32)
    )

    # create a lightweight runner-like structure using runner_arrays
    # We create a simple object with attributes used in _evaluate_step by wrapping into a flax.struct-like container
    @flax.struct.dataclass
    class RunnerArraysWrapper:
        key: Any
        normalizer_state: Any
        env_reset_pool: Any
        params: Any
        fixed_weights: Any

    # Some RunningStatisticsState implementations may contain per-env buffers
    # (leading axis = number of envs in the pool). When we evaluate a different
    # batch size B (e.g. flattened M*N), those per-env buffers can have a
    # mismatched leading axis and will fail broadcasting inside
    # running_statistics.normalize. To be robust, detect array leaves in
    # normalizer_state that have shape (E, obs_dim) where E != B but
    # obs_dim matches, and reduce them to a single (obs_dim,) global stat by
    # taking the mean across axis 0. This yields a normalizer_state that can
    # broadcast across any batch of observations.
    def align_normalizer_state(norm_state, obs_batch):
        obs_shape = obs_batch.shape
        # obs_shape = (B, obs_dim) expected
        def _fix_leaf(x):
            if isinstance(x, jnp.ndarray) and x.ndim == 2:
                # If leaf looks like (E, obs_dim) and obs_dim matches
                if x.shape[1] == obs_shape[1] and x.shape[0] != obs_shape[0]:
                    return jnp.mean(x, axis=0)
            return x
        try:
            return jax.tree_map(_fix_leaf, norm_state)
        except Exception:
            # If tree_map fails for unexpected structure, return original
            return norm_state

    adjusted_normalizer = align_normalizer_state(runner_arrays.get('normalizer_state'), pop.env_states.obs)

    local_runner = RunnerArraysWrapper(
        key=runner_arrays.get('key'),
        normalizer_state=adjusted_normalizer,
        env_reset_pool=env_reset,
        params=runner_arrays.get('params'),
        fixed_weights=runner_arrays.get('fixed_weights')
    )

    # evaluation loop: inline the numeric logic from ec._evaluate_step so we
    # don't capture the Python-level _evaluate_step (which may hold non-hashable
    # closures). This keeps the loop body pure-JAX and jittable with
    # conf_static/network_cls/env_cls as static values.
    def _cond(p):
        return jnp.any(p.fitness_n == 0)

    vmapped_apply = jax.vmap(conf_static.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))

    def _where_done(x, y, done):
        d = done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(d, x, y)

    def _body(p):
        # normalize observations
        obs_norm = ec.running_statistics.normalize(p.env_states.obs, local_runner.normalizer_state)

        # network forward (vmapped over batch)
        new_network_states, act = vmapped_apply({"params": p.network_params, "fixed_weights": local_runner.fixed_weights}, p.network_states, obs_norm)

        # clip and cast
        act = jnp.clip(act, -1, 1)
        if act.dtype != conf_static.action_dtype:
            act = jnp.where(jnp.isnan(act), 0, act).astype(conf_static.action_dtype)

        # env step
        new_env_states = conf_static.env_cls.step(p.env_states, act)

        # fitness updates
        new_fitness_totrew = p.fitness_totrew + new_env_states.reward
        new_fitness_sum = jnp.where(new_env_states.done, p.fitness_sum + new_fitness_totrew, p.fitness_sum)
        new_fitness_n = jnp.where(new_env_states.done, p.fitness_n + 1, p.fitness_n)
        new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)

        # reset done envs using env_reset_pool
        new_env_states = jax.tree_map(lambda x, y: _where_done(x, y, new_env_states.done), local_runner.env_reset_pool, new_env_states)

        return p.replace(
            network_states=new_network_states,
            env_states=new_env_states,
            fitness_totrew=new_fitness_totrew,
            fitness_sum=new_fitness_sum,
            fitness_n=new_fitness_n
        )

    pop = jax.lax.while_loop(_cond, _body, pop)

    fitness = pop.fitness_sum / pop.fitness_n
    return fitness


@partial(jax.jit, static_argnums=(2, 3, 4, 5,6,7,8))
def dc_step_vectorized_core(key, population, sample_size: int, local_lr: float, 
                          eps: float, crossover_ratio: int, 
                          template_shapes, template_dtypes, template_treedef,
                          runner_arrays, conf_static):
    """Fully vectorized DC step - core algorithmic parts only."""
    
    M = population.shape[0]
    
    # Generate all keys at once
    keys = random.split(key, M + 3)
    new_key = keys[0]
    grad_keys = keys[1:M+1]
    off_key = keys[M+1]
    eval_key = keys[M+2]
    
    # Vectorized sampling for gradient estimation
    def sample_and_estimate_grad(grad_key, rho):
        # Sample from this rho
        samples = sample_from_rho_vectorized(grad_key, rho, sample_size)
        return samples, grad_key
    
    # vmap over all individuals in population
    vmap_sample = vmap(sample_and_estimate_grad, in_axes=(0, 0))
    all_samples, sample_keys = vmap_sample(grad_keys, population)
    
    # Placeholder for gradient computation (will be replaced with actual evaluation)
    # For now, use a simple placeholder that can be JIT-compiled
    def compute_grad_placeholder(samples, rho):
        # Simple placeholder: gradient proportional to sample mean
        rewards = jnp.sum(samples, axis=1)  # Simple reward function
        return estimate_gradient_vectorized(rho, samples, rewards)
    
    # vmap gradient computation
    vmap_grad = vmap(compute_grad_placeholder, in_axes=(0, 0))
    grads = vmap_grad(all_samples, population)
    
    # Vectorized population update
    population = population + local_lr * grads
    population = jnp.clip(population, eps, 1.0 - eps)
    
    # Vectorized offspring generation
    offspring = generate_offspring_vectorized(off_key, population, crossover_ratio)
    
    # Placeholder fitness evaluation (deterministic mapping)
    pop_samples = (population > 0.5).astype(jnp.float32)
    pop_fitness = jnp.sum(pop_samples, axis=1)  # Simple fitness
    
    off_samples = (offspring > 0.5).astype(jnp.float32)
    off_fitness = jnp.sum(off_samples, axis=1)  # Simple fitness
    
    # Vectorized selection
    selected = select_next_generation_vectorized(population, pop_fitness, offspring, off_fitness)
    
    metrics = {
        'best': jnp.max(pop_fitness),
        'mean': jnp.mean(pop_fitness)
    }
    
    return new_key, selected, metrics

def dc_step_with_evaluation(key, population, sample_size: int, local_lr: float, 
                          eps: float, crossover_ratio: int,
                          runner, conf, template_info, runner_arrays=None, conf_static=None):
    """DC step with real environment evaluation (non-JIT parts)."""
    
    template_shapes, template_dtypes, template_treedef = template_info
    M = population.shape[0]
    
    # Generate keys (extra key for batched evaluation)
    keys = random.split(key, M + 4)
    new_key = keys[0]
    grad_keys = keys[1:M+1]
    off_key = keys[M+1]
    eval_key_samples = keys[M+2]
    eval_key_parents_off = keys[M+3]

    # --- Batched sampling for all M individuals ---
    # vmap sampling: produces shape (M, sample_size, D)
    vmap_sample = vmap(lambda k, rho: sample_from_rho_vectorized(k, rho, sample_size), in_axes=(0, 0))
    all_samples = vmap_sample(grad_keys, population)  # (M, N, D)

    # Flatten to (M*N, D) and evaluate once on the large batch
    M_val = M
    N_val = all_samples.shape[1]
    D_val = all_samples.shape[2]
    samples_flat = jnp.reshape(all_samples, (M_val * N_val, D_val))

    # Prefer jitted array-based evaluator if runner_arrays/conf_static are provided
    if runner is not None and (runner_arrays is not None) and (conf_static is not None):
        rewards_flat = evaluate_population_batch_jit(
            eval_key_samples, samples_flat, runner_arrays, conf_static,
            template_shapes, template_dtypes, template_treedef
        )
    elif runner is not None:
        rewards_flat = evaluate_population_batch(
            eval_key_samples, samples_flat, runner, conf,
            template_shapes, template_dtypes, template_treedef
        )
    else:
        # placeholder deterministic reward when no runner
        rewards_flat = jnp.sum(samples_flat, axis=1)

    # Reshape rewards back to (M, N)
    rewards = jnp.reshape(rewards_flat, (M_val, N_val))

    # Compute gradients for each individual via vmap (all in JAX)
    vmap_est_grad = vmap(lambda rho, samples, r: estimate_gradient_vectorized(rho, samples, r), in_axes=(0, 0, 0))
    grads = vmap_est_grad(population, all_samples, rewards)
    
    # Vectorized population update (JIT-compiled)
    population = population + local_lr * grads
    population = jnp.clip(population, eps, 1.0 - eps)
    
    # Vectorized offspring generation (JIT-compiled)
    offspring = generate_offspring_vectorized(off_key, population, crossover_ratio)
    
    # Evaluate parents and offspring
    # Batch parent + offspring evaluation together for efficiency (single call)
    pop_samples = (population > 0.5).astype(jnp.float32)
    off_samples = (offspring > 0.5).astype(jnp.float32)

    # concat parents then offspring -> shape (2*M, D)
    eval_batch = jnp.concatenate([pop_samples, off_samples], axis=0)

    if runner is not None:
        if (runner_arrays is not None) and (conf_static is not None):
            fitness_concat = evaluate_population_batch_jit(
                eval_key_parents_off, eval_batch, runner_arrays, conf_static,
                template_shapes, template_dtypes, template_treedef
            )
        else:
            # legacy single-call evaluate also supports batching
            fitness_concat = evaluate_population_batch(
                eval_key_parents_off, eval_batch, runner, conf,
                template_shapes, template_dtypes, template_treedef
            )
        # split back
        pop_fitness = fitness_concat[:M]
        off_fitness = fitness_concat[M:]
    else:
        # fallback deterministic/simple reward
        print("Error: No runner provided for evaluation! Using simple sum reward.")
        summed = jnp.sum(eval_batch, axis=1)
        pop_fitness = summed[:M]
        off_fitness = summed[M:]
    
    # Vectorized selection (JIT-compiled)
    selected = select_next_generation_vectorized(population, pop_fitness, offspring, off_fitness)
    
    def binary_entropy(p):
        return - (p * jnp.log(p + 1e-8) + (1 - p) * jnp.log(1 - p + 1e-8))
    
    metrics = {
        'best_fitness_parent': jnp.max(pop_fitness),
        'mean_fitness_parent': jnp.mean(pop_fitness),
        'var_fitness_parent': jnp.var(pop_fitness),
        'best_fitness_offspring': jnp.max(off_fitness),
        'mean_fitness_offspring': jnp.mean(off_fitness),
        'var_fitness_offspring': jnp.var(off_fitness),
        # Diversity
        'entropy_population': jnp.mean(binary_entropy(population)),
        # Gradient stats
        'grad_norm': jnp.linalg.norm(grads) / grads.size,
        'grad_var': jnp.var(grads),
        # Evolution pressure (offspring - parent mean)
        'evolution_force': jnp.mean(off_fitness) - jnp.mean(pop_fitness),
    }
    
    return new_key, selected, metrics

def dc_step_with_evaluation_core(
    key, population, sample_size: int, local_lr: float, 
    eps: float, crossover_ratio: int,
    runner_arrays, conf_static, template_info
):
    """Jittable DC step with metrics (parents+offspring)."""

    template_shapes, template_dtypes, template_treedef = template_info
    M = population.shape[0]

    # Generate keys
    keys = random.split(key, M + 4)
    new_key = keys[0]
    grad_keys = keys[1:M+1]
    off_key = keys[M+1]
    eval_key_samples = keys[M+2]
    eval_key_parents_off = keys[M+3]

    # --- Batched sampling for all M individuals ---
    vmap_sample = vmap(lambda k, rho: sample_from_rho_vectorized(k, rho, sample_size), in_axes=(0, 0))
    all_samples = vmap_sample(grad_keys, population)  # (M, N, D)

    # Flatten and evaluate once
    M_val, N_val, D_val = all_samples.shape
    samples_flat = jnp.reshape(all_samples, (M_val * N_val, D_val))

    rewards_flat = evaluate_population_batch_jit(
        eval_key_samples, samples_flat, runner_arrays, conf_static,
        template_shapes, template_dtypes, template_treedef
    )
    rewards = jnp.reshape(rewards_flat, (M_val, N_val))

    # Compute gradients
    vmap_est_grad = vmap(
        lambda rho, samples, r: estimate_gradient_vectorized(rho, samples, r),
        in_axes=(0, 0, 0)
    )
    grads = vmap_est_grad(population, all_samples, rewards)

    # Update population
    population = population + local_lr * grads
    population = jnp.clip(population, eps, 1.0 - eps)

    # Generate offspring
    offspring = generate_offspring_vectorized(off_key, population, crossover_ratio)

    # --- Evaluate parents + offspring ---
    pop_samples = (population > 0.5).astype(jnp.float32)
    off_samples = (offspring > 0.5).astype(jnp.float32)
    eval_batch = jnp.concatenate([pop_samples, off_samples], axis=0)

    fitness_concat = evaluate_population_batch_jit(
        eval_key_parents_off, eval_batch, runner_arrays, conf_static,
        template_shapes, template_dtypes, template_treedef
    )
    pop_fitness = fitness_concat[:M]
    off_fitness = fitness_concat[M:]

    # --- Selection ---
    selected = select_next_generation_vectorized(population, pop_fitness, offspring, off_fitness)

    # --- Metrics ---
    def binary_entropy(p):
        return - (p * jnp.log(p + 1e-8) + (1 - p) * jnp.log(1 - p + 1e-8))

    metrics = {
        # Fitness stats
        'best_fitness_parent': jnp.max(pop_fitness),
        'mean_fitness_parent': jnp.mean(pop_fitness),
        'var_fitness_parent': jnp.var(pop_fitness),
        'best_fitness_offspring': jnp.max(off_fitness),
        'mean_fitness_offspring': jnp.mean(off_fitness),
        'var_fitness_offspring': jnp.var(off_fitness),
        # Diversity
        'entropy_population': jnp.mean(binary_entropy(population)),
        # Gradient stats
        'grad_norm': jnp.linalg.norm(grads) / grads.size,
        'grad_var': jnp.var(grads),
        # Evolution pressure (offspring - parent mean)
        'evolution_force': jnp.mean(off_fitness) - jnp.mean(pop_fitness),
        # Placeholder for episode length (requires env info)
        # 'episode_length': jnp.nan,  
    }

    return new_key, selected, metrics

def dc_step_pmap(key, pop, sample_size, local_lr, eps, crossover_ratio,
                 runner_arrays, conf_static, template_info):
    new_key, new_pop, metrics = dc_step_with_evaluation_core(
        key, pop, sample_size, local_lr, eps, crossover_ratio,
        runner_arrays, conf_static, template_info
    )
    # metrics 转成固定结构（pmap 需要 pytree of arrays）
    metrics_out = {
        "best_fitness_parent": metrics["best_fitness_parent"],
        "mean_fitness_parent": metrics["mean_fitness_parent"],
        "best_fitness_offspring": metrics["best_fitness_offspring"],
        "mean_fitness_offspring": metrics["mean_fitness_offspring"],
        "entropy_population": metrics["entropy_population"],
        "grad_norm": metrics["grad_norm"],
        "grad_var": metrics["grad_var"],
        "evolution_force": metrics["evolution_force"],
    }
    return new_key, new_pop, metrics_out

def create_population_multi_device(config: DCConfig, n_devices: int) -> jnp.ndarray:
    """Create population distributed across devices."""
    pop_per_device = config.population_size // n_devices
    if config.population_size % n_devices != 0:
        print(f"Adjusting population_size from {config.population_size} to {pop_per_device * n_devices}")
        config.population_size = pop_per_device * n_devices
    
    # Shape: (n_devices, pop_per_device, dims)
    return jnp.full((n_devices, pop_per_device, config.dims), 0.5, dtype=jnp.float32)


def main(conf):
    conf = OmegaConf.merge({
        "seed": 0,
        "dims": 128,
        "population_size": 64,
        "sample_size": 8,
        "crossover_ratio": 2,
        "local_lr": 0.01,
        "total_generations": 100,
        "warmup_steps": 0,
        "save_every": 50
    }, conf)

    # Setup devices
    devices, n_devices, use_multi_device = setup_devices()
    
    dc_conf = DCConfig(
        seed=int(conf.seed),
        dims=int(conf.get("dims", 128)),
        population_size=int(conf.get("population_size", 64)),
        sample_size=int(conf.get("sample_size", 8)),
        crossover_ratio=int(conf.get("crossover_ratio", 2)),
        local_lr=float(conf.get("local_lr", 0.01)),
        eps=float(conf.get("eps", 1e-3)),
        total_generations=int(conf.get("total_generations", 100)),
        project_name=conf.get("project_name", "DC-vectorized"),
        run_name=conf.get("run_name", "DC-run-2"),
        warmup_steps=int(conf.get("warmup_steps", 0)),
        save_every=int(conf.get("save_every", 50))
    )
    # wandb.login(key="29dcdd0e7422a0225de2cb0c5e9dbd04ff6605f0")

    # Init wandb
    wandb.init(project=dc_conf.project_name, name=dc_conf.run_name, 
               config=OmegaConf.to_container(conf),
               mode="online"
        )
            #    settings=wandb.Settings(_disable_stats=True))

    # Initialize ES components
    es_conf = ESConfig()
    
    try:
        if "task" in conf:
            env = envs.get_environment(conf.task, **conf.get("task_conf", {}))
            env = envs.wrappers.EpisodeWrapper(env, 
                conf.get("episode_conf", {}).get("max_episode_length", 1000), 
                conf.get("episode_conf", {}).get("action_repeat", 1))
            env = envs.wrappers.VmapWrapper(env)
            es_conf = es_conf.replace(env_cls=env)

        if "network_type" in conf:
            network_cls = NETWORKS[conf.get("network_type")]
            network = network_cls(out_dims=es_conf.env_cls.action_size, 
                                neuron_dtype=es_conf.network_dtype, 
                                **conf.get("network_conf", {}))
            optim = optax.chain(
                optax.scale_by_adam(mu_dtype=es_conf.p_dtype),
                (optax.add_decayed_weights(es_conf.weight_decay) if es_conf.weight_decay > 0 else optax.identity()),
                optax.scale(-es_conf.lr)
            )
            es_conf = es_conf.replace(network_cls=network, optim_cls=optim)
    except Exception as e:
        print(f"Failed to create es_conf: {e}")

    # Initialize runner
    runner = None
    template_info = None
    
    if getattr(es_conf, "env_cls", None) is not None and getattr(es_conf, "network_cls", None) is not None:
        key = random.PRNGKey(dc_conf.seed)
        key_run, key_network_init = random.split(key)
        runner = _runner_init(key_run, key_network_init, es_conf)

        # Create parameter template for JIT compilation - simplified approach
        leaves, treedef = jax.tree_util.tree_flatten(runner.params)
        template_shapes = tuple(tuple(leaf.shape) for leaf in leaves)
        template_dtypes = tuple(str(jnp.result_type(leaf)) for leaf in leaves)
        template_treedef = treedef
        template_info = (template_shapes, template_dtypes, template_treedef)

        template_is_bool = tuple([True if getattr(leaf, "dtype", None) == jnp.bool_ else False for leaf in leaves])
        
        global flat_masks_to_pytree_jit
        flat_masks_to_pytree_jit = make_flat_masks_to_pytree_jit(template_treedef)
        
        # leaves, treedef = jax.tree_util.tree_flatten(runner.params)
        # template_shapes = [tuple(map(int, leaf.shape)) for leaf in leaves]
        # template_dtypes = [leaf.dtype if hasattr(leaf, "dtype") else jnp.asarray(leaf).dtype for leaf in leaves]
        # template_treedef = treedef
        # template_info = (template_shapes, template_dtypes, template_treedef)
        
        # Warmup if requested
        if dc_conf.warmup_steps > 0:
            print(f"Running {dc_conf.warmup_steps} warmup steps...")
            warm_pop = ec.PopulationState(
                network_params=ec._deterministic_bernoulli_parameter(runner.params, (es_conf.pop_size,)),
                network_states=es_conf.network_cls.initial_carry(runner.key, es_conf.pop_size),
                env_states=runner.env_reset_pool,
                fitness_totrew=jnp.zeros(es_conf.pop_size),
                fitness_sum=jnp.zeros(es_conf.pop_size),
                fitness_n=jnp.zeros(es_conf.pop_size, dtype=jnp.int32)
            )
            warm_pop, _ = jax.lax.scan(
                lambda p, x: (_evaluate_step(p, runner, es_conf), None), 
                warm_pop, None, length=dc_conf.warmup_steps
            )
            runner = runner.replace(
                normalizer_state=ec.running_statistics.update(
                    runner.normalizer_state, warm_pop.env_states.obs
                )
            )

        # Build runner_arrays and conf_static for later jitting/pmaping
        runner_arrays = make_runner_arrays(runner)
        conf_static = make_conf_static(es_conf)
        # If multi-device, replicate runner_arrays to devices
        if 'use_multi_device' in locals() and use_multi_device:
            try:
                runner_arrays = jax.device_put_replicated(runner_arrays, devices)
            except Exception:
                # fallback: keep on host, will be device_put when used
                print("Warning: runner_arrays not replicated, will transfer each step!")
                pass
        else:
            # single-device: put on default device
            try:
                runner_arrays = jax.device_put(runner_arrays)
            except Exception:
                print("Warning: runner_arrays not replicated, will transfer each step!")
                pass

    # Determine problem dimensions
    if runner is not None:
        leaves = jax.tree_util.tree_leaves(runner.params)
        total_size = int(sum([int(jnp.prod(jnp.array(x.shape))) for x in leaves]))
        dc_conf.dims = total_size
        print(f"Using real environment with {total_size} parameters")
    else:
        print("Using placeholder evaluation")

    # Create population
    if use_multi_device:
        population = create_population_multi_device(dc_conf, n_devices)
        keys = random.split(random.PRNGKey(dc_conf.seed), n_devices)
    else:
        population = jnp.full((dc_conf.population_size, dc_conf.dims), 0.5, dtype=jnp.float32)
        keys = random.PRNGKey(dc_conf.seed)

    print(f"Starting Vectorized DC training with {dc_conf.total_generations} generations...")
    print(f"Population size: {dc_conf.population_size}, Dimensions: {dc_conf.dims}")
    if use_multi_device:
        print(f"Using {n_devices} devices, {dc_conf.population_size // n_devices} individuals per device")

    # Training loop
    start_time = time.time()
    
    # JIT wrapper (runner_arrays/conf_static are pytrees but network/env apply are static)
    dc_step_with_evaluation_jit = jax.jit(
        dc_step_with_evaluation_core,
        static_argnums=(6, 7, 8)  # runner_arrays, conf_static, template_info are static
    )
    pmap_step = jax.pmap(
        dc_step_pmap,
        axis_name="devices",
        static_broadcasted_argnums=(2, 3, 4, 5, 7, 8),  # 这些不随设备变化
    )
    for gen in tqdm(range(dc_conf.total_generations), desc="DC generations"):
        if use_multi_device:
            # Multi-device execution (simplified for now)
            # new_keys = []
            # new_populations = []
            # all_metrics = []
            
            # for device_idx in range(n_devices):
            #     key = keys[device_idx]
            #     pop_device = population[device_idx]
                
            #     # new_key, new_pop, metrics = dc_step_with_evaluation(
            #     #     key, pop_device, dc_conf.sample_size, dc_conf.local_lr, dc_conf.eps,
            #     #     dc_conf.crossover_ratio, runner, es_conf, template_info,
            #     #     runner_arrays=runner_arrays, conf_static=conf_static
            #     # )

            #     new_key, new_pop, metrics = dc_step_with_evaluation_core(
            #         key, pop_device, dc_conf.sample_size, dc_conf.local_lr, dc_conf.eps,
            #         dc_conf.crossover_ratio, runner_arrays, conf_static, template_info
            #     )
                
            #     new_keys.append(new_key)
            #     new_populations.append(new_pop)
            #     all_metrics.append(metrics)
            
            # keys = jnp.array(new_keys)
            # population = jnp.array(new_populations)

            # # Aggregate metrics across devices. Metrics from each device are
            # # scalars (jax arrays); convert to float and reduce.
            # def _gather(keyname, reducer=max):
            #     vals = [float(m[keyname]) for m in all_metrics]
            #     if reducer is max:
            #         return max(vals)
            #     return float(jnp.mean(jnp.array(vals)))

            # best_parent = _gather('best_fitness_parent', reducer=max)
            # # compute mean across devices for parent mean
            # mean_parent = float(jnp.mean(jnp.array([float(m['mean_fitness_parent']) for m in all_metrics])))

            # best_off = _gather('best_fitness_offspring', reducer=max)
            # mean_off = float(jnp.mean(jnp.array([float(m['mean_fitness_offspring']) for m in all_metrics])))

            # # Additional aggregated metrics (mean across devices)
            # entropy_population = float(jnp.mean(jnp.array([float(m['entropy_population']) for m in all_metrics])))
            # grad_norm = float(jnp.mean(jnp.array([float(m['grad_norm']) for m in all_metrics])))
            # grad_var = float(jnp.mean(jnp.array([float(m['grad_var']) for m in all_metrics])))
            # evolution_force = float(jnp.mean(jnp.array([float(m['evolution_force']) for m in all_metrics])))
            keys, population, metrics = pmap_step(
                keys, population,
                dc_conf.sample_size, dc_conf.local_lr, dc_conf.eps,
                dc_conf.crossover_ratio, runner_arrays, conf_static, template_info
            )

            # metrics 是 dict of arrays，形状 [n_devices]
            # 聚合为 scalar（max 或 mean）
            def _gather(keyname, reducer=max):
                vals = metrics[keyname]  # shape [n_devices]
                if reducer is max:
                    return float(jnp.max(vals))
                else:
                    return float(jnp.mean(vals))

            best_parent = _gather("best_fitness_parent", reducer=max)
            mean_parent = _gather("mean_fitness_parent", reducer=jnp.mean)
            best_off = _gather("best_fitness_offspring", reducer=max)
            mean_off = _gather("mean_fitness_offspring", reducer=jnp.mean)
            entropy_population = _gather("entropy_population", reducer=jnp.mean)
            grad_norm = _gather("grad_norm", reducer=jnp.mean)
            grad_var = _gather("grad_var", reducer=jnp.mean)
            evolution_force = _gather("evolution_force", reducer=jnp.mean)
        else:
            # Single device execution
            # keys, population, metrics = dc_step_with_evaluation(
            #     keys, population, dc_conf.sample_size, dc_conf.local_lr, dc_conf.eps,
            #     dc_conf.crossover_ratio, runner, es_conf, template_info,
            #     runner_arrays=runner_arrays, conf_static=conf_static
            # )
            # Ensure we pass a single PRNGKey (shape (2,)) into the step.
            # keys may sometimes be an array of keys (e.g. from previous replication);
            # handle that by taking the first key when necessary.
            if hasattr(keys, 'shape') and len(getattr(keys, 'shape', ())) > 1:
                key_for_step = keys[0]
            else:
                key_for_step = keys

            new_key, new_pop, metrics = dc_step_with_evaluation_core(
                key_for_step, population, dc_conf.sample_size, dc_conf.population_size, dc_conf.local_lr, dc_conf.eps,
                dc_conf.crossover_ratio, runner_arrays, conf_static, template_info
            )

            # Update key and population for next generation
            keys = new_key
            population = new_pop

            # Single-device metrics (extract and convert to float)
            best_parent = float(metrics['best_fitness_parent'])
            mean_parent = float(metrics['mean_fitness_parent'])
            best_off = float(metrics['best_fitness_offspring'])
            mean_off = float(metrics['mean_fitness_offspring'])
            entropy_population = float(metrics['entropy_population'])
            grad_norm = float(metrics['grad_norm'])
            grad_var = float(metrics['grad_var'])
            evolution_force = float(metrics['evolution_force'])
            
        
        # Log metrics (align names with core function)
        wandb.log({
            "generation": int(gen),
            "best_fitness_parent": best_parent,
            "mean_fitness_parent": mean_parent,
            "best_fitness_offspring": best_off,
            "mean_fitness_offspring": mean_off,
            "entropy_population": entropy_population,
            "grad_norm": grad_norm,
            "grad_var": grad_var,
            "evolution_force": evolution_force,
        })

        # Checkpointing
        if runner is not None and dc_conf.save_every > 0 and ((gen + 1) % dc_conf.save_every == 0):
            conf.save_model_path = f"models/{dc_conf.project_name}/{dc_conf.run_name}/"
            fn = conf.save_model_path + str(gen + 1)
            save_obj_to_file(fn, dict(
                conf=conf,
                state=dict(
                    normalizer_state=runner.normalizer_state,
                    fixed_weights=runner.fixed_weights,
                    params=runner.params
                )
            ))
            try:
                wandb.save(fn)
            except Exception:
                pass

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Average time per generation: {total_time/dc_conf.total_generations:.3f} seconds")
    
    if use_multi_device:
        print(f"Multi-device execution with {n_devices} devices")

    wandb.finish()


if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    main(_config)
