from hmac import new
import os
# os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"  # or set to your custom W&B server
# os.environ["WANDB_MODE"] = "disabled"  # "offline" for offline logging
from functools import partial
import stat
from turtle import update
from typing import Any, Dict, Tuple
import time
import builtins

import jax
import jax.numpy as jnp

import flax
import optax

from brax import envs
from brax.training.acme import running_statistics
from brax.training.acme import specs

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import optuna

from networks import NETWORKS
from utils.functions import mean_weight_abs, finitemean, save_obj_to_file


# Use RBG generator for less memory consumption
# Default RNG needs 2*N extra memory, while RBG needs none, when generating array with size N
# https://jax.readthedocs.io/en/latest/jax.random.html
jax.config.update("jax_default_prng_impl", "unsafe_rbg")

# Hack for resolving bfloat16 pickling issue https://github.com/google/jax/issues/8505
builtins.bfloat16 = jnp.dtype("bfloat16").type


@flax.struct.dataclass
class ESConfig:
    # Network, optim & env class
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None

    # [Hyperparameters] ES
    pop_size:       int = 10240
    lr:           float = 0.15

    eps:          float = 1e-3

    weight_decay: float = 0.    # For sparsity regularization

    # [Hyperparameters] Warmup
    warmup_steps:   int = 0

    # [Hyperparameters] Eval
    eval_size:      int = 128

    # [Computing] Data types
    action_dtype: Any   = jnp.float32  # brax uses fp32

    p_dtype:       Any  = jnp.float32
    network_dtype: Any  = jnp.float32
    num_runners : int = 2

    #[Crossover & Selection]
    crossover_ratio: float = 1 # num_offsprings = crossover_ratio * num_parents

@flax.struct.dataclass
class OneRunnerState:
    key: Any
    # Normalizer
    normalizer_state: running_statistics.RunningStatisticsState
    # Env reset state pool
    env_reset_pool: Any
    # Network optimization
    params:        Any
    fixed_weights: Any
    opt_state:     Any
    fitness: jnp.ndarray #fitness for each rho individual

@flax.struct.dataclass
class RunnersState:
    key: Any
    runners: OneRunnerState # Set/Tree of OneRunnerState objects
    normalizer_state: Any

@flax.struct.dataclass
class PopulationState:
    # Network
    network_params: Any
    network_states: Any
    # Env
    env_states:     Any
    # Fitness
    fitness_totrew: jnp.ndarray
    fitness_sum:    jnp.ndarray
    fitness_n:      jnp.ndarray

def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Centered rank from: https://arxiv.org/pdf/1703.03864.pdf"""

    shape = x.shape
    x     = x.ravel()

    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)

def _sample_bernoulli_parameter(key: Any, params: Any, sampling_dtype: Any, batch_size: Tuple = ()) -> Any:
    """Sample parameters from Bernoulli distribution. """

    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)

    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (*batch_size, *p.shape), sampling_dtype) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))

    return noise

def _deterministic_bernoulli_parameter(params: Any, batch_size: Tuple = ()) -> Any:
    """Deterministic evaluation, using p > 0.5 as True, p <= 0.5 as False"""

    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p > 0.5, (*batch_size, *p.shape)), params)

# Evaluate the population for a single step
def _evaluate_step(pop: PopulationState, runner: OneRunnerState, conf: ESConfig) -> PopulationState:
    # step env
    # NOTE: vmapping apply for multiple set of parameters (broadcast fixed weights)
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))

    obs_norm                = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
    new_network_states, act = vmapped_apply({"params": pop.network_params, "fixed_weights": runner.fixed_weights}, pop.network_states, obs_norm)
    assert act.dtype == conf.network_dtype   # Sanity check, avoid silent promotion

    act = jnp.clip(act, -1, 1)  # brax do not clip actions internally.

    # NOTE: Cast type and avoid NaNs, set them to 0
    if act.dtype != conf.action_dtype:
        act = jnp.where(jnp.isnan(act), 0, act).astype(conf.action_dtype)

    new_env_states = conf.env_cls.step(pop.env_states, act)

    # calculate episodic rewards
    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward

    new_fitness_sum    = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n      = jnp.where(new_env_states.done, pop.fitness_n   + 1,                  pop.fitness_n)
    # clear tot rew
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)

    # reset done envs
    # Reference: brax / envs / wrapper.py
    def _where_done(x, y):
        done = new_env_states.done
        done = done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done, x, y)

    new_env_states = jax.tree_map(_where_done, runner.env_reset_pool, new_env_states)

    return pop.replace(
        # Network
        network_states=new_network_states,
        # Env
        env_states=new_env_states,
        # Fitness
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n
    )

def _toy_evaluate_step(pop: PopulationState, runner: OneRunnerState, conf: ESConfig, use_fraction = 0.5) -> PopulationState:
    """A toy version of _evaluate_step that skips real env interaction."""

    # Fitness = sum of all network parameters (toy metric)
    # Here we compute one scalar per individual
    def _param_sum(params):
        leaves = jax.tree_util.tree_leaves(params)
        flat = jnp.concatenate([x.ravel() for x in leaves])
        N = flat.shape[0]
        k = int(N * use_fraction)
        return jnp.sum(flat[:k])

    # Compute fitness for each network_params in population
    fitness = jax.vmap(_param_sum)(pop.network_params)
    

    # Update PopulationState like original structure
    new_fitness_sum = pop.fitness_sum + fitness
    new_fitness_n   = pop.fitness_n + 1

    # Dummy update to env_states (keep structure consistent)
    new_env_states = pop.env_states

    return pop.replace(
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n,
        env_states=new_env_states
    )

@partial(jax.jit, static_argnums=(2,))
def _one_runner_init(key: Any, network_init_key: Any, conf: ESConfig) -> OneRunnerState:
    # split run keys for initializing env
    key, env_init_key = jax.random.split(key)

    # init env
    env_reset_pool = conf.env_cls.reset(jax.random.split(env_init_key, conf.pop_size))

    # init network params + opt state
    network_variables = conf.network_cls.init(
        {"params": network_init_key, "fixed_weights": network_init_key},
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        env_reset_pool.obs
    )
    network_params = network_variables["params"]
    network_fixed_weights = network_variables["fixed_weights"]

    # set params to p=0.5 Bernoulli distribution
    network_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5, conf.p_dtype), network_params)
    optim_state = conf.optim_cls.init(network_params)

    # runner state
    runner = OneRunnerState(
        key=key,
        normalizer_state=running_statistics.init_state(specs.Array((conf.env_cls.observation_size, ), jnp.float32)),
        env_reset_pool=env_reset_pool,

        params=network_params,
        fixed_weights=network_fixed_weights,
        opt_state=optim_state,
        fitness = jnp.zeros(1,dtype = jnp.float32)
    )
    return runner

@partial(jax.jit, static_argnums=(2,))
def _runners_init(master_key:Any, network_init_key:Any, conf:ESConfig):
    num_runners = conf.num_runners
    keys = jax.random.split(master_key, num_runners + 1)
    master_key = keys[0]
    subkeys = keys[1:]
    network_keys = jax.random.split(network_init_key, num_runners)
    runners_batched = jax.vmap(_one_runner_init, in_axes = (0,0,None))(subkeys, network_keys, conf)
    normalizer = running_statistics.init_state(specs.Array((conf.env_cls.observation_size,), jnp.float32))
    return RunnersState(key=master_key, runners=runners_batched, normalizer_state=normalizer)

@partial(jax.jit, static_argnums=(1,2))
def _one_runner_run(runner: OneRunnerState, conf: ESConfig, toy : bool = False) -> Tuple[OneRunnerState, Dict, jnp.ndarray]:
    """
    Run the ES/eval step for a single OneRunnerState.

    Returns:
        new_runner: OneRunnerState (params/opt_state/fitness updated; NORMALIZER NOT UPDATED)
        metrics: dict of scalar metrics (fitness, eval_fitness, sparsity, ...)
        obs_for_normalizer: array of shape (pop_size, obs_dim) to be used by the caller to update shared normalizer
    """
    metrics = {}

    # ---- key split & keep new runner.key ----
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)

    # ---- sample train & eval parameters from bernoulli p (runner.params) ----
    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype,
                                               (conf.pop_size - conf.eval_size,))
    eval_params = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size,))
    # concat into network_params of shape (pop_size, ...)
    network_params = jax.tree_map(lambda train, eval: jnp.concatenate([train, eval], axis=0),
                                  train_params, eval_params)

    # helper to split fitness arrays into [train, eval]
    def _split_fitness(x):
        return jnp.split(x, [conf.pop_size - conf.eval_size, ])

    # ---- initialize PopulationState (for this runner only) ----
    pop = PopulationState(
        network_params=network_params,
        network_states=conf.network_cls.initial_carry(carry_key, conf.pop_size),
        env_states=runner.env_reset_pool,
        fitness_totrew=jnp.zeros(conf.pop_size),
        fitness_sum=jnp.zeros(conf.pop_size),
        fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
    )

    # ---- warmup (if configured) ----
    if conf.warmup_steps > 0:
        pop, _ = jax.lax.scan(lambda p, x: (_evaluate_step(p, runner, conf), None),
                              pop, None, length=conf.warmup_steps)

        warmup_fitness, warmup_eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)
        metrics.update({
            "warmup_fitness": finitemean(warmup_fitness),
            "warmup_eval_fitness": finitemean(warmup_eval_fitness)
        })

        # Instead of updating shared normalizer here, collect obs for external aggregation:
        obs_for_normalizer = pop.env_states.obs  # shape (pop_size, obs_dim)

        # Reset envs + Clear fitness (same as original)
        pop = pop.replace(
            env_states=runner.env_reset_pool,
            fitness_totrew=jnp.zeros(conf.pop_size),
            fitness_sum=jnp.zeros(conf.pop_size),
            fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
        )
    else:
        # placeholder; if no warmup, we'll set obs_for_normalizer after evaluation
        obs_for_normalizer = jnp.zeros((0, conf.env_cls.observation_size), dtype=jnp.float32)

    # ---- evaluate until each sampled env finishes at least once ----
    if not toy:
        def _eval_stop_cond(p: PopulationState) -> jnp.ndarray:
            return ~jnp.all(p.fitness_n >= 1)

        pop = jax.lax.while_loop(_eval_stop_cond, lambda p: _evaluate_step(p, runner, conf), pop)
    
    else: 
        pop = jax.lax.while_loop(lambda p: ~jnp.all(p.fitness_n >= 1), 
                                 lambda p: _toy_evaluate_step(p, runner, conf), 
                                 pop)

    # If there was no warmup, we use terminal states to update normalizer
    if conf.warmup_steps <= 0:
        obs_for_normalizer = pop.env_states.obs  # shape (pop_size, obs_dim)

    # ---- optional carry metrics from network ----
    if hasattr(conf.network_cls, "carry_metrics"):
        metrics.update(conf.network_cls.carry_metrics(pop.network_states))

    # ---- compute fitness arrays ----
    fitness, eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)

    # ---- compute NES gradient using centered ranks (same as original) ----
    weight = _centered_rank_transform(fitness)

    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)
        return -jnp.mean(w * (theta - p), axis=0)

    grads = jax.tree_map(
        lambda p, theta: _nes_grad(p, theta[:(conf.pop_size - conf.eval_size)]),
        runner.params, pop.network_params
    )

    # ---- optimizer update (optax style) ----
    updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)

    # ---- clip params to bernoulli valid range ----
    new_params = jax.tree_map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params)

    mean_fitness = jnp.mean(fitness)
    mean_eval_fitness = jnp.mean(eval_fitness)

    runner = runner.replace(
        params=new_params,
        opt_state=new_opt_state,
        fitness=mean_fitness
    )

    def _flatten_params_to_vec(p):
        leaves = jax.tree_util.tree_leaves(p)
        return jnp.concatenate([x.ravel() for x in leaves])

    # ... 在 _one_runner_run() 里，更新完 new_params / metrics 之后，追加：
    flat_p = _flatten_params_to_vec(runner.params)  # 或 new_params，取你想看的时点
    N = flat_p.shape[0]
    k = N // 2
    p_t, p_o = flat_p[:k], flat_p[k:]

    def bernoulli_entropy(p):
        p = jnp.clip(p, 1e-6, 1 - 1e-6)
        return -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))

    # ---- populate metrics same as original ----
    metrics.update({
        "fitness": mean_fitness,
        "eval_fitness": mean_eval_fitness,
        "sparsity": mean_weight_abs(new_params),

        "p_target_mean": jnp.mean(p_t),
        "p_other_mean": jnp.mean(p_o),
        "entropy_target_mean": jnp.mean(bernoulli_entropy(p_t)),
        "entropy_other_mean": jnp.mean(bernoulli_entropy(p_o)),

    })

    # Return: updated runner (note: normalizer_state not changed here),
    # metrics (dict) and obs_for_normalizer (pop_size, obs_dim)
    return runner, metrics, obs_for_normalizer

@partial(jax.jit, static_argnums=(1,2))
def _runners_run(runners_state: RunnersState, conf: ESConfig, update_normalizer : bool = True, toy : bool = True) -> Tuple[RunnersState, Dict]:
    """
    Run all runners in parallel:
      - split master key -> per-runner keys
      - call vmap(_one_runner_run) to run each runner
      - aggregate obs across runners and update shared normalizer once
      - broadcast new shared normalizer to per-runner normalizer_state
    Returns:
      new_runners_state, aggregated_metrics (means across runners)
    """
    num_runners = conf.num_runners
    key = runners_state.key

    # 1) split master key; keep new master key
    keys = jax.random.split(key, num_runners + 1)
    new_master_key = keys[0]
    per_keys = keys[1:]

    # 2) override per-runner key in the batched runners pytree
    #    runners_state.runners is a batched dataclass; replace its 'key' field
    runners_with_keys = runners_state.runners.replace(key=per_keys)

    # 3) vmap run
    #    _one_runner_run returns (runner, metrics, obs_for_normalizer) for each runner,
    #    so vmap returns triples with leading axis = num_runners.
    updated_runners_batched, metrics_batched, obs_batched = jax.vmap(
        _one_runner_run, in_axes=(0, None, None))(runners_with_keys, conf, toy)

    # 4) aggregate obs_for_normalizer across runners -> shape (num_runners * pop_size, obs_dim)
    #    obs_batched shape usually (num_runners, pop_size, obs_dim)
    #    if warmup used obs_for_normalizer may equal warmup-pop.obs; still shape (num_runners, pop_size, obs_dim)
    # If somehow obs_batched is empty for some runner (shape starting with 0), reshape handles OK
    combined_obs = jnp.reshape(obs_batched, (-1, obs_batched.shape[-1]))

    # 5) update global normalizer once using combined observation pool
    if update_normalizer:
        new_global_normalizer = running_statistics.update(runners_state.normalizer_state, combined_obs)
    else:
        new_global_normalizer = runners_state.normalizer_state

    # 6) broadcast the new global normalizer to per-runner normalizer_state fields
    #    we need a batched normalizer with leading axis = num_runners
    def _broadcast_leaf(leaf):
        # leaf shape like (obs_dim,) or scalar; broadcast to (num_runners, ...) 
        return jnp.broadcast_to(leaf, (num_runners,) + leaf.shape)

    batched_norm = jax.tree_util.tree_map(_broadcast_leaf, new_global_normalizer)

    # updated_runners_batched is a batched dataclass (OneRunnerState with leading axes).
    # replace their per-runner normalizer_state with the batched shared one
    updated_runners_batched = updated_runners_batched.replace(normalizer_state=batched_norm)

    # 7) assemble new RunnersState
    new_runners_state = RunnersState(
        runners=updated_runners_batched,
        key=new_master_key,
        normalizer_state=new_global_normalizer
    )

    # 8) aggregate metrics across runners (take mean for each scalar metric)
    # metrics_batched is a pytree dict where each value is an array with leading axis num_runners
    def _mean_metric(v):
        # if v is array shaped (num_runners, ...), average axis 0
        return jnp.mean(v, axis=0)

    aggregated_metrics = jax.tree_util.tree_map(_mean_metric, metrics_batched)

    return new_runners_state, aggregated_metrics

def arithmetic_crossover(parent1, parent2, key):
    """Arithmetic crossover: linear interpolation between two parents."""
    alpha = jax.random.uniform(key, shape=())
    return jax.tree_map(lambda p1, p2: alpha * p1 + (1 - alpha) * p2, parent1, parent2)


def crossover_operator(parent1, parent2, key, method="arithmetic"):
    """Select crossover operator based on method."""
    if method == "arithmetic":
        return arithmetic_crossover(parent1, parent2, key)
    else:
        raise ValueError(f"Unknown crossover method: {method}")

def _generate_offspring(runners: RunnersState, conf: ESConfig, master_key: Any):
    """
    Generate offspring runner parameters via crossover.
    Each offspring = crossover(parent1, parent2).
    """
    num_runners = conf.num_runners
    crossover_ratio = getattr(conf, "crossover_ratio", 1.0)
    num_offspring = int(num_runners * crossover_ratio)

    # split key
    split_keys = jax.random.split(master_key, 2 * num_offspring + 1)
    new_master_key = split_keys[0]

    # sample parent indices and keys for alpha if needed
    pair_keys = split_keys[1 : 1 + num_offspring]  # or use another split if desired
    alpha_keys = split_keys[1 + num_offspring : ]

    parents_params = runners.runners.params

    def _pair_from_perm(key):
        perm = jax.random.permutation(key, num_runners)
        return jnp.array([perm[0], perm[1]],dtype=jnp.int32)
    
    parent_idx = jax.vmap(_pair_from_perm)(pair_keys)

    def _gather_parent(params, idx_vec):
        return jax.vmap(lambda i : jnp.take(params, i, axis=0))(idx_vec)

    def _generate_single_child(pair_idx, k):
        p1 = jax.tree_map(lambda x: x[pair_idx[0]], parents_params)
        p2 = jax.tree_map(lambda x: x[pair_idx[1]], parents_params)
        return crossover_operator(p1, p2, k, method=getattr(conf, "crossover_method", "arithmetic"))
    
    offspring_params = jax.vmap(_generate_single_child, in_axes=(0,0))(parent_idx, alpha_keys)
    return offspring_params, new_master_key

@partial(jax.jit, static_argnums=(1,)) #FIXME: maybe arg 0 could be donated
def _runners_crossover_selection(runners: RunnersState, conf: ESConfig) -> Tuple[RunnersState, Dict]:
    """
    Perform arithmetic crossover between runners, evaluate offspring, and select top performers.
    """
    num_runners = conf.num_runners
    crossover_ratio = getattr(conf, "crossover_ratio", 1.0)
    num_offspring = int(num_runners * crossover_ratio)
   

    # 1) Generate offspring params:
    offspring_params, new_master_key = _generate_offspring(runners, conf, runners.key)

    # 2) Clone runner templates
    offspring_template = jax.tree_map(lambda x: x[:num_offspring], runners.runners)
    offspring_runners = offspring_template.replace(
        params = offspring_params,
        fitness = jnp.zeros(num_offspring, dtype=jnp.float32),
    )

    # Why don't I use _runner_init to init the offspring runners?
    # Because I want to keep the fixed_weights/opt_state/normalizer_state of parents

    # NOTE: Since I manually created offspring runners, 
    # I have to manually create the new RunnersState object to ensure that it could be passed to _runners_run
    offspring_state = RunnersState(
        runners=offspring_runners,
        key=new_master_key,                    # single global key
        normalizer_state=runners.normalizer_state  # share parent's normalizer (or copy)
    )

    # 3) Evaluate offspring runners
    evaluated_offspring_state, offspring_metrics = _runners_run(offspring_state, conf, update_normalizer=False)
    evaluated_offspring_runners = evaluated_offspring_state.runners  # get batched OneRunnerState

    # 4) Combine parents and offspring
    def _concat_safe(p, c):
        if isinstance(p, jnp.ndarray) and isinstance(c, jnp.ndarray):
            if p.ndim == c.ndim:
                return jnp.concatenate([p, c], axis=0)
            elif p.ndim < c.ndim:
                p = jnp.expand_dims(p, axis=-1)
            elif c.ndim < p.ndim:
                c = jnp.expand_dims(c, axis=-1)
            return jnp.concatenate([p, c], axis=0)
        else:
            return c  
        
    all_runners = jax.tree_map(_concat_safe,
                               runners.runners, evaluated_offspring_runners)
    all_fitness = _concat_safe(runners.runners.fitness, evaluated_offspring_runners.fitness)

    # 5) Select top individuals
    top_indices = jnp.argsort(-all_fitness)[:num_runners]
    new_runners = jax.tree_map(lambda x: x[top_indices], all_runners)

    # 6) Prepare metrics
    # Extract offspring fitness arrays
    offspring_train_fitness = offspring_metrics["fitness"]            # (num_offspring,)
    offspring_eval_fitness  = offspring_metrics["eval_fitness"]       # (num_offspring,)
    offspring_sparsity      = offspring_metrics["sparsity"]           # (num_offspring,)
    
    # Compute basic stats
    mean_offspring_train = jnp.mean(offspring_train_fitness)
    mean_offspring_eval  = jnp.mean(offspring_eval_fitness)
    best_offspring_train = jnp.max(offspring_train_fitness)
    best_offspring_eval  = jnp.max(offspring_eval_fitness)
    mean_sparsity        = jnp.mean(offspring_sparsity)
    
    # Evolutionary force: how much offspring diverged from parents
    # (average L2 distance between parent params and offspring params)
    def _mean_param_l2_diff(p_parent, p_child):
        diff = jnp.mean((p_parent - p_child) ** 2)
        return diff

    evolution_force = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(
            _mean_param_l2_diff,
            jax.tree_map(lambda x: x[:num_offspring], runners.runners.params),
            offspring_params
        )
    )
    evolution_force = evolution_force / len(jax.tree_util.tree_leaves(runners.runners.params))
    # Compose metrics dict

    offspring_p_target_mean = offspring_metrics["p_target_mean"]
    offspring_p_other_mean = offspring_metrics["p_other_mean"]
    offspring_entropy_target_mean = offspring_metrics["entropy_target_mean"]
    offspring_entropy_other_mean = offspring_metrics["entropy_other_mean"]

    metrics = {
        "mean_parent_fitness": jnp.mean(runners.runners.fitness),
        "mean_offspring_train_fitness": mean_offspring_train,
        "mean_offspring_eval_fitness": mean_offspring_eval,
        "best_offspring_train_fitness": best_offspring_train,
        "best_offspring_eval_fitness": best_offspring_eval,
        "mean_sparsity": mean_sparsity,
        "fitness_gain_mean": mean_offspring_train - jnp.mean(runners.runners.fitness),
        "fitness_gain_best": best_offspring_train - jnp.max(runners.runners.fitness),
        "params_evolution_force": evolution_force,
        "crossover_ratio": crossover_ratio,

        "offspring_p_target_mean": offspring_p_target_mean,
        "offspring_p_other_mean": offspring_p_other_mean,
        "offspring_entropy_target_mean": offspring_entropy_target_mean,
        "offspring_entropy_other_mean": offspring_entropy_other_mean,
    }

    return runners.replace(runners=new_runners, key=new_master_key), metrics

def main(conf):
    conf = OmegaConf.merge({
        # Task
        "seed": 0,
        "task": "humanoid",
        "task_conf": {
        },
        "episode_conf": {
            "max_episode_length": 1000,
            "action_repeat": 1
        },

        # Train & Checkpointing
        "total_generations": 1000,
        "save_every": 10000,

        # Network
        "network_type": "ConnSNN",
        "network_conf": {},

        # ES hyperparameter (see ESConfig)
        "es_conf": {}
    }, conf)
    # Naming
    conf = OmegaConf.merge({
        "project_name": f"toy-DC-SNN-{conf.task}",
        "run_name":     f"toy-DC-noCR {conf.seed} {conf.network_type} {time.strftime('%H:%M %m-%d')}"
    }, conf)
    # ES Config
    es_conf = ESConfig(**conf.es_conf)

    print(OmegaConf.to_yaml(conf))
    print(es_conf)

    # create env cls
    env = envs.get_environment(conf.task, **conf.task_conf)
    env = envs.wrappers.EpisodeWrapper(env, conf.episode_conf.max_episode_length, conf.episode_conf.action_repeat)
    env = envs.wrappers.VmapWrapper(env)

    # create network cls
    network_cls = NETWORKS[conf.network_type]
    network     = network_cls(
        out_dims=env.action_size,
        neuron_dtype=es_conf.network_dtype,
        **conf.network_conf
    )

    # create optim cls
    optim = optax.chain(
        optax.scale_by_adam(mu_dtype=es_conf.p_dtype),
        (optax.add_decayed_weights(es_conf.weight_decay) if es_conf.weight_decay > 0 else optax.identity()),
        optax.scale(-es_conf.lr)
    )

    # [initialize]
    # initialize cls in es conf
    es_conf = es_conf.replace(
        network_cls=network,
        optim_cls=optim,
        env_cls=env
    )

    # runner state
    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    # runner = _runner_init(key_run, key_network_init, es_conf)
    runners = _runners_init(key_run, key_network_init, es_conf)
    # save model path
    conf.save_model_path = "models/{}/{}/".format(conf.project_name, conf.run_name)

    # wandb
    if "log_group" in conf:
        wandb.init(reinit=True, project=f"(G) E-SNN-{conf.task}", group=conf.log_group, name=str(conf.seed), config=OmegaConf.to_container(conf))
    else:
        wandb.init(reinit=True, project=conf.project_name, name=conf.run_name, config=OmegaConf.to_container(conf),
                   mode = "online")

    # run
    for step in tqdm(range(1, conf.total_generations + 1)):
        # ES Step
        if step == 1:
            runners, metrics = _runners_run(runners, es_conf)
            metrics = jax.device_get(metrics)
            wandb.log(metrics, step=step)
            print(f"EC Step {step} | Fitness {metrics['fitness']:.2f} | Eval Fitness {metrics['eval_fitness']:.2f} | Sparsity {metrics['sparsity']:.4f}"
                f" | p_target_mean {metrics['p_target_mean']:.4f} | p_other_mean {metrics['p_other_mean']:.4f} | "
                f"entropy_target_mean {metrics['entropy_target_mean']:.4f} | entropy_other_mean {metrics['entropy_other_mean']:.4f}")

        # Crossover & Selection

        runners, crossover_metrics = _runners_crossover_selection(runners, es_conf)
        crossover_metrics = jax.device_get(crossover_metrics)
        wandb.log(crossover_metrics, step=step)
        print(f"CR Step {step} | Mean Parent Fitness {crossover_metrics['mean_parent_fitness']:.2f} | "
              f"Mean Offspring Fitness (Train) {crossover_metrics['mean_offspring_train_fitness']:.2f} | "
              f"Best Offspring Fitness (Train) {crossover_metrics['best_offspring_train_fitness']:.2f} | "
              f"Mean Offspring Fitness (Eval) {crossover_metrics['mean_offspring_eval_fitness']:.2f} | "
              f"Best Offspring Fitness (Eval) {crossover_metrics['best_offspring_eval_fitness']:.2f} | "
              f"Sparsity {crossover_metrics['mean_sparsity']:.4f} | "
              f"Fitness Gain (Mean) {crossover_metrics['fitness_gain_mean']:.2f} | "
              f"Fitness Gain (Best) {crossover_metrics['fitness_gain_best']:.2f} | "
              f"Evolution Force {crossover_metrics['params_evolution_force']:.6f}"
              f" | Offspring p_target_mean {crossover_metrics['offspring_p_target_mean']:.4f} | "
              f"Offspring p_other_mean {crossover_metrics['offspring_p_other_mean']:.4f} | "
              f"Offspring entropy_target_mean {crossover_metrics['offspring_entropy_target_mean']:.4f} |"
            f" Offspring entropy_other_mean {crossover_metrics['offspring_entropy_other_mean']:.4f}")
        
        if not (step % conf.save_every):
            fn = conf.save_model_path + str(step)
            save_obj_to_file(fn, dict(
                conf=conf,
                state=dict(
                    normalizer_state=runners.normalizer_state,
                    runners_params = runners.runners.params,
                    runners_opt_state = runners.runners.opt_state,
                    fixed_weights=runners.runners.fixed_weights,
                    runners_fitness = runners.runners.fitness,
                )
            ))
            wandb.save(fn)


def sweep(seed: int, conf_override: OmegaConf):
    def _objective(trial: optuna.Trial):
        conf = OmegaConf.merge(conf_override, {
            "seed": seed * 1000 + trial.number,

            "project_name": f"E-SNN-sweep",

            "es_conf": {
                "lr":           trial.suggest_categorical("lr",  [0.01, 0.05, 0.1, 0.15, 0.2]),
                "eps":          trial.suggest_categorical("eps", [1e-4, 1e-3, 0.01, 0.1, 0.2]),
            },
            "network_conf": {
                "num_neurons":  trial.suggest_categorical("num_neurons", [128, 256]),
            }
        })

        metrics = main(conf)
        return metrics["eval_fitness"]

    optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed)).optimize(_objective)


if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    if hasattr(_config, "sweep"):
        sweep(_config.sweep, _config)
    else:
        main(_config)
