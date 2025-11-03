import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_MODE"] = "online"
from functools import partial
from typing import Any, Dict, Tuple
import time
import builtins

import jax
import jax.numpy as jnp
import flax
import optax
from brax import envs
from brax.training.acme import running_statistics, specs
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

from networks import NETWORKS
from utils.functions import mean_weight_abs, finitemean, save_obj_to_file

jax.config.update("jax_default_prng_impl", "unsafe_rbg")
builtins.bfloat16 = jnp.dtype("bfloat16").type

@flax.struct.dataclass
class ESConfig:
    network_cls: Any = None
    optim_cls: Any = None
    env_cls: Any = None
    pop_size: int = 2560 # ! num_samples in EC
    lr: float = 0.15
    eps: float = 1e-3
    weight_decay: float = 0.
    warmup_steps: int = 0
    eval_size: int = 128
    action_dtype: Any = jnp.float32
    p_dtype: Any = jnp.float32
    network_dtype: Any = jnp.float32
    num_runners: int = 4 # ! num_individuals in 'runners' population, used for crossover
    crossover_ratio: float = 2 # n_offspring = n_runners * crossover_ratio
    crossover_method: str = "random-pick" # arithmetic, random-pick, logit-space, sample-based
    add_noise: bool = False # whether to add gaussian noise after crossover
    crossover_noise_std: float = 0.01 # std of gaussian noise added after crossover

@flax.struct.dataclass
class OneRunnerState:
    key: Any
    normalizer_state: running_statistics.RunningStatisticsState
    env_reset_pool: Any
    params: Any
    fixed_weights: Any
    opt_state: Any
    fitness: jnp.ndarray

@flax.struct.dataclass
class RunnersState:
    key: Any
    runners: OneRunnerState
    normalizer_state: Any

@flax.struct.dataclass
class PopulationState:
    network_params: Any
    network_states: Any
    env_states: Any
    fitness_totrew: jnp.ndarray
    fitness_sum: jnp.ndarray
    fitness_n: jnp.ndarray

def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    shape = x.shape
    x = x.ravel()
    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)

def _sample_bernoulli_parameter(key: Any, params: Any, sampling_dtype: Any, batch_size: Tuple = ()) -> Any:
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (*batch_size, *p.shape), sampling_dtype) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))
    return noise

def _deterministic_bernoulli_parameter(params: Any, batch_size: Tuple = ()) -> Any:
    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p > 0.5, (*batch_size, *p.shape)), params)

def _evaluate_step(pop: PopulationState, runner: OneRunnerState, conf: ESConfig) -> PopulationState:
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))
    obs_norm = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
    new_network_states, act = vmapped_apply({"params": pop.network_params, "fixed_weights": runner.fixed_weights}, pop.network_states, obs_norm)
    assert act.dtype == conf.network_dtype
    act = jnp.clip(act, -1, 1)
    if act.dtype != conf.action_dtype:
        act = jnp.where(jnp.isnan(act), 0, act).astype(conf.action_dtype)
    new_env_states = conf.env_cls.step(pop.env_states, act)
    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward
    new_fitness_sum = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n = jnp.where(new_env_states.done, pop.fitness_n + 1, pop.fitness_n)
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)
    def _where_done(x, y):
        done = new_env_states.done
        done = done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done, x, y)
    new_env_states = jax.tree_map(_where_done, runner.env_reset_pool, new_env_states)
    return pop.replace(
        network_states=new_network_states,
        env_states=new_env_states,
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n
    )

def _toy_evaluate_step(pop: PopulationState, runner: OneRunnerState, conf: ESConfig, use_fraction = 0.5) -> PopulationState:
    def _param_sum(params):
        leaves = jax.tree_util.tree_leaves(params)
        flat = jnp.concatenate([x.ravel() for x in leaves])
        N = flat.shape[0]
        k = int(N * use_fraction)
        return jnp.sum(flat[:k])
    fitness = jax.vmap(_param_sum)(pop.network_params)
    new_fitness_sum = pop.fitness_sum + fitness
    new_fitness_n = pop.fitness_n + 1
    new_env_states = pop.env_states
    return pop.replace(
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n,
        env_states=new_env_states
    )

@partial(jax.jit, static_argnums=(2,))
def _one_runner_init(key: Any, network_init_key: Any, conf: ESConfig) -> OneRunnerState:
    key, env_init_key = jax.random.split(key)
    env_reset_pool = conf.env_cls.reset(jax.random.split(env_init_key, conf.pop_size))
    network_variables = conf.network_cls.init(
        {"params": network_init_key, "fixed_weights": network_init_key},
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        env_reset_pool.obs
    )
    network_params = network_variables["params"]
    network_fixed_weights = network_variables["fixed_weights"]
    network_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5, conf.p_dtype), network_params)
    optim_state = conf.optim_cls.init(network_params)
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

@partial(jax.jit, static_argnums=(1,2,3))
def _one_runner_run(runner: OneRunnerState, conf: ESConfig, 
                    toy : bool = False, 
                    do_update : bool = False) -> Tuple[OneRunnerState, Dict, jnp.ndarray]:
    """
    Run the EC/eval step for a single OneRunnerState.

    Args:
        runner: OneRunnerState
        conf: ESConfig
        toy: bool, whether to use toy evaluation step
        do_update: bool, whether to perform parameter(rhos) update,
            if True, perform standard ES update,
            if False, only evaluate current params without update (for crossover evaluation).

    Returns:
        new_runner: OneRunnerState (params/opt_state/fitness updated; NORMALIZER NOT UPDATED)
        metrics: dict of scalar metrics (fitness, eval_fitness, sparsity, ...)
        obs_for_normalizer: array of shape (pop_size, obs_dim) to be used by the caller to update shared normalizer
    """

    metrics = {}
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)
    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype,
                                            (conf.pop_size - conf.eval_size,))
    eval_params = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size,))
    network_params = jax.tree_map(lambda train, eval: jnp.concatenate([train, eval], axis=0),
                                train_params, eval_params)
    split_idx = conf.pop_size - conf.eval_size
    def _split_fitness(x):
        return jnp.split(x, [split_idx, ])
    pop = PopulationState(
        network_params=network_params,
        network_states=conf.network_cls.initial_carry(carry_key, conf.pop_size),
        env_states=runner.env_reset_pool,
        fitness_totrew=jnp.zeros(conf.pop_size),
        fitness_sum=jnp.zeros(conf.pop_size),
        fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
    )
    if conf.warmup_steps > 0:
        pop, _ = jax.lax.scan(lambda p, x: (_evaluate_step(p, runner, conf), None),
                              pop, None, length=conf.warmup_steps)
        warmup_fitness, warmup_eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)
        metrics.update({
            "warmup_fitness": finitemean(warmup_fitness),
            "warmup_eval_fitness": finitemean(warmup_eval_fitness)
        })
        obs_for_normalizer = pop.env_states.obs
        pop = pop.replace(
            env_states=runner.env_reset_pool,
            fitness_totrew=jnp.zeros(conf.pop_size),
            fitness_sum=jnp.zeros(conf.pop_size),
            fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
        )
    else:
        obs_for_normalizer = jnp.zeros((0, conf.env_cls.observation_size), dtype=jnp.float32)
    if not toy:
        def _eval_stop_cond(p: PopulationState) -> jnp.ndarray:
            return ~jnp.all(p.fitness_n >= 1)
        pop = jax.lax.while_loop(_eval_stop_cond, lambda p: _evaluate_step(p, runner, conf), pop)
    else: 
        pop = jax.lax.while_loop(lambda p: ~jnp.all(p.fitness_n >= 1), 
                                 lambda p: _toy_evaluate_step(p, runner, conf), 
                                 pop)
    if conf.warmup_steps <= 0:
        obs_for_normalizer = pop.env_states.obs
    if hasattr(conf.network_cls, "carry_metrics"):
        metrics.update(conf.network_cls.carry_metrics(pop.network_states))
    fitness_all = pop.fitness_sum / pop.fitness_n
    fitness, eval_fitness = _split_fitness(fitness_all)
    mean_fitness = jnp.mean(fitness)
    mean_eval_fitness = jnp.mean(eval_fitness)
    metrics.update({
        "sample_fitness_distribution": fitness,
        "sample_eval_fitness_distribution": eval_fitness,
    })
    if do_update :
        weight = _centered_rank_transform(fitness)
        def _nes_grad(p, theta):
            w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)
            return -jnp.mean(w * (theta - p), axis=0)
        grads = jax.tree_map(
            lambda p, theta: _nes_grad(p, theta[:(conf.pop_size - conf.eval_size)]),
            runner.params, pop.network_params
        )
        updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
        new_params = optax.apply_updates(runner.params, updates)
        new_params = jax.tree_map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params)
    else:
        new_params = runner.params
        new_opt_state = runner.opt_state
    runner = runner.replace(
        params=new_params,
        opt_state=new_opt_state,
        fitness = mean_fitness
    )
    def _flatten_params_to_vec(p):
        leaves = jax.tree_util.tree_leaves(p)
        return jnp.concatenate([x.ravel() for x in leaves])
    flat_p = _flatten_params_to_vec(runner.params)
    N = flat_p.shape[0]
    k = N // 2
    p_t, p_o = flat_p[:k], flat_p[k:]
    def bernoulli_entropy(p):
        p = jnp.clip(p, 1e-6, 1 - 1e-6)
        return -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))
    metrics.update({
        "fitness": mean_fitness,
        "eval_fitness": mean_eval_fitness,
        "sparsity": mean_weight_abs(runner.params),
        "p_target_mean": jnp.mean(p_t),
        "p_other_mean": jnp.mean(p_o),
        "entropy_target_mean": jnp.mean(bernoulli_entropy(p_t)),
        "entropy_other_mean": jnp.mean(bernoulli_entropy(p_o)),
    })
    return runner, metrics, obs_for_normalizer

@partial(jax.jit, static_argnums=(1,2,3,4))
def _runners_run(runners_state: RunnersState, conf: ESConfig, 
                 update_normalizer : bool = True, 
                 toy : bool = True,
                 do_update : bool = True) -> Tuple[RunnersState, Dict]:
    num_runners = conf.num_runners
    key = runners_state.key
    keys = jax.random.split(key, num_runners + 1)
    new_master_key = keys[0]
    per_keys = keys[1:]
    runners_with_keys = runners_state.runners.replace(key=per_keys)
    updated_runners_batched, metrics_batched, obs_batched = jax.vmap(
        _one_runner_run, in_axes=(0, None, None, None))(runners_with_keys, conf, toy, do_update)
    combined_obs = jnp.reshape(obs_batched, (-1, obs_batched.shape[-1]))
    if update_normalizer:
        new_global_normalizer = running_statistics.update(runners_state.normalizer_state, combined_obs)
    else:
        new_global_normalizer = runners_state.normalizer_state
    def _broadcast_leaf(leaf):
        return jnp.broadcast_to(leaf, (num_runners,) + leaf.shape)
    batched_norm = jax.tree_util.tree_map(_broadcast_leaf, new_global_normalizer)
    updated_runners_batched = updated_runners_batched.replace(normalizer_state=batched_norm)
    new_runners_state = RunnersState(
        runners=updated_runners_batched,
        key=new_master_key,
        normalizer_state=new_global_normalizer
    )
    def _mean_metric(v):
        return jnp.mean(v, axis=0)
    aggregated_metrics = jax.tree_util.tree_map(_mean_metric, metrics_batched)
    try:
        sample_fitness_distribution = metrics_batched["sample_fitness_distribution"].reshape(-1)
        sample_eval_fitness_distribution = metrics_batched["sample_eval_fitness_distribution"].reshape(-1)
    except Exception:
        sample_fitness_distribution = jnp.array([], dtype=jnp.float32)
        sample_eval_fitness_distribution = jnp.array([], dtype=jnp.float32)
    runners_fitness_distribution = updated_runners_batched.fitness.reshape(-1)
    aggregated_metrics = dict(aggregated_metrics)
    aggregated_metrics["sample_fitness_distribution"] = sample_fitness_distribution
    aggregated_metrics["sample_eval_fitness_distribution"] = sample_eval_fitness_distribution
    aggregated_metrics["runners_fitness_distribution"] = runners_fitness_distribution
    return new_runners_state, aggregated_metrics

def arithmetic_crossover(parent1, parent2, key):
    alpha = jax.random.uniform(key, shape=())
    return jax.tree_map(lambda p1, p2: alpha * p1 + (1 - alpha) * p2, parent1, parent2)

def random_pick_crossover(parent1, parent2, key):
    def _crossover_gene(gene1, gene2, k):
        mask = jax.random.bernoulli(k, 0.5, gene1.shape)
        return jnp.where(mask, gene1, gene2)
    treedef = jax.tree_util.tree_structure(parent1)
    num_leaves = len(jax.tree_util.tree_leaves(parent1))
    keys = jax.random.split(key, num_leaves)
    keys_tree = jax.tree_util.tree_unflatten(treedef, list(keys))
    return jax.tree_map(_crossover_gene, parent1, parent2, keys_tree)

def logit_space_crossover(parent1, parent2, key):
    def _logit(p):
        p = jnp.clip(p, 1e-6, 1 - 1e-6)
        return jnp.log(p / (1 - p))
    def _sigmoid(x): return 1 / (1 + jnp.exp(-x))
    alpha = jax.random.uniform(key, shape=())
    return jax.tree_map(lambda p1, p2: _sigmoid(alpha * _logit(p1) + (1 - alpha) * _logit(p2)),
                        parent1, parent2)

def sample_based_crossover(parent1, parent2, key, sample_size=32):
    treedef = jax.tree_util.tree_structure(parent1)
    leaves1 = jax.tree_util.tree_leaves(parent1)
    leaves2 = jax.tree_util.tree_leaves(parent2)
    num_leaves = len(leaves1)
    if num_leaves == 0:
        return parent1
    keys = jax.random.split(key, 3 * num_leaves)
    k1s = keys[:num_leaves]
    k2s = keys[num_leaves:2 * num_leaves]
    kms = keys[2 * num_leaves:]
    new_leaves = []
    for i, (l1, l2) in enumerate(zip(leaves1, leaves2)):
        k1 = k1s[i]
        k2 = k2s[i]
        km = kms[i]
        s1 = jax.random.bernoulli(k1, l1, (sample_size,) + l1.shape).astype(jnp.float32)
        s2 = jax.random.bernoulli(k2, l2, (sample_size,) + l2.shape).astype(jnp.float32)
        mask = jax.random.bernoulli(km, 0.5, s1.shape)
        children = jnp.where(mask, s1, s2)
        new_p_leaf = jnp.mean(children, axis=0)
        new_leaves.append(new_p_leaf)
    new_p = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return new_p

def crossover_operator(parent1, parent2, key, method="arithmetic"):
    if method == "arithmetic":
        return arithmetic_crossover(parent1, parent2, key)
    elif method == "random-pick":
        return random_pick_crossover(parent1, parent2, key)
    elif method == "logit-space":
        return logit_space_crossover(parent1, parent2, key)
    elif method == "sample-based":
        return sample_based_crossover(parent1, parent2, key)
    else:
        raise ValueError(f"Unknown crossover method: {method}")

def _generate_offspring(runners: RunnersState, conf: ESConfig, master_key: Any):
    num_runners = conf.num_runners
    crossover_ratio = getattr(conf, "crossover_ratio", 1.0)
    num_offspring = int(num_runners * crossover_ratio)
    split_keys = jax.random.split(master_key, 2 * num_offspring + 1)
    new_master_key = split_keys[0]
    pair_keys = split_keys[1 : 1 + num_offspring]
    alpha_keys = split_keys[1 + num_offspring : ]
    parents_params = runners.runners.params
    def _pair_from_perm(key):
        perm = jax.random.permutation(key, num_runners)
        return jnp.array([perm[0], perm[1]],dtype=jnp.int32)
    parent_idx = jax.vmap(_pair_from_perm)(pair_keys)
    def _generate_single_child(pair_idx, k):
        p1 = jax.tree_map(lambda x: x[pair_idx[0]], parents_params)
        p2 = jax.tree_map(lambda x: x[pair_idx[1]], parents_params)
        return crossover_operator(p1, p2, k, method=getattr(conf, "crossover_method", "arithmetic"))
    offspring_params = jax.vmap(_generate_single_child, in_axes=(0,0))(parent_idx, alpha_keys)
    if conf.add_noise:
        sigma = getattr(conf, "crossover_noise_std", 0.01)
        treedef = jax.tree_util.tree_structure(offspring_params)
        leaves = jax.tree_util.tree_leaves(offspring_params)
        num_leaves = len(leaves)
        if num_leaves > 0:
            nkeys = jax.random.split(new_master_key, num_leaves + 1)
            new_master_key_final = nkeys[0]
            leaf_keys = nkeys[1:]
            new_leaves = []
            for lk, leaf in zip(leaf_keys, leaves):
                noise = jax.random.normal(lk, shape=leaf.shape) * sigma
                new_leaf = jnp.clip(leaf + noise, conf.eps, 1.0 - conf.eps)
                new_leaves.append(new_leaf)
            offspring_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
        else:
            new_master_key_final = new_master_key
        return offspring_params, new_master_key_final, parent_idx
    else:
        return offspring_params, new_master_key, parent_idx

@partial(jax.jit, static_argnums=(1,))
def _runners_crossover_selection(runners: RunnersState, conf: ESConfig) -> Tuple[RunnersState, Dict]:
    """
    Perform crossover between runners, evaluate offspring, and select top performers.
    """
    num_runners = conf.num_runners
    crossover_ratio = getattr(conf, "crossover_ratio", 1.0)
    num_offspring = int(num_runners * crossover_ratio)
    offspring_params, new_master_key, parent_idx = _generate_offspring(runners, conf, runners.key)
    template_parent_idx = parent_idx[:, 0]
    def _gather_from_parents(x):
        if hasattr(x, "ndim") and x.ndim > 0:
            return jnp.take(x, template_parent_idx, axis=0)
        return x
    offspring_template = jax.tree_map(_gather_from_parents, runners.runners)
    offspring_runners = offspring_template.replace(
        params = offspring_params,
        fitness = jnp.zeros(num_offspring, dtype=jnp.float32),
    )
    offspring_state = RunnersState(
        runners=offspring_runners,
        key=new_master_key,
        normalizer_state=runners.normalizer_state
    )
    offspring_conf = conf.replace(num_runners=num_offspring)
    evaluated_offspring_state, offspring_metrics = _runners_run(offspring_state, offspring_conf, update_normalizer=True, toy=True, do_update=False)
    evaluated_offspring_runners = evaluated_offspring_state.runners
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
    all_runners = jax.tree_map(_concat_safe, runners.runners, evaluated_offspring_runners)
    all_fitness = _concat_safe(runners.runners.fitness, evaluated_offspring_runners.fitness)
    top_indices = jnp.argsort(-all_fitness)[:num_runners]
    new_runners = jax.tree_map(lambda x: x[top_indices], all_runners)
    offspring_train_fitness = offspring_metrics["fitness"]
    offspring_eval_fitness = offspring_metrics["eval_fitness"]
    offspring_sparsity = offspring_metrics["sparsity"]
    mean_offspring_train = jnp.mean(offspring_train_fitness)
    mean_offspring_eval = jnp.mean(offspring_eval_fitness)
    best_offspring_train = jnp.max(offspring_train_fitness)
    best_offspring_eval = jnp.max(offspring_eval_fitness)
    mean_sparsity = jnp.mean(offspring_sparsity)
    def _mean_param_l2_diff(p_parent, p_child):
        diff = jnp.mean((p_parent - p_child) ** 2)
        return diff
    evolution_force = jax.tree_util.tree_reduce(
        lambda a, b: a + b,
        jax.tree_util.tree_map(
            _mean_param_l2_diff,
            jax.tree_map(
                lambda x: jnp.take(x, template_parent_idx, axis=0)
                if hasattr(x, "ndim") and x.ndim > 0 else x,
                runners.runners.params,
            ),
            offspring_params
        )
    )
    evolution_force = evolution_force / len(jax.tree_util.tree_leaves(runners.runners.params))
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
    parents_fitness_distribution = runners.runners.fitness.reshape(-1)
    offspring_fitness_distribution = evaluated_offspring_runners.fitness.reshape(-1)
    sample_offspring_fitness_distribution = offspring_metrics.get("sample_fitness_distribution", jnp.array([], dtype=jnp.float32))
    sample_offspring_eval_fitness_distribution = offspring_metrics.get("sample_eval_fitness_distribution", jnp.array([], dtype=jnp.float32))
    distributional_metrics = {
        "parents_fitness_distribution": parents_fitness_distribution,
        "offspring_fitness_distribution": offspring_fitness_distribution,
        "sample_offspring_fitness_distribution": sample_offspring_fitness_distribution,
        "sample_offspring_eval_fitness_distribution": sample_offspring_eval_fitness_distribution,
    }
    try:
        new_population_params_leaves = jax.tree_util.tree_leaves(new_runners.params)
        population_params_distribution = jnp.concatenate([x.ravel() for x in new_population_params_leaves])
    except Exception:
        population_params_distribution = jnp.array([], dtype=jnp.float32)
    distributional_metrics["population_params_distribution"] = population_params_distribution
    return runners.replace(runners=new_runners, key=new_master_key), metrics, distributional_metrics

def main(conf):
    conf = OmegaConf.merge({
        "seed": 0,
        "task": "humanoid",
        "task_conf": {},
        "episode_conf": {
            "max_episode_length": 1000,
            "action_repeat": 1
        },
        "total_generations": 1000,
        "save_every": 10000,
        "network_type": "ConnSNN",
        "network_conf": {},
        "es_conf": {},
        "cr_only": False
    }, conf)
    conf = OmegaConf.merge({
        "project_name": f"toy-DC-SNN-{conf.task}-1",
        "run_name": f"toy-DC-noEC- {conf.seed} {conf.network_type} {time.strftime('%H:%M %m-%d')}"
    }, conf)
    es_conf = ESConfig(**conf.es_conf)
    print(OmegaConf.to_yaml(conf))
    print(es_conf)
    env = envs.get_environment(conf.task, **conf.task_conf)
    env = envs.wrappers.EpisodeWrapper(env, conf.episode_conf.max_episode_length, conf.episode_conf.action_repeat)
    env = envs.wrappers.VmapWrapper(env)
    network_cls = NETWORKS[conf.network_type]
    network = network_cls(
        out_dims=env.action_size,
        neuron_dtype=es_conf.network_dtype,
        **conf.network_conf
    )
    optim = optax.chain(
        optax.scale_by_adam(mu_dtype=es_conf.p_dtype),
        (optax.add_decayed_weights(es_conf.weight_decay) if es_conf.weight_decay > 0 else optax.identity()),
        optax.scale(-es_conf.lr)
    )
    es_conf = es_conf.replace(
        network_cls=network,
        optim_cls=optim,
        env_cls=env
    )
    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    runners = _runners_init(key_run, key_network_init, es_conf)
    conf.save_model_path = "models/{}/{}/".format(conf.project_name, conf.run_name)
    conf.run_name = f'no-noise+{es_conf.crossover_method}-cr{getattr(es_conf, "crossover_ratio", 1.0)}-' + conf.run_name
    if "log_group" in conf:
        wandb.init(reinit=True, project=f"(G) E-SNN-{conf.task}", group=conf.log_group, name=str(conf.seed), config=OmegaConf.to_container(conf))
    else:
        wandb.init(reinit=True, project=conf.project_name, name=conf.run_name, config=OmegaConf.to_container(conf), mode="disabled")
    for step in tqdm(range(1, conf.total_generations + 1)):
        if (not conf.cr_only) and step % 10 == 1:
            runners, metrics = _runners_run(runners, es_conf)
            metrics = jax.device_get(metrics)
            print(f"EC Step {step} | Fitness {metrics['fitness']:.2f} | Eval Fitness {metrics['eval_fitness']:.2f} | Sparsity {metrics['sparsity']:.4f}"
                f" | p_target_mean {metrics['p_target_mean']:.4f} | p_other_mean {metrics['p_other_mean']:.4f} | "
                f"entropy_target_mean {metrics['entropy_target_mean']:.4f} | entropy_other_mean {metrics['entropy_other_mean']:.4f}")
        runners, crossover_metrics, distributional_metrics = _runners_crossover_selection(runners, es_conf)
        crossover_metrics = jax.device_get(crossover_metrics)
        distributional_metrics = jax.device_get(distributional_metrics)
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
        metrics = {}
        metrics['fitness'] = crossover_metrics['mean_offspring_train_fitness']
        metrics['eval_fitness'] = crossover_metrics['mean_offspring_eval_fitness']
        metrics['sparsity'] = crossover_metrics['mean_sparsity']
        metrics['p_target_mean'] = crossover_metrics['offspring_p_target_mean']
        metrics['p_other_mean'] = crossover_metrics['offspring_p_other_mean']
        metrics['entropy_target_mean'] = crossover_metrics['offspring_entropy_target_mean']
        metrics['entropy_other_mean'] = crossover_metrics['offspring_entropy_other_mean']
        wandb.log(metrics, step=step)
        try:
            wandb.log({
                "distribution_sample_offspring_fitness": wandb.Histogram(distributional_metrics['sample_offspring_fitness_distribution']),
                "distribution_sample_eval_offspring_fitness": wandb.Histogram(distributional_metrics['sample_offspring_eval_fitness_distribution']),
                "distribution_parents_fitness": wandb.Histogram(distributional_metrics['parents_fitness_distribution']),
                "distribution_offspring_fitness": wandb.Histogram(distributional_metrics['offspring_fitness_distribution']),
                "distribution_population_params": wandb.Histogram(distributional_metrics.get('population_params_distribution', [])),
            }, step=step)
        except Exception:
            pass
        if not (step % conf.save_every):
            fn = conf.save_model_path + str(step)
            save_obj_to_file(fn, dict(
                conf=conf,
                state=dict(
                    normalizer_state=runners.normalizer_state,
                    runners_params=runners.runners.params,
                    runners_opt_state=runners.runners.opt_state,
                    fixed_weights=runners.runners.fixed_weights,
                    runners_fitness=runners.runners.fitness,
                )
            ))
            wandb.save(fn)

if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    main(_config)
