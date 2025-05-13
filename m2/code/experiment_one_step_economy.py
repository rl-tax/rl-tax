import numpy as np
import random
from copy import deepcopy
from ai_economist import foundation


def random_planner_policy(obs, planner):
    """
    A random policy for the planner. If planner is in multi-action mode, returns a list
    of random actions, one for each sub-action space.
    """
    if planner.multi_action_mode:
        dims = planner.action_spaces  # a numpy array of sub-action dimensions
        return [np.random.randint(0, d) for d in dims]
    else:
        n_actions = planner.action_spaces
        return np.random.randint(0, n_actions)

def noop_policy(obs, planner):
    """
    A NO-OP policy that returns zeros.
    """
    if planner.multi_action_mode:
        dims = planner.action_spaces
        return [0 for _ in dims]
    else:
        return 0


def run_episode(env, planner_policy):
    """
    Run one full episode on the environment.
    For one-step-economy, the episode is two steps.
    
    Args:
      env: an environment instance.
      planner_policy: a function that takes (obs, planner) and returns an action.
    
    Returns:
      total_planner_reward, metrics from the completed episode.
    """
    obs = env.reset()
    total_planner_reward = 0.0
    done = {"__all__": False}
    # Retrieve the planner agent once.
    planner_idx = env.world.planner.idx
    planner = env.get_agent(planner_idx)
    
    while not done["__all__"]:
        planner_obs = obs[planner_idx]
        action = planner_policy(planner_obs, planner)
        # Ensure that the action is a list if the planner is in multi-action mode.
        if planner.multi_action_mode and not isinstance(action, (list, tuple)):
            # If a scalar is returned by mistake, replicate it.
            action = [action] * planner._unique_actions
        actions = {planner_idx: action}
        obs, rew, done, info = env.step(actions)

        total_planner_reward += rew[planner_idx]
    return total_planner_reward, env.previous_episode_metrics

tax_scheme_configs = {
    "RL": {
        "tax_component_params": {
            "disable_taxes": False,
            "tax_model": "model_wrapper",
            "period": 2,
        },
        "planner_policy": random_planner_policy
    },
    "FreeMarket": {
        "tax_component_params": {
            "disable_taxes": True,
            "tax_model": "model_wrapper",
            "period": 2,
        },
        "planner_policy": noop_policy
    },
    "USFederal": {
        "tax_component_params": {
            "disable_taxes": False,
            "tax_model": "us-federal-single-filer-2018-scaled",
            "period": 2,
        },
        "planner_policy": noop_policy
    },
}


base_env_config = {
    "n_agents": 4,
    "world_size": [25, 25],
    "episode_length": 2,
    "multi_action_mode_agents": False,
    "multi_action_mode_planner": True,
    "scenario_name": "one-step-economy",
}

def get_components_config(tax_component_params):
    return [
        {"SimpleLabor": {}},
        {"PeriodicBracketTax": tax_component_params},
    ]


num_episodes = 100
results = {}

for scheme_name, scheme_info in tax_scheme_configs.items():
    print("===== Running Tax Scheme:", scheme_name, "=====")
    env_config = deepcopy(base_env_config)
    env_config["components"] = get_components_config(scheme_info["tax_component_params"])
    env = foundation.make_env_instance(**env_config)
    
    scheme_rewards = []
    all_metrics = []
    for ep in range(num_episodes):
        # Call run_episode with the proper policy (which now accepts planner_obs and planner).
        total_reward, metrics = run_episode(env, planner_policy=scheme_info["planner_policy"])
        scheme_rewards.append(total_reward)
        all_metrics.append(metrics)
    
    # Simple average of metrics across episodes.
    avg_metrics = {}
    for met in all_metrics:
        for key, value in met.items():
            avg_metrics.setdefault(key, []).append(value)
    for key in avg_metrics:
        avg_metrics[key] = np.mean(avg_metrics[key])
    
    results[scheme_name] = {
        "avg_planner_reward": np.mean(scheme_rewards),
        "avg_metrics": avg_metrics,
    }
        
    print("Results for {}:".format(scheme_name))
    print("  Average Planner Reward: {:.3f}".format(results[scheme_name]["avg_planner_reward"]))
    for key, value in avg_metrics.items():
        print("    {}: {:.3f}".format(key, value))
    print("\n")

print("===== Summary =====")
for scheme_name, out in results.items():
    print("Scheme: {:10s} | Avg Planner Reward: {:.3f}".format(scheme_name, out["avg_planner_reward"]))
