from ai_economist import foundation
from ai_economist.foundation.env_wrapper import FoundationEnvWrapper
import numpy as np
import matplotlib.pyplot as plt
from agent import sample_random_actions
from utils import play_random_episode
from agent import plot_and_save_utility_curves


# Create environment and its dynamics

# Define the configuration of the environment that will be built
env_config = {
    # ===== SCENARIO CLASS =====
    # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
    # The environment object will be an instance of the Scenario class.
    'scenario_name': 'layout_from_file/simple_wood_and_stone',
    
    # ===== COMPONENTS =====
    # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
    #     "component_name" refers to the Component class's name in the Component Registry (foundation.components)
    #     {component_kwargs} is a dictionary of kwargs passed to the Component class
    # The order in which components reset, step, and generate obs follows their listed order below.
    'components': [
        # (1) Building houses
        ('Build', dict(skill_dist="pareto", payment_max_skill_multiplier=3)),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', dict(max_num_orders=5)),
        # (3) Movement and resource collection
        ('Gather', dict()),
        # (4) Income tax & lump-sum redistribution
        # ('PeriodicBracketTax', dict(bracket_spacing="us-federal", period=100))
    ],
    
    # ===== SCENARIO CLASS ARGUMENTS =====
    # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
    'env_layout_file': "quadrant_25x25_20each_30clump.txt",
    'fixed_four_skill_and_loc': True,
    
    # ===== STANDARD ARGUMENTS ======
    # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
    'n_agents': 4,          # Number of non-planner agents (must be >1)
    'world_size': [25, 25], # [Height, Width] of the env world
    'episode_length': 1000, # Number of timesteps per episode
    
    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,
    
    # When flattening observations, concatenate scalar & vector observations before output.
    # Otherwise, return observations with minimal processing.
    'flatten_observations': False,
    # When Flattening masks, concatenate each action subspace mask into a single array.
    # Note: flatten_masks = True is required for masking action logits in the code below.
    'flatten_masks': True,
}
env = foundation.make_env_instance(**env_config)

# Quick Test 
obs = env.reset()
actions = sample_random_actions(env, obs)
obs, rewards, done, info = env.step(actions)

print(f"obs: {obs}, rewards: {rewards}, done: {done}, info: {info}")

for key, val in obs['0'].items(): 
    print("{:50} {}".format(key, type(val)))

for agent_idx, reward in rewards.items(): 
    print("{:2} {:.3f}".format(agent_idx, reward))


play_random_episode(env, plot_every=100)

# Check utility curves based on skills
skills_to_plot = [10, 15, 20, 30]
plot_and_save_utility_curves(skills_to_plot, filename="./images/utility_curves_based_on_skill.png")

# -----------------------------
# Run Multiple Episodes and Collect Social Welfare Metrics
# -----------------------------
def run_free_market_simulation(num_episodes=50):
    """
    Run several episodes on the free market (no tax) simulation and collect a social
    welfare metric (here, using the 'coin_eq_times_productivity' metric).
    """
    social_welfare_history = []  # list to record the welfare metric each episode
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = {'__all__': False}
        step_count = 0

        # Run episode until done.
        while not done['__all__']:
            actions = sample_random_actions(env, obs)
            obs, rewards, done, info = env.step(actions)
            step_count += 1

        # After episode, extract the social welfare from the environment’s metrics.
        # In our layout_from_file scenario, the scenario_metrics() method returns a dictionary
        # that may include a key like 'social_welfare/coin_eq_times_productivity'.
        metrics = env.previous_episode_metrics
        social_welfare = metrics.get('social_welfare/coin_eq_times_productivity', np.nan)
        social_welfare_history.append(social_welfare)
        print(f"Episode {ep+1:02d}: {step_count:4d} steps, Social welfare = {social_welfare:.3f}")

    return social_welfare_history


# Run the simulation over a set number of episodes.
num_episodes = 50
sw_history = run_free_market_simulation(num_episodes=num_episodes)

# Plot social welfare metric over episodes.
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_episodes+1), sw_history, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Social Welfare (coin_eq_times_productivity)")
plt.title("Free Market Simulation: Social Welfare vs. Episode")
plt.grid(True)
plt.savefig("./images/free_market_sw.png")  # Save the figure here.
plt.show()


