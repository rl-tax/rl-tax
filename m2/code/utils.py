import plotting
import matplotlib.pyplot as plt
from agent import sample_random_actions
import os

def do_plot(env, ax, fig, filename="./images/world_state.png"):
    """Plots world state during episode sampling."""
    os.makedirs("images", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10)) 
    plotting.plot_env_state(env, ax)
    ax.set_aspect('equal')
    plt.savefig(filename)  
    plt.close(fig) 

def play_random_episode(env, plot_every=100, do_dense_logging=False):
    """Plays an episode with randomly sampled actions.
    
    Demonstrates gym-style API:
        obs                  <-- env.reset(...)         # Reset
        obs, rew, done, info <-- env.step(actions, ...) # Interaction loop
    
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Reset
    obs = env.reset(force_dense_logging=do_dense_logging)

    # Interaction loop (w/ plotting)
    for t in range(env.episode_length):
        actions = sample_random_actions(env, obs)
        obs, rew, done, info = env.step(actions)

        if ((t+1) % plot_every) == 0:
            do_plot(env, ax, fig, filename=f"./images/world_state_{t+1}.png")

    if ((t+1) % plot_every) != 0:
        do_plot(env, ax, fig, filename=f"./images/world_state_{t+1}.png")        