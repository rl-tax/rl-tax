
import numpy as np
import matplotlib.pyplot as plt
import os

def income_earned(labor, skill):
    return labor * skill

def utility(labor, skill):
    def isoelastic_utility(z, eta=0.35):
        return (z**(1-eta) - 1) / (1 - eta)
    
    income = income_earned(labor, skill)
    utility_from_income = isoelastic_utility(income)
    disutility_from_labor = labor
    
    return utility_from_income - disutility_from_labor

def plot_and_save_utility_curves(skills, filename="./images/utility_curves.png"):
    """
    Plots utility curves for different skill levels on a single figure
    and saves the figure to the 'images' directory.
    """
    # Create the 'image' directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    for skill in skills:
        labor_array = np.linspace(0, 1000, 501)
        utility_array = utility(labor_array, skill)
        ax.plot(labor_array, utility_array, label="Skill = {}".format(skill))
        ax.plot(labor_array[np.argmax(utility_array)], np.max(utility_array), 'k*', markersize=10)

    ax.set_xlabel("Labor Performed", fontsize=20)
    ax.set_ylabel("Utility Experienced", fontsize=20)
    ax.set_xlim(left=0, right=1000)
    ax.set_ylim(bottom=0)
    ax.legend()

    plt.savefig(filename)
    plt.close(fig)


def sample_random_action(agent, mask):
    """Sample random UNMASKED action(s) for agent."""
    # Return a list of actions: 1 for each action subspace
    if agent.multi_action_mode:
        split_masks = np.split(mask, agent.action_spaces.cumsum()[:-1])
        return [np.random.choice(np.arange(len(m_)), p=m_/m_.sum()) for m_ in split_masks]

    # Return a single action
    else:
        return np.random.choice(np.arange(agent.action_spaces), p=mask/mask.sum())

def sample_random_actions(env, obs):
    """Samples random UNMASKED actions for each agent in obs."""
        
    actions = {
        a_idx: sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
        for a_idx, a_obs in obs.items()
    }

    return actions

