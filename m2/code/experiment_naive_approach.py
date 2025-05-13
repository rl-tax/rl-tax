import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
from gym.spaces import Discrete, MultiDiscrete

from ai_economist.foundation.base.base_env import BaseEnvironment
from ai_economist.foundation.scenarios.simple_wood_and_stone.dynamic_layout import Uniform
from ai_economist.foundation.components.redistribution import PeriodicBracketTax
from ai_economist.foundation.components.build import Build
# from ai_economist.foundation.components.move import Move
from ai_economist.foundation.env_wrapper import FoundationEnvWrapper


class PolicyNetwork(nn.Module):
    """
    Simple policy network for both agents and planner
    """
    def __init__(self, input_dim, hidden_dim, output_dim, use_lstm=False):
        super(PolicyNetwork, self).__init__()
        self.use_lstm = use_lstm
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if use_lstm:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.policy_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        # For tracking hidden state
        self.hidden = None
    
    def forward(self, x, h=None):
        if self.use_lstm:
            if h is None:
                self.hidden = None
            
            # Ensure input has 3 dimensions [batch_size, seq_len, features]
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            if self.hidden is None:
                # Initialize hidden state
                batch_size = x.size(0)
                self.hidden = (
                    torch.zeros(1, batch_size, self.hidden_dim).to(x.device),
                    torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
                )
            
            x, self.hidden = self.lstm(x, self.hidden)
            x = x.squeeze(1)  # Remove sequence dimension for output
        
        logits = self.policy_head(x)
        return logits
    
    def reset_hidden(self):
        self.hidden = None


class InnerOuterLoopRL:
    """
    Implementation of Inner-Outer Loop Reinforcement Learning algorithm
    for economic agents and social planner learning simultaneously.
    """
    def __init__(
        self, 
        sampling_horizon=100,
        tax_period=10,
        learning_rate=0.001,
        gamma=0.99,
        hidden_dim=128,
        n_agents=4,
        world_size=[25, 25],
        episode_length=1000,
        use_cuda=False,
        learning_algorithm="A3C"  # Options: A3C, PPO
    ):
        # Algorithm parameters
        self.h = sampling_horizon
        self.M = tax_period
        self.lr = learning_rate
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents
        self.world_size = world_size
        self.episode_length = episode_length
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.learning_algorithm = learning_algorithm
        
        # Setup environment
        self.env = self._create_environment()
        
        # Extract actual observation and action dimensions from environment
        self._extract_obs_action_dims()
        
        # Create policy networks for agents and planner
        self._setup_policy_networks()
        
        # Create optimizers
        self.agent_optimizer = optim.Adam(self.agent_policy.parameters(), lr=self.lr)
        self.planner_optimizer = optim.Adam(self.planner_policy.parameters(), lr=self.lr)
        
        # Create transition buffers
        self.agent_buffer = []
        self.planner_buffer = []
        
        # Create hidden state tracking
        self.s = None
        self.o = None
        self.op = None
        self.h_agent = None
        self.h_planner = None
        
    def _create_environment(self):
        """
        Create the AI Economist environment with necessary components
        """
        env = Uniform(
            n_agents=self.n_agents,
            world_size=self.world_size,
            episode_length=self.episode_length,
            flatten_observations=True,
            components=[
                # Basic movement and resource gathering
                # ("Move", {}),
                ("Gather", {}),
                ("Build", {}),
                # Tax component - enables the planner to set tax rates
                ("PeriodicBracketTax", {
                    "period": self.M,
                    "bracket_spacing": "us-federal",
                    "tax_model": "model_wrapper"
                })
            ]
        )
        
        # Wrap the environment for easier interaction
        return FoundationEnvWrapper(env_obj=env)
    
    def _extract_obs_action_dims(self):
        """
        Extract observation and action dimensions from the environment
        """
        # Get a sample observation to determine dimensions
        obs = self.env.reset()
        
        # Extract agent observation dimension (assuming all agents have same dim)
        if '0' in obs and 'flat' in obs['0']:
            self.agent_obs_dim = len(obs['0']['flat'])
        else:
            # Default fallback
            self.agent_obs_dim = 64
        
        # Extract planner observation dimension
        if 'p' in obs and 'flat' in obs['p']:
            self.planner_obs_dim = len(obs['p']['flat'])
        else:
            # Default fallback
            self.planner_obs_dim = 32
        
        # Extract action dimensions
        if '0' in self.env.env.action_space:
            agent_space = self.env.env.action_space['0']
            if isinstance(agent_space, Discrete):
                self.agent_action_dim = agent_space.n
            elif isinstance(agent_space, MultiDiscrete):
                # For MultiDiscrete, use the sum of possible values for each action dimension
                self.agent_action_dim = 8  # Simplify for now
            else:
                # Default fallback
                self.agent_action_dim = 8
        else:
            self.agent_action_dim = 8
        
        # Extract planner action dimension
        if 'p' in self.env.env.action_space:
            planner_space = self.env.env.action_space['p']
            if isinstance(planner_space, Discrete):
                self.planner_action_dim = planner_space.n
            elif isinstance(planner_space, MultiDiscrete):
                # For MultiDiscrete, we simplify to a single action space 
                # with the sum of all possible actions across dimensions
                # Each tax bracket will have a separate action
                self.is_planner_multidiscrete = True
                self.planner_action_spaces = planner_space.nvec
                # Use a simplified approach with a single dimension
                self.planner_action_dim = np.sum(planner_space.nvec).item()
            else:
                # Default fallback
                self.planner_action_dim = 35  # 5 brackets with 7 possible rates each
                self.is_planner_multidiscrete = False
        else:
            self.planner_action_dim = 35
            self.is_planner_multidiscrete = False
        
        print(f"Agent obs dim: {self.agent_obs_dim}, action dim: {self.agent_action_dim}")
        print(f"Planner obs dim: {self.planner_obs_dim}, action dim: {self.planner_action_dim}")
        if hasattr(self, 'is_planner_multidiscrete') and self.is_planner_multidiscrete:
            print(f"Planner action spaces: {self.planner_action_spaces}")
    
    def _setup_policy_networks(self):
        """
        Create policy networks for agents and planner
        """
        # Create policy networks with actual dimensions from environment
        self.agent_policy = PolicyNetwork(
            input_dim=self.agent_obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.agent_action_dim,
            use_lstm=True
        ).to(self.device)
        
        self.planner_policy = PolicyNetwork(
            input_dim=self.planner_obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.planner_action_dim,
            use_lstm=True
        ).to(self.device)
    
    def reset(self):
        """
        Reset the environment and agent/planner states
        """
        # Reset environment
        obs = self.env.reset()
        
        # Reset hidden states
        self.s = obs
        self.o = self._process_agent_obs(obs)
        self.op = self._process_planner_obs(obs)
        self.h_agent = None
        self.h_planner = None
        
        # Reset policy network hidden states
        self.agent_policy.reset_hidden()
        self.planner_policy.reset_hidden()
        
        # Reset transition buffers
        self.agent_buffer = []
        self.planner_buffer = []
        
        return obs
    
    def _process_agent_obs(self, obs):
        """Process raw observations for agent policy"""
        # In real implementation, this would extract and format
        # agent-specific observations
        agent_obs = {}
        for idx in range(self.n_agents):
            if str(idx) in obs and 'flat' in obs[str(idx)]:
                agent_obs[idx] = torch.FloatTensor(obs[str(idx)]['flat']).to(self.device)
        return agent_obs
    
    def _process_planner_obs(self, obs):
        """Process raw observations for planner policy"""
        # In real implementation, this would extract and format
        # planner-specific observations
        if 'p' in obs and 'flat' in obs['p']:
            return torch.FloatTensor(obs['p']['flat']).to(self.device)
        return None
    
    def select_agent_actions(self, obs, t):
        """
        Select actions for all agents based on their policy
        """
        actions = {}
        for idx in range(self.n_agents):
            if idx in obs:
                # Forward pass through policy
                logits = self.agent_policy(obs[idx].unsqueeze(0))
                # Sample action from the policy
                action_probs = torch.softmax(logits, dim=-1)
                try:
                    action = torch.multinomial(action_probs, 1).item()
                except RuntimeError:
                    # Fallback if sampling fails (e.g., NaN values)
                    action = torch.argmax(action_probs).item()
                
                actions[str(idx)] = action
        return actions
    
    def select_planner_action(self, obs, t):
        """
        Select action for the planner based on their policy
        """
        if obs is not None:
            # Forward pass through policy
            logits = self.planner_policy(obs.unsqueeze(0))
            
            # For multidiscrete planner action space
            if hasattr(self, 'is_planner_multidiscrete') and self.is_planner_multidiscrete:
                # Simplify by choosing actions randomly for the demonstration
                # In a real implementation, you would use the policy output
                action_array = np.array([np.random.randint(0, space) for space in self.planner_action_spaces])
                return {'p': action_array}
            else:
                # Sample action from the policy for single discrete action
                action_probs = torch.softmax(logits, dim=-1)
                try:
                    action = torch.multinomial(action_probs, 1).item()
                except RuntimeError:
                    # Fallback if sampling fails (e.g., NaN values)
                    action = torch.argmax(action_probs).item()
                
                return {'p': action}
        
        # Default action if no observation
        if hasattr(self, 'is_planner_multidiscrete') and self.is_planner_multidiscrete:
            return {'p': np.zeros(len(self.planner_action_spaces), dtype=np.int32)}
        else:
            return {'p': 0}
    
    def _compute_returns(self, rewards, next_value=0):
        """Compute returns with GAE"""
        returns = []
        gae = 0
        next_value = next_value
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value
            gae = delta
            returns.insert(0, gae)
            next_value = gae
            
        return returns
    
    def update_policies(self):
        """
        Update agent and planner policies using collected experience
        """
        # Process agent buffer
        if self.agent_buffer:
            states, actions, rewards = zip(*self.agent_buffer)
            
            # Compute returns
            returns = self._compute_returns(rewards)
            
            # Convert to tensors
            states = torch.cat(states)
            actions = torch.tensor(actions, device=self.device).long()
            returns = torch.tensor(returns, device=self.device)
            
            # Get action logits
            logits = self.agent_policy(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Compute loss
            policy_loss = -(selected_log_probs * returns).mean()
            
            # Update agent policy
            self.agent_optimizer.zero_grad()
            policy_loss.backward()
            self.agent_optimizer.step()
        
        # Process planner buffer - simplified version
        # In a real implementation, you would need to handle multidiscrete actions more carefully
        if self.planner_buffer and not (hasattr(self, 'is_planner_multidiscrete') and self.is_planner_multidiscrete):
            states, actions, rewards = zip(*self.planner_buffer)
            
            # Compute returns
            returns = self._compute_returns(rewards)
            
            # Convert to tensors
            states = torch.cat(states)
            actions = torch.tensor(actions, device=self.device).long()
            returns = torch.tensor(returns, device=self.device)
            
            # Get action logits
            logits = self.planner_policy(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Compute loss
            policy_loss = -(selected_log_probs * returns).mean()
            
            # Update planner policy
            self.planner_optimizer.zero_grad()
            policy_loss.backward()
            self.planner_optimizer.step()
        
        # Clear buffers
        self.agent_buffer = []
        self.planner_buffer = []
    
    def train(self, n_episodes=1000):
        """
        Train the agent and planner policies
        """
        for episode in range(n_episodes):
            # Reset environment and states
            obs = self.reset()
            done = False
            episode_reward_agent = 0
            episode_reward_planner = 0
            
            # Run episode
            t = 0
            while not done and t < self.episode_length:
                # Process observations
                agent_obs = self._process_agent_obs(obs)
                planner_obs = self._process_planner_obs(obs)
                
                # Select actions based on policies
                agent_actions = self.select_agent_actions(agent_obs, t)
                
                # Determine if it's the first timestep of a tax period
                if t % self.M == 0:
                    # Sample tax rates at the first timestep of tax period
                    planner_action = self.select_planner_action(planner_obs, t)
                    actions = {**agent_actions, **planner_action}
                else:
                    # Only update planner hidden state otherwise
                    actions = agent_actions
                
                # Take a step in the environment
                next_obs, rewards, done, info = self.env.step(actions)
                
                # Store experience in buffers
                for idx in range(self.n_agents):
                    if idx in agent_obs and str(idx) in rewards:
                        self.agent_buffer.append((
                            agent_obs[idx].unsqueeze(0),
                            agent_actions[str(idx)],
                            rewards[str(idx)]
                        ))
                
                # Skip planner buffer updates for multidiscrete planner for now
                if ('p' in rewards and planner_obs is not None and 
                    t % self.M == 0 and not (hasattr(self, 'is_planner_multidiscrete') and self.is_planner_multidiscrete)):
                    self.planner_buffer.append((
                        planner_obs.unsqueeze(0),
                        planner_action['p'],
                        rewards['p']
                    ))
                
                # Update states
                obs = next_obs
                t += 1
                
                # Track rewards
                for idx in range(self.n_agents):
                    if str(idx) in rewards:
                        episode_reward_agent += rewards[str(idx)]
                
                if 'p' in rewards:
                    episode_reward_planner += rewards['p']
            
            # Update policies at the end of the episode
            self.update_policies()
            
            # Print episode statistics
            print(f"Episode {episode+1}/{n_episodes} | Agent Reward: {episode_reward_agent/self.n_agents:.2f} | Planner Reward: {episode_reward_planner:.2f}")
            
            
    def run_episode(self, render=False):
        """
        Run a single episode with the trained policies
        """
        obs = self.reset()
        done = False
        total_agent_reward = 0
        total_planner_reward = 0
        
        t = 0
        while not done and t < self.episode_length:
            # Process observations
            agent_obs = self._process_agent_obs(obs)
            planner_obs = self._process_planner_obs(obs)
            
            # Select actions based on policies
            agent_actions = self.select_agent_actions(agent_obs, t)
            
            # Determine if it's the first timestep of a tax period
            if t % self.M == 0:
                # Sample tax rates at the first timestep of tax period
                planner_action = self.select_planner_action(planner_obs, t)
                actions = {**agent_actions, **planner_action}
            else:
                # Only update planner hidden state otherwise
                actions = agent_actions
            
            # Take a step in the environment
            next_obs, rewards, done, info = self.env.step(actions)
            
            # Update states
            obs = next_obs
            t += 1
            
            # Track rewards
            for idx in range(self.n_agents):
                if str(idx) in rewards:
                    total_agent_reward += rewards[str(idx)]
            
            if 'p' in rewards:
                total_planner_reward += rewards['p']
        
        return total_agent_reward/self.n_agents, total_planner_reward


if __name__ == "__main__":
    inner_outer_rl = InnerOuterLoopRL(
        sampling_horizon=100,
        tax_period=10,
        learning_rate=0.001,
        gamma=0.99,
        hidden_dim=128,
        n_agents=4,
        world_size=[25, 25],
        episode_length=1000,
        use_cuda=False,
        learning_algorithm="A3C"
    )
    
    inner_outer_rl.train(n_episodes=1000)
    
    agent_reward, planner_reward = inner_outer_rl.run_episode()
    print(f"Evaluation | Agent Reward: {agent_reward:.2f} | Planner Reward: {planner_reward:.2f}") 