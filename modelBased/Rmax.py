import numpy as np
import sys
sys.path.append('/home/siyao/project/rlPractice/MiniGrid')
from modelBased.data.datamodule import extract_agent_cross_mask

class RMaxExploration:
    def __init__(self, R_max=1.0, exploration_threshold=10):
        """
        Initialize R-max exploration parameters and buffer for storing collected data.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            R_max (float): Maximum reward for unexplored state-action pairs.
            exploration_threshold (int): Number of visits before using real rewards.
            buffer_size (int): Maximum size of the buffer to store interactions.
        """
        self.R_max = R_max  # Maximum assumed reward for unexplored state-action pairs
        self.exploration_threshold = exploration_threshold  # Exploration threshold
        self.visit_count = {}  # Visit count for each state-action pair
    
    def get_rmax_reward(self, state, action, real_reward):
        """
        Return the R-max reward if the state-action pair has not been sufficiently explored.
        Otherwise, return the real reward.

        Args:
            state (array): Current state.
            action (int): Action taken.
            real_reward (float): The real reward observed from the environment.

        Returns:
            float: R-max reward or real reward depending on visit count.
        """
        state = tuple(state.flatten()) if isinstance(state, np.ndarray) else state.flatten()
        key = (state, action)  # Define the key as the state-action pair

        # If the key does not exist, treat it as unexplored
        if key not in self.visit_count:
            self.visit_count[key] = 0  # Initialize visit count for the new state-action pair

        # Return R_max if the state-action pair is underexplored
        if self.visit_count[key] < self.exploration_threshold:
            return self.R_max  # Return R_max for unexplored or underexplored pairs
        else:
            return real_reward  # Use actual reward if sufficiently explored

    def update_visit_count(self, state, action):
        """
        Increment the visit count for the given state-action pair and store the interaction
        in the buffer.

        Args:
            state (array): Current state.
            action (int): Action taken.
            reward (float): Observed reward.
            next_state (array): The next state observed after taking the action.
        """
        # Count the visit for the state-action pair
        # process and extract the state
        state = extract_agent_cross_mask(state)
        state = tuple(state.flatten()) if isinstance(state, np.ndarray) else state.flatten()
        key = (state, action)

        # Update visit count
        if key not in self.visit_count:
            self.visit_count[key] = 1
        else:
            self.visit_count[key] += 1
        
    
    def save_count(self, path):
        np.savez_compressed(path, Rmax=self.visit_count)

    def load_count(self, path): 
        self.visit_count = np.load(path, allow_pickle=True)['Rmax'].item()

