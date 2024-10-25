import numpy as np

class RMaxExploration:
    def __init__(self, state_dim, action_dim, R_max=1.0, exploration_threshold=10, buffer_size=10000):
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
        self.buffer = []  # Initialize the buffer to store collected interactions
        self.buffer_size = buffer_size  # Maximum buffer size
    
    def get_rmax_reward(self, state, action, real_reward):
        """
        Return the R-max reward if the state-action pair has not been sufficiently explored.
        Otherwise, return the real reward.

        Args:
            state (int): Current state.
            action (int): Action taken.
            real_reward (float): The real reward observed from the environment.

        Returns:
            float: R-max reward or real reward depending on visit count.
        """
        if self.visit_count[state, action] < self.exploration_threshold:
            return self.R_max  # Return maximum reward if not explored enough
        else:
            return real_reward  # Use the actual reward after sufficient exploration

    def update_visit_count(self, state, action, reward, next_state):
        """
        Increment the visit count for the given state-action pair and store the interaction
        in the buffer.

        Args:
            state (array): Current state.
            action (int): Action taken.
            reward (float): Observed reward.
            next_state (array): The next state observed after taking the action.
        """
        # Add the interaction to the buffer
        self.add_to_buffer(state, action, reward, next_state)
        # Count the visit for the state-action pair
        state = tuple(state)
        key = (state, action)
        if key not in self.visit_count.keys():
            self.visit_count[key] = 1
        self.visit_count[key] += 1
        

    
    def add_to_buffer(self, state, action, reward, next_state):
        """
        Add the interaction to the buffer and ensure the buffer doesn't exceed its maximum size.

        Args:
            state (float): Current state.
            action (int): Action taken.
            reward (float): Observed reward.
            next_state (float): The next state observed after taking the action.
        """
        # Add the interaction to the buffer
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)  # Remove the oldest data if buffer is full
        self.buffer.append((state, action, reward, next_state))

    def should_explore(self, state, action):
        """
        Check if the agent should explore the state-action pair based on the visit count.

        Args:
            state (int): Current state.
            action (int): Action taken.

        Returns:
            bool: True if the state-action pair needs more exploration, False otherwise.
        """
        return self.visit_count[state, action] < self.exploration_threshold

    def sample_buffer(self, batch_size):
        """
        Sample a batch of interactions from the buffer.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            list of tuples: A batch of (state, action, reward, next_state) from the buffer.
        """
        if len(self.buffer) < batch_size:
            return self.buffer  # If the buffer has fewer than batch_size elements, return all
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear_buffer(self):
        """
        Clear the buffer, useful when you want to start fresh or after training the world model.
        """
        self.buffer = []