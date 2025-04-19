import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, 
                 state_shape=(9, 9), 
                 action_size=729, 
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_min=0.1, 
                 epsilon_decay=0.997,  # Slower decay
                 learning_rate=0.001,
                 memory_size=10000):  # Bigger memory
        
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Build a more complex neural network with enhanced features for Sudoku"""
        # Input for the board state
        board_input = tf.keras.layers.Input(shape=self.state_shape)
        
        # Input for constraint features (which numbers are used in each row/col/box)
        rows_input = tf.keras.layers.Input(shape=(9, 9))
        cols_input = tf.keras.layers.Input(shape=(9, 9))
        boxes_input = tf.keras.layers.Input(shape=(9, 9))
        
        # Input for action mask (which actions are valid)
        mask_input = tf.keras.layers.Input(shape=(self.action_size,))
        
        # Process board state
        flat_board = tf.keras.layers.Flatten()(board_input)
        
        # Process constraint features
        flat_rows = tf.keras.layers.Flatten()(rows_input)
        flat_cols = tf.keras.layers.Flatten()(cols_input)
        flat_boxes = tf.keras.layers.Flatten()(boxes_input)
        
        # Concatenate all features
        concat = tf.keras.layers.Concatenate()([flat_board, flat_rows, flat_cols, flat_boxes])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(512, activation='relu')(concat)
        dense2 = tf.keras.layers.Dense(512, activation='relu')(dense1)
        
        # Output Q-values
        q_values = tf.keras.layers.Dense(self.action_size, activation='linear')(dense2)
        
        # Apply mask to make invalid actions have very negative Q-values
        masked_q_values = tf.keras.layers.Lambda(
            lambda x: x[0] * x[1] - 1e6 * (1 - x[1])
        )([q_values, mask_input])
        
        # Create model
        model = tf.keras.Model(
            inputs=[board_input, rows_input, cols_input, boxes_input, mask_input],
            outputs=masked_q_values
        )
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mean_squared_error')

        return model

    def update_target_model(self):
        """Update target model to match weights of the main model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action_index, reward, next_state, done, mask, next_mask, constraints, next_constraints):
        """Store experience in memory"""
        self.memory.append((state, action_index, reward, next_state, done, 
                           mask, next_mask, constraints, next_constraints))

    def act(self, state, mask, constraints):
        """Choose an action using epsilon-greedy policy with guided exploration"""
        # Normalize the state
        norm_state = state / 9.0
        rows, cols, boxes = constraints
        
        # With probability epsilon, choose a random valid action
        if np.random.rand() <= self.epsilon:
            # Get indices of valid actions (where mask is 1)
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) > 0:
                return np.random.choice(valid_indices)
            else:
                # If no valid actions, choose any action (will be invalid)
                return random.randrange(self.action_size)
        
        # Otherwise, choose the action with highest Q-value
        q_values = self.model.predict(
            [np.expand_dims(norm_state, axis=0), 
             np.expand_dims(rows, axis=0),
             np.expand_dims(cols, axis=0),
             np.expand_dims(boxes, axis=0),
             np.expand_dims(mask, axis=0)],
            verbose=0
        )
        return np.argmax(q_values[0])

    def replay(self, batch_size=64):
        """Train the model on experiences from memory"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        # Extract components
        states = np.array([sample[0] for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])
        masks = np.array([sample[5] for sample in minibatch])
        next_masks = np.array([sample[6] for sample in minibatch])
        
        # Extract constraints
        rows_list = np.array([sample[7][0] for sample in minibatch])
        cols_list = np.array([sample[7][1] for sample in minibatch])
        boxes_list = np.array([sample[7][2] for sample in minibatch])
        next_rows_list = np.array([sample[8][0] for sample in minibatch])
        next_cols_list = np.array([sample[8][1] for sample in minibatch])
        next_boxes_list = np.array([sample[8][2] for sample in minibatch])
        
        # Normalize states
        norm_states = states / 9.0
        norm_next_states = next_states / 9.0

        # Get current Q values
        target_qs = self.model.predict(
            [norm_states, rows_list, cols_list, boxes_list, masks], 
            verbose=0
        )
        
        # Get next Q values from target model
        next_qs = self.target_model.predict(
            [norm_next_states, next_rows_list, next_cols_list, next_boxes_list, next_masks], 
            verbose=0
        )

        # Calculate targets
        for i in range(batch_size):
            if dones[i]:
                target_qs[i][actions[i]] = rewards[i]
            else:
                target_qs[i][actions[i]] = rewards[i] + self.gamma * np.max(next_qs[i])

        # Train the model
        self.model.fit(
            [norm_states, rows_list, cols_list, boxes_list, masks],
            target_qs,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decode_action(self, action_index):
        """Convert action index to (row, col, num) format"""
        row = action_index // 81
        col = (action_index % 81) // 9
        num = (action_index % 9) + 1
        
        return (row, col, num)
    
    
    def encode_action(self, row, col, num):
        """Convert (row, col, num) to action index"""
        return row * 81 + col * 9 + (num - 1)
        

    def save(self, filepath):
        self.model.save(filepath)
        

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.update_target_model()



