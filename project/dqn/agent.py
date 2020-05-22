import numpy as np
import random
from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy

"""
An agent for Deep Q-Learning models.

It encapsulates a neural network to predict Q-values and an experience-replay buffer to train the model on data batches.
"""


class SingleDQNAgent:
    """
    Args:
        env: OpenAI gym associated environment
        optimizer: Neural Network optimizer
        gamma: Discount
        epsilon: Exploration factor
        replay_buffer_length: Length of the experience replay buffer
    """

    def __init__(self, env, optimizer, gamma=0.6, epsilon=1, delta_epsilon=0.02, replay_buffer_length=100):
        self.env = env
        self._state_size = env.observation_space.n
        self._action_size = env.action_space.n
        self._optimizer = optimizer

        self.replay_buffer = deque(maxlen=replay_buffer_length)

        self.gamma = gamma
        self.epsilon = epsilon
        self.delta_epsilon = delta_epsilon

        # Build q anf target networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.update_target_model()

    """
    Store the experience in the Replay Buffer
    """

    def store(self, state, action, reward, next_state, terminated):
        self.replay_buffer.append((state, action, reward, next_state, terminated))

    """
    Build the neural network to estimate q values
    """

    def _build_compile_model(self):
        model = Sequential()
        model.add(Dense(10, input_shape=(self._state_size,), activation='relu'))
        model.add(Dense(10, input_shape=(self._state_size,), activation='relu'))
        model.add(Dense(self._action_size))

        model.compile(loss=CategoricalCrossentropy(), optimizer=self._optimizer)
        return model

    """
    Update the target network using the weights of the q one
    """

    def update_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    """
    Decide the action to take based on the model or exploring new one
    
    Args:
        state: the current state extracted from the observation
    """

    def act(self, state):
        # print(self.q_network.get_weights())
        # Exploration
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()

        self.epsilon = self.epsilon - self.delta_epsilon if self.epsilon > 0.1 else 0.1

        # Exploitation
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    """
    The method used to train the model using a batch of experiences.

    Args:
        batch_size: the number of samples extracted from the Replay Buffer and used to train the model.
    """

    def retrain(self, batch_size):
        if batch_size > len(self.replay_buffer):
            raise ValueError("Replay Buffer length exceeded.")

        minibatch = random.sample(self.replay_buffer, batch_size)

        for state, action, reward, new_state, done in minibatch:
            target = self.q_network.predict(state)

            if done:
                target[0][action] = reward
            else:
                t = self.target_network.predict(new_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=5, verbose=0)
