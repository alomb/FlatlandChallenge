import numpy as np
import random
from collections import deque

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.python.keras import Input

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
    """

    REPLAY_BUFFER_MAX_LEN = 2000

    def __init__(self, env, optimizer, gamma=0.6, epsilon=0.1):
        self.env = env
        self._state_size = env.observation_space.n
        self._action_size = env.action_space.n
        self._optimizer = optimizer

        self.replay_buffer = deque(maxlen=self.REPLAY_BUFFER_MAX_LEN)

        self.gamma = gamma
        self.epsilon = epsilon

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
        model.add(Input(shape=(11,)))
        model.add(Reshape((11,)))
        model.add(Dense(22, activation='relu'))
        model.add(Dense(22, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
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

            self.q_network.fit(state, target, epochs=1, verbose=0)
