import os
import argparse
# I NEED A MODEL BUILD FROM KERAS

import threading
import multiprocessing
import numpy as np
from queue import Queue

import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description="Run a3c on TRON")

parser.add_argument('--train', dest='train', action='store_true', help='Train the model')
parser.add_argument('--lr', default=0.001, type=float,help='Learning rate for the shared optimizer')
parser.add_argument('--update-freq', default=20, type=int, help='How often to update the global model')
parser.add_argument('--max-eps', default=1000, type=int, help='Global maximum number of episodes to run')
parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor of rewards')

class ActorCriticModel(keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.dense1 = layers.Dense(100, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(100, activation='relu')
        self.values = layers.Dense(1)


    def call(self, inputs):
        # FORWARD PASS
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values


def record(episode,
           episode_reward,
           worker_idx,
           
            )

    