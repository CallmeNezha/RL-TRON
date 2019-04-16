import os
# I NEED A MODEL BUILD FROM KERAS

import threading
import multiprocessing
import numpy as np
from queue import Queue

import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers

tf.enable_eager_execution()

