from collections import Counter
import csv
import os
from datetime import datetime

import gym_super_mario_bros
import gym as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers2 import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "mario_1_1_ppo2.pt")     # Checkpoint im Skript-Verzeichnis
SUCCESS_FILE = os.path.join(SCRIPT_DIR, "mario_1_1_clear2.pt")      # Success-File im Skript-Verzeichnis
CSV_LOG_FILE = os.path.join(SCRIPT_DIR, "training_log2.csv")  