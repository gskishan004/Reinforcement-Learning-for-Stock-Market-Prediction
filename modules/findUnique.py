import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
import math



def findUniqueStocks(df):

    return df['ticker'].unique()


def findUniqueTargets(df):
    
    return df['target_name'].unique()