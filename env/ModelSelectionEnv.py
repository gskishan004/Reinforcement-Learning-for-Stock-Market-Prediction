import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
import math


'''
            To-Do
------------------------------

Change the definition of done

Fixed
- problems in some global and self variables like model_dict, current_eval_date 

'''


STOCK_NAME          = "FAST"
TARGET_NAME         = "1P28D"
MAX_REWARD          = 2147483647
ALL_DATES           = []
MODEL_DICT          = {} 
CURRENT_MODELS      = []
CURRENT_EVAL_DATE   = ""
MAX_STEPS           = 0

class ModelSelectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        global MAX_STEPS

        super(ModelSelectionEnv, self).__init__()
        self.df             = df
        self.reward_range   = (0, MAX_REWARD)
        self.df             = df[(df['ticker'] == STOCK_NAME) & (df['target_name'] == TARGET_NAME )]
        unique_modelid      = self.df['modelid'].unique()
        N_DISCRETE_ACTIONS  = len(unique_modelid) 
        MAX_STEPS           = len(self.df['eval_date'].unique())        #for first run, max_steps = 29
        self.CURRENT_REWARD = 0

        """
        - Can be conceptualized as 3 discrete action spaces:
            1) Discrete 5   - params: min: 0, max: 4
            2) Discrete 2   - params: min: 0, max: 1
            3) Discrete 2   - params: min: 0, max: 1
        - Can be initialized as
            MultiDiscrete([ 5, 2, 2 ])
        """
        #Action Space
        if (N_DISCRETE_ACTIONS == 0): print("ERROR in Finding N_DISCRETE_ACTIONS")

        self.action_space = spaces.MultiDiscrete([ N_DISCRETE_ACTIONS , 2 ]) 

        for i,num in enumerate (unique_modelid):
            MODEL_DICT [i]=num
            CURRENT_MODELS.append(num)

        for date in (self.df['eval_date'].unique()):
            ALL_DATES.append(date)


        #Observation Space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(N_DISCRETE_ACTIONS, 3), dtype=np.float64)


    def _next_observation(self):

        global CURRENT_EVAL_DATE

        frame   = np.empty((0,3), float)
        df      = self.df


        CURRENT_EVAL_DATE = ALL_DATES.pop(0)

        for model in MODEL_DICT.values():
            if df[(df['modelid'] == int(model)) & (df['eval_date'] == CURRENT_EVAL_DATE )].empty:
                    print ("Data frame empty for model:", model, "eval_date", CURRENT_EVAL_DATE)
            else:
                curr_precent_change  = df[(df['modelid'] == int(model)) & (df['eval_date'] == CURRENT_EVAL_DATE )]["percent_change"].iloc[0]
                curr_USD_change      = df[(df['modelid'] == int(model)) & (df['eval_date'] == CURRENT_EVAL_DATE )]["USD_change"].iloc[0]
                curr_probability     = df[(df['modelid'] == int(model)) & (df['eval_date'] == CURRENT_EVAL_DATE )]["probability"].iloc[0]

                frame = np.append(frame, np.array([[curr_precent_change,curr_USD_change,curr_probability]]), axis=0)


        return frame

    def _take_action(self, action):

        global CURRENT_MODELS
        global CURRENT_EVAL_DATE


        selected_model = action[0]  #1,2,3,4 ...
        add_or_remove  = action[1]  #1 or 2



        if add_or_remove < 1:
            CURRENT_MODELS.append(MODEL_DICT[selected_model])
            CURRENT_MODELS = list(set(CURRENT_MODELS))

        elif add_or_remove < 2:
            if (MODEL_DICT[selected_model] in CURRENT_MODELS):  
                CURRENT_MODELS.remove(MODEL_DICT[selected_model])

        else:
            print("Error IN add_or_remove selected")

        df = self.df
        for model in CURRENT_MODELS:
            if df[(df['modelid'] == int(model)) & (df['eval_date'] == CURRENT_EVAL_DATE )].empty:
                    print ("In take action ... Data frame empty for model:", model, "eval_data", CURRENT_EVAL_DATE)
            else:
                current_probability = float(df[(df['modelid'] == int(model)) & (df['eval_date'] == CURRENT_EVAL_DATE )]["probability"].iloc[0])
                current_target_val  = int(df[(df['modelid'] == int(model)) & (df['eval_date'] == CURRENT_EVAL_DATE )]["target_val"].iloc[0])

            
                if "_" in TARGET_NAME:
                    if current_target_val == 0:
                        self.CURRENT_REWARD += math.pow(MAX_REWARD, current_probability)
                    else:
                        self.CURRENT_REWARD -= math.pow(MAX_REWARD, current_probability)
                else:
                    if current_target_val == 1:
                        self.CURRENT_REWARD += math.pow(MAX_REWARD, current_probability)
                    else:
                        self.CURRENT_REWARD -= math.pow(MAX_REWARD, current_probability)

    def step(self, action):
        global MAX_STEPS

        self._take_action(action)

        self.current_step += 1

        if self.current_step > MAX_STEPS :
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.CURRENT_REWARD*delay_modifier

        done = self.CURRENT_REWARD == MAX_REWARD

        obs = self._next_observation()

        return obs, reward, done, {}



    def reset(self):
        self.MODEL_DICT         = {} 
        self.CURRENT_MODELS     = [] 
        self.CURRENT_REWARD     = 0
        self.ALL_DATES          = []
        self.current_step       = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        global CURRENT_MODELS

        print(f'Step: {self.current_step}')
        print(f'Current Models: {CURRENT_MODELS}')
        print(f'Current Reward: {self.CURRENT_REWARD}')


'''
Deprecated in favor of operator or tf.math.divide.
Step: 1
Current Models: [3376, 4445, 3755, 4658]
Current Reward: 5097033.2774529625
Step: 2
Current Models: [3376, 4658, 3755, 4445]
Current Reward: 12087023.25751828
Step: 3
Current Models: [3376, 3755, 4445]
Current Reward: 16472960.139603585
Step: 4
Current Models: [3376, 3755]
Current Reward: 19380921.449753474
Step: 5
Current Models: [3376, 3755, 2901]
Current Reward: 34217659.13909666
Step: 6
Current Models: [3376, 4658, 3755, 2901]
Current Reward: 48500477.594296694
Data frame empty for model: 4658 eval_date 2018-10-11
Traceback (most recent call last):
  File "main.py", line 23, in <module>
    obs, rewards, done, info = env.step(action)

ValueError: could not broadcast input array from shape (4,3) into shape (5,3)
'''