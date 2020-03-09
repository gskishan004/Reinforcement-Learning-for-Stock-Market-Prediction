import random
import gym
import math
import pandas   as pd
import numpy    as np
from   gym      import spaces


MAX_REWARD          = 9999999999
starting_money      = 100000.0
current_money       = 100000.0
actions             = ['buy','sell']
askPriceList        = []
bidPriceList        = []
debug               = 1

class securities_trading_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):

        super(securities_trading_env, self).__init__()

        self.df                 = df
        self.reward_range       = (0, MAX_REWARD)
        self.MAX_STEPS          = len(df["L1bid_price"])  
        self.CURRENT_REWARD     = 0
        self.current_askPrice   = 0
        self.current_bidPrice   = 0
        self.current_held_sec   = 0

        
        # 25 columns of L1 bid price
        df_bidPrice = df[df.columns[[1,21,41,61,81,101,121,141,161,181,201,221,241,261,281,301,321,341,361,381,401,421,441,461,481]]]
        # 25 columns of L1 ask price
        df_askPrice = df[df.columns[[3,23,43,63,83,103,123,143,163,183,203,223,243,263,283,303,323,343,363,383,403,423,443,463,483]]]

        for data in df_askPrice["L1ask_price"]:
            askPriceList.append(data)

        for data in df_bidPrice["L1bid_price"]:
            bidPriceList.append(data)


        N_DISCRETE_ACTIONS  = len(actions) 
        
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

        self.action_space   = spaces.MultiDiscrete([N_DISCRETE_ACTIONS])


        #Observation Space: agent will only see ask price and bid price 
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(N_DISCRETE_ACTIONS, 2), dtype=np.float64)


    def _next_observation(self):

        global askPriceList, bidPriceList

        frame   = np.empty((0,2), float)
        df      = self.df

        curr_askPrice  = askPriceList.pop(0)
        curr_bidPrice  = bidPriceList.pop(0)

        self.current_askPrice   = curr_askPrice
        self.current_bidPrice   = curr_bidPrice


        frame = np.append(frame, np.array([[curr_askPrice,curr_bidPrice]]), axis=0)


        return frame

    def _take_action(self, action):

        global  current_money, actions, starting_money

        buy_or_sell  = action  #1 or 2

        if(debug ==1):
            print("Action is :", actions[buy_or_sell[0]])

        if buy_or_sell == 0 and (current_money -self.current_bidPrice)>0:
            #that means the action is buy
            current_money -= self.current_bidPrice 
            self.current_held_sec +=1
            self.CURRENT_REWARD = math.pow(10000000, current_money/starting_money)

        elif buy_or_sell == 1 and self.current_held_sec>0:
            #that means the action is sell
            current_money += self.current_bidPrice 
            self.current_held_sec -=1
            self.CURRENT_REWARD = math.pow(10000000, current_money/starting_money)

        else:
            print("cant buy or sell")
            self.CURRENT_REWARD = -10000000

    def step(self, action):

        self._take_action(action)

        self.current_step += 1

        if self.current_step > self.MAX_STEPS :
            self.current_step = 0

        if current_money < 0:
            self.current_step = 0

        delay_modifier = (self.current_step / self.MAX_STEPS)

        reward = self.CURRENT_REWARD*delay_modifier

        done = self.CURRENT_REWARD == MAX_REWARD

        obs = self._next_observation()

        return obs, reward, done, {}



    def reset(self):
        global askPriceList, bidPriceList, current_money

        self.CURRENT_REWARD     = 0
        self.current_step       = 0
        current_money           = starting_money
        df                      = self.df

        self.current_held_sec   = 0
        askPriceList            = []
        bidPriceList            = []

        # 25 columns of L1 bid price
        df_bidPrice = df[df.columns[[1,21,41,61,81,101,121,141,161,181,201,221,241,261,281,301,321,341,361,381,401,421,441,461,481]]]
        # 25 columns of L1 ask price
        df_askPrice = df[df.columns[[3,23,43,63,83,103,123,143,163,183,203,223,243,263,283,303,323,343,363,383,403,423,443,463,483]]]


        for data in df_askPrice["L1ask_price"]:
            askPriceList.append(data)

        for data in df_bidPrice["L1bid_price"]:
            bidPriceList.append(data)

        return self._next_observation()

    def render(self, mode='human', close=False):
        global  current_money

        print(f'Step: {self.current_step}')
        print(f'Price: {self.current_bidPrice}')
        print(f'Balance: {current_money}')
        print(f'Securities Held: {self.current_held_sec}')
        print(f'Current Reward: {self.CURRENT_REWARD}')