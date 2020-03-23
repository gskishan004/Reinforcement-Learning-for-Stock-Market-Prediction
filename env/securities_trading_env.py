import random
import gym
import math
import configparser 
import pandas   as pd
import numpy    as np
from   gym      import spaces

config = configparser.ConfigParser()
config.read('config.ini')


MAX_REWARD          = int(config['ENV']['MaximumReward'])
starting_money      = float(config['ENV']['StartingMoney'])
bid_price_columns   = list(map(int,config['ENV']['ColumnsOfBidPrice'].split(',')))
ask_price_columns   = list(map(int,config['ENV']['ColumnsOfAskPrice'].split(',')))
current_money       = float(config['ENV']['StartingMoney'])
actions             = ['buy','sell','hold']
askPriceList        = []
bidPriceList        = []
debug               = 1
obsSpace            = int(config['ENV']['ObservationSpace'])
initial_flag        = True 
old_data            = np.empty((0,2), float)


print (ask_price_columns)

class securities_trading_env(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df):

        global bid_price_columns, ask_price_columns

        super(securities_trading_env, self).__init__()

        self.df                 = df
        self.reward_range       = (0, MAX_REWARD)
        self.MAX_STEPS          = len(df["L1bid_price"])  
        self.CURRENT_REWARD     = 0
        self.current_askPrice   = 0
        self.current_bidPrice   = 0
        self.current_held_sec   = 0

        
        # 25 columns of L1 bid price
        df_bidPrice = df[df.columns[bid_price_columns]]
        # 25 columns of L1 ask price
        df_askPrice = df[df.columns[ask_price_columns]]

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
        self.observation_space = spaces.Box(low=0, high=1, shape=(obsSpace, 2), dtype=np.float64)


    def _next_observation(self):

        global askPriceList, bidPriceList, initial_flag, old_data
        df      = self.df

        if(initial_flag):
            for _ in range(obsSpace-1):
                curr_askPrice  = askPriceList.pop(0)
                curr_bidPrice  = bidPriceList.pop(0)
                old_data = np.append(old_data, np.array([[curr_askPrice,curr_bidPrice]]), axis=0)

            initial_flag = False

        frame = np.empty((0,2), float)

        if (obsSpace != 1):
            frame   = old_data
            old_data = np.delete(old_data, (0), axis=0)


        curr_askPrice  = askPriceList.pop(0)
        curr_bidPrice  = bidPriceList.pop(0)

        self.current_askPrice   = curr_askPrice
        self.current_bidPrice   = curr_bidPrice
        frame = np.append(frame, np.array([[curr_askPrice,curr_bidPrice]]), axis=0)

        if (obsSpace != 1):
            old_data = np.append(old_data, np.array([[curr_askPrice,curr_bidPrice]]), axis=0)


        return frame

    def _take_action(self, action):

        global  current_money, actions, starting_money

        buy_or_sell  = action  #1 or 2 or 3

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

        elif buy_or_sell == 2:
            #that means the action is hold
            #current_money      - remains unchanged 
            #current_held_sec   - remains unchanged
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
        global askPriceList, bidPriceList, current_money, bid_price_columns, ask_price_columns, initial_flag, old_data

        self.CURRENT_REWARD     = 0
        self.current_step       = 0
        current_money           = starting_money
        df                      = self.df
        self.current_held_sec   = 0
        askPriceList            = []
        bidPriceList            = []
        initial_flag            = True
        old_data                = np.empty((0,2), float)


        df_bidPrice = df[df.columns[bid_price_columns]]
        df_askPrice = df[df.columns[ask_price_columns]]


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