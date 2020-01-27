import gym
import json
import datetime as dt
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from modules.findUnique 	import findUniqueStocks
from modules.findUnique 	import findUniqueTargets
from modules.stockModels	import findConstantModelsList
from env.ModelSelectionEnv 	import ModelSelectionEnv

#Turn this to 1 for debugging info, later put this in config
debugging_flag = 1


df = pd.read_csv('./data/data.csv')

stocksList = findUniqueStocks(df)
targetList = findUniqueTargets(df)
print("Number of Unique Stocks: ", len(stocksList))
print("Number of Unique Targets: ", len(targetList))


STOCK_NAME          = "FAST"
TARGET_NAME         = "1P28D"

modelList = findConstantModelsList(df,STOCK_NAME, TARGET_NAME,debugging_flag)
print("STOCK_NAME:",STOCK_NAME,"TARGET_NAME:",TARGET_NAME, "modelList:",modelList)
if(len(modelList)==0):
	print("Data not suitable for the selected stock as no common list of models accross all the dates could be found")

env = DummyVecEnv([lambda: ModelSelectionEnv(df,STOCK_NAME,TARGET_NAME, modelList)])



model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=28)

obs = env.reset()
for i in range(26):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

#model.load("./save/cartpole-dqn.h5"
model.save("./save/model_save.h5")