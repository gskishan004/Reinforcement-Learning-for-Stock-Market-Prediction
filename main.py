import gym
import json
import argparse
import datetime
import configparser 

import datetime as dt
import pandas as pd
import numpy as np

from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy as td3MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines.common.policies 	import MlpPolicy
from stable_baselines.common.policies 	import MlpLstmPolicy
from stable_baselines.common.policies 	import MlpLnLstmPolicy
from stable_baselines.common.policies 	import CnnPolicy
from stable_baselines.common.policies 	import CnnLstmPolicy
from stable_baselines.common.policies 	import CnnLnLstmPolicy
from stable_baselines.common.vec_env 	import DummyVecEnv
from stable_baselines 					import PPO2

from env.securities_trading_env 		import securities_trading_env


'''
Stable baselines framework
==========================

Policies:
---------
MlpPolicy		Policy object that implements actor critic, using a MLP (2 layers of 64)


Optimization algorithm:
------------------
PP02			Combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor).
TD3 			Is an off-policy algorithm. TD3 can only be used for environments with continuous action spaces.

'''

config = configparser.ConfigParser()
config.read('config.ini')

data_file 		= config['MAIN']['Data']
test_steps 		= int(config['MAIN']['TestSize'])
train_steps 	= int(config['MAIN']['TrainSize'])
no_of_agents 	= int(config['MAIN']['BotNumber'])
aglo 			= str(config['MAIN']['Policy'])

parser = argparse.ArgumentParser()
df = pd.read_csv(data_file)
env = DummyVecEnv([lambda: securities_trading_env(df)])

#-p MlpLstmPolicy -a PP02 
parser.add_argument("-l", "--load",  	dest = "loadFlag", 		default = "no_path", 	help="Only load the model")
parser.add_argument("-v", "--verbose",  dest = "verboseFlag", 	default = 1, 			help="Flag for verbose either 1 or 0")


args = parser.parse_args()

datenow = datetime.datetime.now().strftime("%I%M%p-%d%B%Y")

print(args.loadFlag)

if (args.loadFlag == "no_path"):

	if (aglo == "PPO2"):
		model = PPO2(MlpPolicy, env, verbose=int(args.verboseFlag))
		model.learn(total_timesteps=train_steps)
		obs = env.reset()

		for i in range(test_steps):
		    action, _states = model.predict(obs)
		    obs, rewards, done, info = env.step(action)
		    env.render()

		model_save = "./save/PPO2"+"-"+datenow+".h5"
		print("Model saved as: ",model_save)
		model.save(model_save)


	elif (aglo == "TD3"):
		model = TD3(td3MlpPolicy, env, verbose=int(args.verboseFlag))
		model.learn(total_timesteps=train_steps, log_interval=10)
		obs = env.reset()

		for i in range(test_steps):
		    action, _states = model.predict(obs)
		    obs, rewards, done, info = env.step(action)
		    env.render()

		model_save = "./save/TD3"+"-"+datenow+".h5"
		print("Model saved as: ",model_save)
		model.save(model_save)

else:
	if (aglo == "PPO2"):
		model = PPO2(MlpPolicy, env, verbose=int(args.verboseFlag))
		model.load(args.loadFlag)
		obs = env.reset()

		for i in range(test_steps):
		    action, _states = model.predict(obs)
		    obs, rewards, done, info = env.step(action)
		    env.render()

	elif (aglo == "TD3"):
		model = TD3(td3MlpPolicy, env, verbose=int(args.verboseFlag))
		model.load(args.loadFlag)
		obs = env.reset()

		for i in range(test_steps):
		    action, _states = model.predict(obs)
		    obs, rewards, done, info = env.step(action)
		    env.render()