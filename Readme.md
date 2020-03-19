# Reinforcement Learning for trading securities

## Note: Please Document any changes before requesting a commit to the master repo

For more details contact: ishan.khanka@gmail.com

Code Structure
* ./data /							: Location of concat.csv
* ./env/securities_trading_env.py	: Env for agent
* ./modules/preprocessing.py		: Making concat.csv
* ./main.py							: Driver code 
* ./save/							: Location for saved models

Flag Information
"-p", "--policy"	default = "MlpPolicy" 	--> RL Policy
"-a", "--algorithm"	default = "PP02"		--> Optimization algorithm
"-t", "--testSize"	default = 50			--> Test Size
"-l", "--load"		default = "no_path" 	--> Only load the model
"-v", "--verbose"	default = 1				--> Flag for verbose either 1 or 0

To-Do:
* general-
	* Option to start agent with a specified observation window
	* Add option for hold
	* Add more params to reward function 
	* Add more policies to the list - Currenly working policy is MLPPolicy
* env-
	* Add Connection to RabbitMQ


Fixed/ Done
* general-
	* Create basic GYM environmet for trading securities 
	* Added stable baselines for RL 
	* Added flags for model selction
	* Added ability to save models with unique key based on time and date


Parent GitHub Link for archives and version control: https://github.com/gskishan004/Reinforcement-Learning-for-trading-securities