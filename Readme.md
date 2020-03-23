# Reinforcement Learning for trading securities

## Note: Please Document any changes before requesting a commit to the master repo

For more details contact: ishan.khanka@gmail.com

Dir Structure
* ./config.ini 						: contains configuration 
* ./data /							: Location of input files (change input location in config)
* ./env/securities_trading_env.py	: Env for agent
* ./modules/preprocessing.py		: Making concat.csv
* ./main.py							: Driver code 
* ./save/							: Location for saved models


Flag Information
"-p", "--policy"	default = "MlpPolicy" 	--> RL Policy
"-a", "--algorithm"	default = "PP02"		--> Optimization algorithm
"-l", "--load"		default = "no_path" 	--> Only load the model
"-v", "--verbose"	default = 1				--> Flag for verbose either 1 or 0

To-Do:
* general-
	* Add more params to reward function 
	* Add more policies to the list - Currenly working policy is MLPPolicy
* env-
	* Add Connection to RabbitMQ


Fixed/ Done
* general-
	* Added option to start agent with a specified observation window in config.ini
	* Added config.ini to reduce the number of flags 
	* Added Hold Option
	* Create basic GYM environmet for trading securities 
	* Added stable baselines for RL 
	* Added flags for model selction
	* Added ability to save models with unique key based on time and date


Parent GitHub Link for archives and version control: https://github.com/gskishan004/Reinforcement-Learning-for-trading-securities