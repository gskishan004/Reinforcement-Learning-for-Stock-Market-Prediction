# Reinforcement Learning for stock market prices forecast #

This repo contains a novel approach and a custom RL environment for the selection of best models from a pool to accurately forcast stock market.

For more details contact: ishan.khanka@gmail.com

To-Do:
* general-
	* Add a functionality to run for all stocks by default or for only a singular stock
	* Functionality to save the results in a file, if possible CSV
	* Move the config file from the code to seperate file
	* Functionality to automatically detect the dates

* env-
	* Change the definition of done
	* Add a safety feature where there cant be 0 models in the list
	* Change the reward function

* modules-
	* Add a section to calculate the total number models for each target value for a stock


Fixed
* env-
	* problems in some global and self variables like model_dict, current_eval_date
	* making code more modular by taking target name and stock name as params to the class 
	* Initial check for consistency in frame shape, done by adding file and function : modules/stockModels/findConstantModelsList