#This function finds the constant models present for all days
#If a model is missing even once then it will be removed from the list

def findConstantModelsList(df, STOCK_NAME, TARGET_NAME,debugging_flag):

	date_count = 0
	
	modelDictWithDates = {}
	finalModelList	= []

	df 				= df[(df['ticker'] == STOCK_NAME) & (df['target_name'] == TARGET_NAME )]
	unique_modelid  = df['modelid'].unique()

	for model in unique_modelid:
		modelDictWithDates[model] = 0

	for date in (df['eval_date'].unique()):
		for model in unique_modelid:
			if df[(df['modelid'] == int(model)) & (df['eval_date'] == date )].empty:
				if(debugging_flag):print ("Data frame empty for model:", model, "eval_date", date)
			else:
				modelDictWithDates[model]+=1
		date_count +=1 

	for model in unique_modelid:
		if modelDictWithDates[model] == date_count:
			finalModelList.append(model)
		else:
			if(debugging_flag): print("Model: ", model," removed from the list as it wasn't present for all dates")


	return (finalModelList)
