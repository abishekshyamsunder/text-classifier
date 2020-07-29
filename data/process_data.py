import pandas as pd
import numpy as np
import sqlalchemy
import sys

def load_and_merge_datasets(csv_file_1 = "disaster_messages.csv",csv_file_2 = "disaster_categoriess.csv"):
	'''
	For Loading and merging datasets
	Input:
		2 Strings, that are names to .CSV files (with file path if necessary)
	Output:
		3 Pandas Dataframes, the original two dataframes, and the merged dataframe
	'''
	messages = pd.read_csv(csv_file_1)
	categories = pd.read_csv(csv_file_2)
	df = pd.merge(messages,categories,on='id')
	return (messages,categories,df)

def clean_categories_df(categories):
	'''
	For cleaning and restructuring categories DataFrame
	Input:
		Pandas DataFrame containing the categories
	Output:
		Cleaned pandas Dataframe
	'''
	# Extracting list of column names from data in first row
	categories = categories['categories'].str.split(";",expand=True)
	row = categories.iloc[0,:]
	category_colnames = row.apply(lambda x:x[:-2])
	categories.columns = category_colnames

	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].apply(lambda x: x[-1])	
		# convert column from string to numeric
		categories[column] = categories[column].apply(lambda x: int(x))
	return categories

def concat_dfs(df,categories):
	'''
	Merge cleaned Categories and complete df
	Also drop duplicate entries based on id
	Input:
		Two pandas dataframes, for merging
	Output:
		Merged DataFrame
	'''
	df = df.drop('categories',axis=1)
	categories['id'] = df['id']
	df = pd.merge(df,categories,on='id')
	df = df.drop_duplicates(subset=['id'],keep='first')
	return df

def push_to_db(df,db):
	'''
	Save Final cleaned and merged pandas DataFrame in Database
	Input:
		DataFrame to save and database to save it
	Ouput:
		None
	'''
	engine = sqlalchemy.create_engine('sqlite:///'+db)
	df.to_sql('messages', engine, index=False)


def main():
	if len(sys.argv) !=4:
		print('Please run script as `python process_data.py disaster_messages.csv disaster_categoriess.csv database.db`')
	else:
		args = sys.argv
		csv_file_1 = args[1]
		csv_file_2 = args[2]
		db = args[3]
		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(csv_file_1, csv_file_2))
		(messages,categories,df) = load_and_merge_datasets(csv_file_1,csv_file_2)
		print('Cleaning data...')
		categories = clean_categories_df(categories)
		df = concat_dfs(df,categories)
		print('Saving data...\n    DATABASE: {}'.format(db))
		push_to_db(df,db)
		print(df)

if __name__=="__main__":
	main()


