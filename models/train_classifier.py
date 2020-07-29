import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
import numpy as np
import sqlalchemy
import sys

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from pipelinehelper import PipelineHelper


from sklearn import datasets
import pickle

def load_data(db):
	'''
	Load Data From Database
	Input:
		Name of Database
	Output:
		Data extracted from table in database (x and y)
			X is messages column in table
			Y is DataFrame of 
	'''
	engine = sqlalchemy.create_engine('sqlite:///'+db)
	df = pd.read_sql('select * from messages',engine)
	X = df['message']
	Y = df.iloc[:,4:]
	return (X,Y)

def tokenize(text):
	'''
	Custom tokenise function to pass as argument to CountVectorizer
	Input:
		Text column from a DataFrame
	Output:
		Output Tokens
	'''
	tokens = word_tokenize(text)
	lemmatizer = WordNetLemmatizer()

	clean_tokens = []
	for tok in tokens:
		clean_tok = lemmatizer.lemmatize(tok).lower().strip()
		clean_tokens.append(clean_tok)

	return clean_tokens

def my_custom_loss_func(ground_truth, predictions):
	'''
	Grid Searches do not support multiclass-multioutput.
	This is a custom loss_function ('can be thought of as accuracy') to deal with multiclass output
	Fed to Grid Search as a parameter 
	'''
	acc = (ground_truth==predictions).mean().mean()
	return acc

def build_model(X,y):
	'''
	Setup and pipeline and perform GridSearch.
	Input:
		loaded Data
	output:
		Best model Identified by Grid Search
	'''
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	pipe = Pipeline([
		('vect',CountVectorizer(tokenizer=tokenize)),
		('tfidf',TfidfTransformer()),
		('classifier', PipelineHelper([
			('nn', MultiOutputClassifier(KNeighborsClassifier())),
			('rf', MultiOutputClassifier(RandomForestClassifier())),
		])),
	])

	score = make_scorer(my_custom_loss_func, greater_is_better=True)

	params = {
		'classifier__selected_model': pipe.named_steps['classifier'].generate({
		})
	}
	grid = GridSearchCV(pipe, params,scoring=score, verbose=1)
	grid.fit(X_train, y_train)
	print(grid.best_params_)
	print(grid.best_score_)
	y_pred = grid.predict(X_test)
	print('Accuracy on test set is: ',(y_pred == y_test).mean().mean())
	return grid

def save_model(model_file_name,grid):
	'''
	Save model to pickle file
	Input:
		model and name of file to save model to
	Output:
		None
	'''
	pkl_filename = model_file_name
	with open(pkl_filename, 'wb') as file:
		pickle.dump(grid, file)

def main():
	if len(sys.argv) !=3:
		print('Please run script as `python train_classifier.py  database.db model_save_file.pkl`')
	else:
		args = sys.argv
		db = args[1]
		model_file_name = args[2]
		(X,y) = load_data(db)
		best_model = build_model(X,y)
		save_model(model_file_name,best_model)

if __name__=="__main__":
	main()