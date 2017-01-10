import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR
from sklearn.model_selection import learning_curve, cross_val_score, validation_curve
from sklearn.feature_selection import SelectFromModel, f_regression, SelectPercentile, SelectKBest
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import re

def munge(df):

	num_vars = df.select_dtypes(include=['float','int']).columns.tolist()
	num_vars = filter(lambda x: x not in ['SalePrice','Id'], num_vars)

	num_vars = [(x, y) for x in num_vars for y in num_vars if (x != y)]
	for x, y in num_vars:
		df[x+'+' +y] = df[x] + df[y]
		df[x +'-' +y] = df[x] - df[y]
		df[y +'-' +x] = df[y] - df[x]
		df[x +'x'+ y] = df[x]*df[y]

	#Encode Categorical Variables
	cat_vars = df.select_dtypes(include=['object']).columns.tolist()
	for var in cat_vars:
		dummies = pd.get_dummies(df[var],prefix=var, dummy_na=True)
		df = pd.concat([df, dummies],axis=1)

	#remove old stuff
	df = df.drop(cat_vars,axis=1)

	return df.fillna(0)

def generate_munged():
	for x in ['train', 'test']:
		path = x + '.csv'
		df = pd.read_csv(path)
		munged_df = munge(df)
		munged_df.to_csv('munged_' + x + '.csv',index=False)

#custom Root Mean Square Log Error scorer
def rmsle(predicted, actual):
	return np.sqrt(mean_squared_error(np.log(predicted+1),np.log(actual+1)))

class HousingModel():
	def __init__(self):
		print 'loading data'
		self.train_df = pd.read_csv('munged_train.csv')
		self.test_df = pd.read_csv('munged_test.csv')
		self.scorer = make_scorer(rmsle, greater_is_better=False)

	def train_model(self):
		#run training
		print 'training'
		x_train = self.train_df.drop(['Id','SalePrice'],1)
		self.y_train = self.train_df['SalePrice']

		rf = RandomForestRegressor(n_estimators=100, max_depth=7,n_jobs=-1)
		transformation = SelectFromModel(rf)
		transformed_train = transformation.fit_transform(x_train, self.y_train)
		self.selected_features = x_train.columns.values[transformation.get_support()]
		print '# Features selected: ', len(self.selected_features)

		print 'Fitting model'
		self.clf = XGBRegressor(n_estimators=330,min_child_weight=4,
			subsample=.9, colsample_bytree=.5, reg_alpha=.3, reg_lambda=.6)
		

		self.clf.fit(transformed_train, self.y_train)
		print 'calculating cross val score'
		scores = cross_val_score(self.clf, transformed_train, self.y_train, cv=5,scoring=self.scorer)
		print "RMSLE: %f +/- %f" % (scores.mean(), 2*scores.std()) 

	def validate_model(self, param, param_range):
		train_scores, test_scores = validation_curve(
			self.clf, self.x_train, self.y_train, param, param_range,
			verbose=0,cv=5,scoring=self.scorer)
		
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)

		# Plot the average training and test score lines at each training set size
		plt.plot(param_range, train_scores_mean, 'o-', color="b", label="Training score")
		plt.plot(param_range, test_scores_mean, 'o-', color="r", label="Test score")
		 
		# Plot the std deviation as a transparent range at each training set size
		plt.fill_between(param_range, train_scores_mean - train_scores_std, 
						train_scores_mean + train_scores_std, 
		                alpha=0.1, color="b")

		plt.fill_between(param_range, test_scores_mean - test_scores_std, 
						test_scores_mean + test_scores_std, 
		                alpha=0.1, color="r")
		 
		# Draw the plot and reset the y-axis
		plt.grid()
		plt.draw()
		plt.show()

	def plot_results(self):
		train_sizes, train_scores, test_scores = learning_curve(
			self.clf, self.x_train, self.y_train, train_sizes=np.linspace(.1, 1., 10), 
			verbose=0,cv=5, scoring=self.scorer,n_jobs=-1)
		 
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		# Plot the average training and test score lines at each training set size
		plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Test score")
		 
		# Plot the std deviation as a transparent range at each training set size
		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
						train_scores_mean + train_scores_std, 
		                alpha=0.1, color="b")

		plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
						test_scores_mean + test_scores_std, 
		                alpha=0.1, color="r")
		 
		# Draw the plot and reset the y-axis
		plt.grid()
		plt.draw()
		plt.show()

	#generate submission dataframe
	def run_model(self):
		print 'predicting'
		missing_features = np.setdiff1d(self.selected_features, self.test_df.columns) 
		for feature in missing_features:
			self.test_df[feature] = 0

		x_test = self.test_df[self.selected_features].values
		prediction = self.clf.predict(x_test)
		return prediction

	#format and save submission dataframe
	def save_results(self, prediction):
		df3 = pd.DataFrame({'Id':self.test_df.Id, 'SalePrice':prediction})
		df3.to_csv('submission.csv',index=False)

if __name__ == '__main__':
	#generate new features and save to disk
	#generate_munged()
	
	#train model
	model = HousingModel()
	model.train_model()
	#model.plot_results()
	
	#run on test data
	prediction = model.run_model()
	model.save_results(prediction)
