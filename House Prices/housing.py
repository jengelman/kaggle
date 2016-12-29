import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import learning_curve, cross_val_score, validation_curve, cross_val_predict
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.feature_selection import SelectFromModel, f_regression, SelectPercentile, SelectKBest
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import re

def rmsle(predicted, actual):
	return np.sqrt(mean_squared_error(np.log(predicted+1),np.log(actual+1)))

class HousingModel():
	
	def __init__(self):
		self.train_df = pd.read_csv('train.csv')
		self.test_df = pd.read_csv('test.csv')

	def clean_data(self, df):
		#Encode Categorical Variables
		cat_vars = df.select_dtypes(include=['object']).columns.tolist()
		for var in cat_vars:
			dummies = pd.get_dummies(df[var],prefix=var, dummy_na=True)
			df = pd.concat([df, dummies],axis=1)

		#remove old stuff
		df = df.drop(cat_vars,axis=1)
		#generate interaction terms
		num_vars = df.select_dtypes(include=['float','int']).columns.tolist()
		num_vars = filter(lambda x: x != 'SalePrice', num_vars)

		num_vars = [(x, y) for x in num_vars for y in num_vars if (x != y)]
		for x, y in num_vars:
			df[x+'+' +y] = df[x] + df[y]
			df[x +'-' +y] = df[x] - df[y]
			df[y +'-' +x] = df[y] - df[x]
			df[x +'x'+ y] = df[x]*df[y]
			df[x +'/'+ y] = df[x]/df[y]
			df[y +'/'+ x] = df[y]/df[x]

		return df.fillna(0)

	def train_model(self):

		#run training
		print 'training'
		self.x_train = self.train_df.drop(['Id','SalePrice'],1)
		print self.x_train[pd.isnull(self.x_train)]
		self.features = self.x_train.columns.tolist()
		self.y_train = self.train_df['SalePrice']

		rf = RandomForestRegressor(n_estimators=100, max_depth=7)
		xgb = XGBRegressor(n_estimators=330,min_child_weight=4,
			subsample=.9, colsample_bytree=.5, reg_alpha=.3, reg_lambda=.6)

		self.transformation = SelectFromModel(rf,threshold='median')#SelectPercentile(f_regression, 75)
		self.clf = Pipeline([('feat', self.transformation),('xgb',xgb)])
		scorer = make_scorer(rmsle, greater_is_better=False)
		self.clf.fit(self.x_train, self.y_train)
		features = self.clf.named_steps['feat']
		for feature in self.x_train.columns[features.transform(np.arange(len(self.x_train.columns)))]:
			print feature
		scores = cross_val_score(self.clf, self.x_train, self.y_train, cv=5, verbose=0,scoring=scorer)
		print "RMSLE: %f +/- %f" % (scores.mean(), 2*scores.std()) 

	def run_model(self):
		#run prediction
		print 'predicting'
		missing_features = np.setdiff1d(self.features, self.test_df.columns) 
		for feature in missing_features:
			self.test_df[feature] = 0
		x_test = self.test_df.drop('Id', 1).values
		prediction = self.clf.predict(x_test)
		return prediction

	def save_results(self, prediction):
		df3 = pd.DataFrame({'Id':self.test_df.Id, 'SalePrice':prediction})
		df3.to_csv('submission.csv',index=False)

	def validate_model(self, param, param_range):
		scorer = make_scorer(rmsle, greater_is_better=False)
		train_scores, test_scores = validation_curve(
			self.clf, self.x_train, self.y_train, param, param_range,
			verbose=0,cv=5,scoring=scorer)
		
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
		scorer = make_scorer(rmsle, greater_is_better=False)
		train_sizes, train_scores, test_scores = learning_curve(
			self.clf, self.x_train, self.y_train, train_sizes=np.linspace(.1, 1., 10), 
			verbose=0,cv=5, scoring=scorer)
		 
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

if __name__ == '__main__':
	model = HousingModel()
	model.train_df = model.clean_data(model.train_df)
	model.test_df = model.clean_data(model.test_df)
	model.train_model()
	#param_range = [x for x in range(120, 150)]
	#print model.clf.get_params().keys()
	#model.validate_model('feat__k',param_range)
	#prediction = model.run_model()
	#model.save_results(prediction)
	model.plot_results()
