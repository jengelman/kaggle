import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, learning_curve, cross_val_score, validation_curve, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, f_classif
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import re

class TitanicModel():
	
	def __init__(self):
		self.train_df = pd.read_csv('~/Downloads/train.csv')
		self.test_df = pd.read_csv('~/Downloads/test.csv')

	def clean_data(self, df):
		#set fares to median of each passenger class
		if len(df.Fare[ df.Fare.isnull() ]) > 0:
		    median_fare = np.zeros(3)
		    for f in range(0,3):                                              
		        median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
		    for f in range(0,3):                                              
		        df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]		

		#set missing ages to median of passenger class
		median_age = df['Age'].dropna().median()
		if len(df.Age[ df.Age.isnull() ]) > 0:
		   	median_age = np.zeros(3)
		   	for f in range(0,3):
		   		median_age[f] = df[ df.Pclass == f+1 ]['Age'].dropna().median()
		   	for f in range(0,3):
		   		df.loc[ (df.Age.isnull()) & (df.Pclass == f+1 ), 'Age'] = median_age[f]		

		#Title Feature
		df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

		# Group low-occuring, related titles together
		df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
		df['Title'][df.Title == 'Mme'] = 'Mrs'
		df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir', 'Rev','Dr'])] = 'Sir'
		df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess','Jonkheer'])] = 'Lady'
		df['Title'][df.Title == 'Master'] = 'Mr'

		#Deck feature from Cabin
		cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
		df.Cabin.fillna('U',inplace=True)
		df['Deck']=df.Cabin.map(lambda x: x[0])

		#set missing embarcation points to most common values
		if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
			df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

		#Encode Categorical Variables
		cat_vars = ['Sex','Deck','Title', 'Embarked', 'Pclass']
		for var in cat_vars:
			dummies = pd.get_dummies(df[var],prefix=var)
			df = pd.concat([df, dummies],axis=1)

		#remove old stuff
		df.drop(['Cabin','Name','Ticket','Sex','Title','Deck', 'Embarked','Pclass'],axis=1,inplace=True)

		#interaction features
		df['Family_Size'] = df.Parch + df.SibSp
		df['Fare per Family Member'] = df.Fare/(df.Family_Size+1)

		#family size categorization
		df['Single'] = df.Family_Size.apply(lambda x: 1 if x == 0 else 0)
		df['Small Family'] = df.Family_Size.apply(lambda x: 1 if 0 < x < 5 else 0)
		df['Large Family'] = df.Family_Size.apply(lambda x: 1 if x > 4 else 0)

		return df

	def train_model(self):
		#run training
		print 'training'
		self.x_train = self.train_df.drop(['Survived','PassengerId'],1)
		self.features = self.x_train.columns.tolist()
		self.x_train = scale(self.x_train)
		self.y_train = self.train_df['Survived']

		#throw a bunch of models at it
		xgb = CalibratedClassifierCV(XGBClassifier(subsample=.85, colsample_bytree=.9,
						max_depth=7, min_child_weight=5,n_estimators=300,
						reg_alpha = .3, reg_lambda=.7, colsample_bylevel=.7,
						scale_pos_weight=.525),method='isotonic')

		rf = RandomForestClassifier(n_estimators=107,max_depth=7)
		lr = LogisticRegression(C=4,class_weight='balanced')

		svc = CalibratedClassifierCV(SVC(probability=True,class_weight='balanced',gamma=.2),method='isotonic')

		extra = CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=15,max_depth=12),method='isotonic')

		voting_clf = VotingClassifier(estimators=(('xgb',xgb), ('rf',rf),('lr',lr),('svc',svc),('extra',extra)),voting='soft', weights=[2,2,1,1,1])
		feat = SelectFromModel(rf,threshold=.04)
		self.clf = Pipeline([('feat', feat),('voting_clf',voting_clf)])
		self.clf.fit(self.x_train, self.y_train)
		print self.clf.named_steps['feat'].get_support().sum()
		scores = cross_val_score(self.clf, self.x_train, self.y_train,cv=5, verbose=0,n_jobs=-1)
		print "Accuracy of Soft voting combiner: %f +/- %f " % (scores.mean()*100, 2*scores.std()*100) 

	def run_model(self):
		#run prediction
		print 'predicting'
		x_test = self.test_df.drop('PassengerId', 1)
		x_test['Deck_T'] = 0
		x_test = x_test[self.features]
		prediction = self.clf.predict(x_test)
		return prediction

	def save_results(self, prediction):
		df3 = pd.DataFrame({'PassengerId':self.test_df.PassengerId, 'Survived':prediction})
		df3.to_csv('submission6.csv',index=False)

	def validate_model(self, param, param_range):
		train_scores, test_scores = validation_curve(
			self.clf, self.x_train, self.y_train, param, param_range,
			verbose=0,cv=5,n_jobs=-1)
		 
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
			verbose=0,cv=5,n_jobs=-1)
		 
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
	model = TitanicModel()
	model.train_df = model.clean_data(model.train_df)
	model.test_df = model.clean_data(model.test_df)
	model.train_model()
	#param_range = []
	#model.validate_model('feat__threshold',param_range)
	prediction = model.run_model()
	model.save_results(prediction)
	#model.plot_results()
