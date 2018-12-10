import statistics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import json



#Loading dataset
path="kidnap/Kidnap case dataset.csv"
data = pd.read_csv(path, header=0)

#Defining dataset attributes
#X=data[['surrounding people', 'place', 'diatance from location', 'gun','knife']]  # Features
X=data[['number of surounding people','guns','knife','place','age','work','sex']]
y=data['kidnap']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

model1 = LogisticRegression(random_state=0,C=0.1)
model2 = RandomForestClassifier(n_estimators=26,oob_score=True,min_samples_leaf=5)
model3 = SVC(kernel='linear',gamma=1,C=10)
model = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('svc', model3)], voting='hard')
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
#print(model.predict([[5, 2, 1, 1, 23, 1, 1]]))

filename = 'finalized_model.sav'
joblib.dump(model, filename)
