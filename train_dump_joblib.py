import joblib # this module saves your model 
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris= load_iris()
X, y = iris.data, iris.target

model= RandomForestClassifier()
model.fit(X,y)   #   creates model
joblib.dump(model, 'model.joblib') # saves your model variable, named "model"