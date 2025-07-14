import os
import sys
import pickle
import random
import numpy as np
import pandas as pd

# Import for traditional machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

# Import NLP
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer



#---------------------------------------
#         1- Random Model
#---------------------------------------

def random_price(item):
    return random.randrange(1, 1000)


#---------------------------------------
#         2- Constant Model
#---------------------------------------

def constant_price(train):
    prices = [record.price for record in train]
    avg_prices = sum(prices) / len(prices)

    def predict(item):
        return avg_prices

    predict.__name__ = "constant_model"
    return predict


#------------------------------------------------------------------
#         3- Feature engineering & Linear Regression
#------------------------------------------------------------------

def linear_regression_model(df_train, df_test, target: str):
    """
       Train a LinearRegression on the train set (via feature engineering we did.),
       and return a predict(item) function.
    """
    # Separate features and target
    feature_columns = [col for col in df_train.columns if col != target]
    X_train = df_train[feature_columns]
    y_train = df_train[target]
    X_test = df_test[feature_columns]
    y_test = df_test[target]

    # Fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Linear Regression coefficients:")
    for feature, coef in zip(feature_columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Predict the test set and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

    return model, feature_columns


def linear_regression_price(model, feature_columns, feature_fn):
    """
        Returns a function item→price that avoids per-call DataFrame creation.
    """
    def pricer(item):
        features = feature_fn(item)
        row = [features[col] for col in feature_columns]
        return model.predict([row])[0]   # predict on a 1×N array
    pricer.__name__ = "linear_regression"
    return pricer



#------------------------------------------------------------
#         4- Bag of Words & Linear Regression
#------------------------------------------------------------

def bow_and_linear_regression(train, PREFIX):
    prices = np.array([float(item.price) for item in train])
    documents = [item.testPromt(PREFIX) for item in train]

    # Use the CountVectorizer for a Bag of Words model
    np.random.seed(42)
    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(documents)
    lr_model = LinearRegression().fit(X, prices)
    







#----------------------------------------------------------
#         5- word2vec & Linear Regression
#----------------------------------------------------------









#-------------------------------------------------
#         6- word2vec & Random Forest
#-------------------------------------------------







#----------------------------------------
#         7- word2vec & SVM
#----------------------------------------




