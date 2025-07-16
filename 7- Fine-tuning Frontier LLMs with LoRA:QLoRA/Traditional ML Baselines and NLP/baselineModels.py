import os
import sys
import pickle
import random
import numpy as np
import pandas as pd

# Import for traditional machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

def bow_linear_regression_model(documents,
                                prices,
                                max_features=1000,
                                stop_words="english",
                                PREFIX="Price is $"):
    """
        Train a Bag-of-Words + LinearRegression on `train` examples and
        return a predictor(item) function.

        text_fn:   item -> str  (what text to vectorize)
        target_fn: item -> float (what numeric target to predict)
    """

    # Use the CountVectorizer for a Bag of Words model
    vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    X_train = vectorizer.fit_transform(documents)

    # Fit regressor
    lr_model = LinearRegression()
    lr_model.fit(X_train, prices)

    # Build the predictor closure
    def predict(item):
        x = vectorizer.transform([item.testPrompt(prefix=PREFIX)])
        y_pred = lr_model.predict(x)[0]
        return max(y_pred, 0.0)

    predict.__name__ = "bow_linear_regression"
    return predict


#------------------------------------------------------------------------------------------
#       5- word2vec & Linear Regression or Support Vector Machines or Random Forest
#------------------------------------------------------------------------------------------


def document_vector(w2v_model, doc):
    """
        Given a trained gensim w2v_model and one raw doc string,
        returns the mean of its word vectors (or zero-vector).
    """
    doc_words = simple_preprocess(doc)
    word_vectors = [w2v_model.wv[word] for word in doc_words if word in w2v_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(w2v_model.vectore_size)


def w2v_regression_model(documents: list[str],
                         prices: list[float],
                         regressor_cls,
                         # W2V kwargs
                         vector_size=400,
                         window=5,
                         min_count=1,
                         workers=8,
                         # any kwargs for the regressor
                         regressor_kwargs: dict = {}
                         ):
    """
        Train Word2Vec on `documents`, average into X_w2v, fit `regressor_cls` on (X_w2v, prices),
        and return a predict(item) function.
        Note: regressor_cls can be LinearRegression, LinearSVR, RandomForestRegressor and so on.
    """
    # Preprocess the documents
    tokens = [simple_preprocess(docs) for docs in documents]
    # Train w2v
    w2v_model = Word2Vec(sentences=tokens,
                         vector_size=vector_size,
                         window=window,
                         min_count=min_count,
                         workers=workers)
    # Build training matrix
    X_w2v = np.array([document_vector(w2v_model, doc) for doc in documents])

    # Fit the chosen regressor
    linear_regressor = regressor_cls(**regressor_kwargs)
    linear_regressor.fit(X_w2v, prices)

    def predict(item):
        doc = item.testPrompt(prefix="Price is $")
        doc_vector = document_vector(w2v_model, doc)
        return max(float(linear_regressor.predict([doc_vector])[0]), 0.0)

    predict.__name__ = regressor_cls.__name__.lower()
    return predict








