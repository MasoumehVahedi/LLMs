
# Traditional ML Baselines for Price Prediction

Before we dive into LLM-based solutions, we first establish a set of traditional ML baselines to predict product prices. These models—from linear regression on hand-crafted features to Random Forests and SVR on word2vec embeddings—give us a clear, quantitative starting point for our price-prediction business problem.

## Traditional ML Methods

- **Feature engineering & Linear Regression**  
  Manually craft features from raw data and fit a linear regression model.

- **Bag of Words & Linear Regression**  
  Vectorize text into Bag-of-Words counts and apply linear regression.

- **word2vec & Linear Regression**  
  Embed words with pre-trained word2vec and train a linear regression model.

- **word2vec & Random Forest**  
  Use word2vec embeddings as input to a Random Forest regressor for non-linear learning.

- **word2vec & SVR**  
  Feed word2vec embeddings into a Support Vector Regression model for robust predictions.

---
