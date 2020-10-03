# Disaster - real or not

## Overview
[Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview) is a NLP learning project/competition on [Kaggle](https://www.kaggle.com/). The challenge is to build a machine learning model that predicts which __Tweets__ are about __real disasters__ and which ones are not.

## Contents
This project consists of training data and test data provided by Kaggle, python files for the different machine learning models that were built to make predictions and a util file with functions written to preprocess/clean tweets. Below are the files in the project and their purpose:
  * __dataset/train.csv__ contains training tweets with target labels i.e. label saying a tweet is disaster or not
  * __dataset/test.csv__ contains test tweets without target labels which can be used to test the machine learning models built
  * __dataset/sample\_submission.csv__ contains sample submission format of prediction results to submit for the competition
  * __utils/text\_utils.py__ contains functions to preprocess/clean tweets like removing punctuations and stopwords and lemmatizing words
  * __requirements.txt__ contains all the libraries installed for this project
  * __ridge\_classifier\_no\_preprocessing.py__ contains model classifying tweets using Ridge Classifier without preprocessing tweets
  * __ridge\_classifier.py__ contains model classifying tweets using Ridge Classifier after preprocessing tweets
  * __random\_forest.py__ contains model classifying tweets using Random Forest Classifier after preprocessing tweets
  * __gradient\_boost.py__ contains model classifying tweets using Gradient Boost Classifier after preprocessing tweets

## Variations used in the models
* Ridge classifier was used with count vectorizer for vectorizing the tweets. Ridge classifier was run with and without preprocessing tweets.
* Randome Forest Classifier and Gradient Boost Classifier were used with Tfidf vectorizer for vectorizing the tweets. Both were run after preprocessing tweets. Both were run with different variations of estimators (100 and 1000).

## Results
* When the predictions were submitted to Kaggle for each above variations, the score was 0.725 and 0.789 with the highest score acheived when using random forest classifier with 100 estimators i.e. 100 trees in the forest.

## Future work
* Use RNN and LSTM to classify tweets.

