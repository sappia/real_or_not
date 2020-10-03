import pandas as pd
from utils import text_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, model_selection

# show upto 40 characters instead of default 50
pd.set_option('max_colwidth', 100)
# read the data
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# remove punctuations, tokenize, remove stop words and lemmatize the words from lowercased tweets in training data
train_df['text_clean'] = train_df['text'].apply(lambda x: text_utils.clean_text(x.lower()))

# remove punctuations, tokenize, remove stop words and lemmatize the words from lowercased tweets in test data
test_df['text_clean'] = test_df['text'].apply(lambda x: text_utils.clean_text(x.lower()))

# Look at first 5 rows of training and test data
print("First 5 rows in training data:")
print(train_df['text'].head(5))
print("First 5 rows in training data - cleaned:")
print(train_df['text_clean'].head(5))

print("First 5 rows in test data:")
print(test_df['text'].head(5))
print("First 5 rows in test data - cleaned:")
print(test_df['text_clean'].head(5))

# Apply Count Vectorizer
count_vect = CountVectorizer(analyzer=text_utils.clean_text)

# create vectors for all training tweets
train_vect = count_vect.fit_transform(train_df["text"])
# create vectors for all test tweets
test_vect = count_vect.transform(test_df["text"])

# build a linear model for classification using Ridge regression
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_vect, train_df["target"], cv=3, scoring="accuracy")
# [0.714342   0.65602837 0.69846275]
# scores = model_selection.cross_val_score(clf, train_vect, train_df["target"], cv=3, scoring="f1")
# [0.59878251 0.53089737 0.60949464]
print(scores)

# fit the train datapredict labels for test tweets
clf.fit(train_vect, train_df["target"])

sample_submission = pd.read_csv("dataset/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vect)

print(sample_submission.head(5))

# create a submission csv file
sample_submission.to_csv("submission_1.csv", index=False)
