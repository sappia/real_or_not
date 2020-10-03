import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection

# show upto 40 characters instead of default 50
pd.set_option('max_colwidth', 40)
# read the data
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# Look at first 5 rows of training and test data
print("First 5 rows in training data:")
print(train_df.head(5))
print("First 5 rows in test data:")
print(test_df.head(5))

# Look at sample of non-disaster tweets
print(train_df[train_df["target"] == 0]["text"].values[0:5])

# Look at sample of disaster tweets
print(train_df[train_df["target"] == 1]["text"].values[0:5])

# using count vectorizer from scikit-learn
count_vect = feature_extraction.text.CountVectorizer()

# counts for first 5 tweets
sample_train_vect = count_vect.fit_transform(train_df["text"][0:5])

# see vector for first tweet
print(sample_train_vect[0].todense().shape)
print(sample_train_vect[0].todense())

# see vector for second tweet
print(sample_train_vect[1].todense().shape)
print(sample_train_vect[1].todense())

# create vectors for all training tweets
train_vect = count_vect.fit_transform(train_df["text"])
# create vectors for all test tweets
test_vect = count_vect.transform(test_df["text"])

# build a linear model for classification using Ridge regression
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_vect, train_df["target"], cv=3, scoring="accuracy")
# [0.70133964 0.65051221 0.71186441]
# scores = model_selection.cross_val_score(clf, train_vect, train_df["target"], cv=3, scoring="f1")
# [0.59421842 0.56455572 0.64149093]
print(scores)

# predict labels for test tweets
clf.fit(train_vect, train_df["target"])


sample_submission = pd.read_csv("dataset/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vect)

# create a submission csv file
sample_submission.to_csv("submission.csv", index=False)
