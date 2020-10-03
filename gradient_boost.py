import pandas as pd
from utils import text_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection

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

# Apply Tfidf Vectorizer
tfidf_vect = TfidfVectorizer(analyzer=text_utils.clean_text)

# create vectors for all training tweets
train_vect = tfidf_vect.fit_transform(train_df["text"])
# create vectors for all test tweets
test_vect = tfidf_vect.transform(test_df["text"])

# Instantiate RandomForestClassifier
gb = GradientBoostingClassifier(n_estimators=1000, max_depth=None, learning_rate=150)
# Fit the data
gb_model = gb.fit(train_vect, train_df["target"])


scores = model_selection.cross_val_score(gb, train_vect, train_df["target"], cv=3, scoring="accuracy")
# [0.68360914 0.62332545 0.66456445]
#scores = model_selection.cross_val_score(gb, train_vect, train_df["target"], cv=3, scoring="f1")
# [0.56521739 0.47841914 0.54630109]
print(scores)

# def train_GB(n_est, max_depth, lr):
#   gb = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, learning_rate=lr)
#   gb_model = gb.fit(train_vect, train_df["target"])
#   y_pred = gb_model.predict(test_vect)
#   scores = model_selection.cross_val_score(rf, train_vect, train_df["target"], cv=3, scoring="accuracy")
#   print('Est: {} / Depth: {} ----> {}'.format(n_est, depth, scores))
#
# for n_est in [50, 100, 150]:
#   for max_depth in [3, 7, 11, 15]:
#     for lr in [0.01, 0.1, 1]:
#       train_GB(n_est, max_depth, lr)


sample_submission = pd.read_csv("dataset/sample_submission.csv")
sample_submission["target"] = gb.predict(test_vect)

print(sample_submission.head(5))

# create a submission csv file
sample_submission.to_csv("submission_3.csv", index=False)
