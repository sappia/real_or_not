import string
import re
import nltk


# function to remove punctuations from a given text
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct


# function to tokenize a given text
def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


stopwords = nltk.corpus.stopwords.words('english')


# function to remove stopwords from a given text list
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopwords]
    return text


ps = nltk.PorterStemmer()


# function to stem words
def stemming(tokenized_list):
    text = [ps.stem(word) for word in tokenized_list]
    return text


wn = nltk.WordNetLemmatizer()


# function to lemmatize words
def lemmatizing(tokenized_list):
    text = [wn.lemmatize(word) for word in tokenized_list]
    return text


# function to do all the above steps of cleaning
def clean_text(text):
    # remove punctuations from a given text
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    # tokenize a given text
    tokens = re.split('\W+', text_nopunct)
    # remove stopwords from a given text list and lemmatize the words
    text_clean = [wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text_clean
