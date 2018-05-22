import numpy as np # linear algebra
import pandas as pd # data processing

reviews = pd.read_csv('labeledTrainData.tsv', sep='\t',escapechar='\\')
print(reviews["review"].head())
#parsing

from bs4 import BeautifulSoup
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    return text
reviews["clean_review"] = reviews.review.apply(clean_text)
print(reviews.head())
#tekenizing,extraction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import text
tf_vec = TfidfVectorizer(lowercase=1,min_df=0.001,stop_words=text.ENGLISH_STOP_WORDS)
X_voc = reviews["clean_review"].values
print(tf_vec.fit(X_voc))
X_train = reviews["clean_review"][0:20000].values
X_tf =  tf_vec.transform(X_train)
print(X_tf.shape)
clf = MultinomialNB()
target = reviews["sentiment"][0:20000].values
print(clf.fit(X_tf,target))
X_test = reviews["clean_review"][20000:25000].values.astype("U")
test_x_tf = tf_vec.transform(X_test)
Y_test = reviews["sentiment"][20000:25000]
print(clf.score(test_x_tf,Y_test))