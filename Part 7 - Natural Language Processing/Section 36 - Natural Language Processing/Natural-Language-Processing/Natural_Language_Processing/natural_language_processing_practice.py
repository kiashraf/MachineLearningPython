# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:10:30 2018

@author: KAshraf
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter ='\t',quoting=3)

# Cleaning the texts!!
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for index in dataset.index:
	review = re.sub('[^a-zA-Z]', ' ' ,  dataset['Review'][index])
	review = review.lower()
	review = review.split()
	#Remove words not useful for for NLP like the,and, or etc.
	review = [ word for word in review if  not word  in set(stopwords.words('english')) ]
	#Stemming : finding root of the words
	porterStemmer = PorterStemmer()
	review = [porterStemmer.stem(wrd) for wrd in review]
	review = ' '.join(review)
	corpus.append(review)


#Creating Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
countVectorizer = CountVectorizer(max_features=1500)
X = countVectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#Splitting the data into train and test
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training Set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict the test result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)









	

























