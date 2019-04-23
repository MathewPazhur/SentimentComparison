# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:56:37 2019

@author: Mathew
"""  
# Importing the libraries

import PySimpleGUI as sg          
from sklearn import metrics
import pandas as pd
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re  
from sklearn import svm
import time
    
# Importing the dataset

#sg.PopupOK('Please select csv')
#amazon = pd.read_csv('C:\\Users\\lenovo\\Desktop\\code\\10kreviews.csv')

csv_loc = sg.PopupGetFile('Please enter a dataset')      
amazon = pd.read_csv(csv_loc)

#Removing null entries

amazon.isnull().sum()
amazon = amazon.fillna(' ')
amazon.shape

# Text Length 

amazon['text length'] = amazon['reviewText'].apply(len)

# Creating a class with only 5 and 1 stars 

amazon = amazon[(amazon['overall'] == 1) | (amazon['overall'] == 5)]

# Generating X and Y coordinates

X = amazon['reviewText']
y = amazon['overall']

# Resetting key values

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


# Data Preprocessing
    
documents = []
stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)
    

#TFIDF Vectorization
    
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(documents).toarray()

# Split Dataset into training and testing
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# The callback functions  
    
def button1(): 
    
    Smilli = int(round(time.time() * 1000))    
    
    # Training and Testing
    
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    # Predicting sentiment

    preds = nb.predict(X_test)
    
    
    # Print the Results
    
    Emilli = int(round(time.time() * 1000))
    ntime = Emilli-Smilli
    
    accuracy = round(metrics.accuracy_score(y_test, preds)*100,2)
    precision = round(metrics.precision_score(y_test, preds, average='weighted')*100,2)
    recall = round(metrics.recall_score(y_test, preds, average='weighted')*100,2)
    f1score = round(metrics.f1_score(y_test, preds, average='weighted')*100,2)
    sg.PopupOK('Naive Bayes Results : \n accuracy:', str(accuracy) + '%' ,'\n precision:', str(precision) + '%'  ,'\n recall:', str(recall) + '%'  ,'\n F-measure:', str(f1score) + '%'  ,'\n Computation Time:',str(ntime) + ' ms')    

def button2():      
    
    Smilli = int(round(time.time() * 1000))    
    
    # Training the model
    
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
    classifier.fit(X_train, y_train) 
    
    # Predicting sentiment
    
    y_pred = classifier.predict(X_test)  
    
    Emilli = int(round(time.time() * 1000))
    ntime = Emilli-Smilli
    
    accuracy = round(metrics.accuracy_score(y_test, y_pred)*100,2)
    precision = round(metrics.precision_score(y_test, y_pred, average='weighted')*100,2)
    recall = round(metrics.recall_score(y_test, y_pred, average='weighted')*100,2)
    f1score = round(metrics.f1_score(y_test, y_pred, average='weighted')*100,2)
    
    # Print the Results
    sg.PopupOK('Random Forest Results : \n accuracy:', str(accuracy) + '%' ,'\n precision:', str(precision) + '%'  ,'\n recall:', str(recall) + '%'  ,'\n F-measure:', str(f1score) + '%'  ,'\n Computation Time:',str(ntime) + ' ms')       


def button3():    

    Smilli = int(round(time.time() * 1000))    

    # Training the model

    classifier = svm.SVC(kernel='linear', random_state=12345)
    
    classifier.fit(X_train, y_train)

    # Predicting sentiment

    y_pred = classifier.predict(X_test)

    Emilli = int(round(time.time() * 1000))
    ntime = Emilli-Smilli

    accuracy = round(metrics.accuracy_score(y_test, y_pred)*100,2)
    precision = round(metrics.precision_score(y_test, y_pred, average='weighted')*100,2)
    recall = round(metrics.recall_score(y_test, y_pred, average='weighted')*100,2)
    f1score = round(metrics.f1_score(y_test, y_pred, average='weighted')*100,2)
    
    # Print the Results

    sg.PopupOK('SVM Results : \n accuracy:', str(accuracy) + '%' ,'\n precision:', str(precision) + '%'  ,'\n recall:', str(recall) + '%'  ,'\n F-measure:', str(f1score) + '%'  ,'\n Computation Time:',str(ntime) + ' ms')      
    
    
# Layout the design of the GUI      
layout = [[sg.Text('Please select an algorithm: ', auto_size_text=True)],      
[sg.Button('Naive Bayes'), sg.Button('Random Forest'), sg.Button('SVM'), sg.Quit()]]      

# Show the Window to the user    
window = sg.Window('Comparison System').Layout(layout)      

    

def gui():
    # Read the Window 
    event, value = window.Read()      
    # Take appropriate action based on button      
    if event == 'Naive Bayes':      
        button1()
        gui()
    elif event == 'Random Forest':      
        button2()
        gui()
    elif event == 'SVM':      
        button3()
        gui()     
    elif event =='Quit'  or event is None:   
        window.Close()    
        
gui()

# All done!      











