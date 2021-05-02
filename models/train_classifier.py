# import libraries
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, recall_score, precision_score, f1_score
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine
import pickle


def load_data(database_filepath):

    """Function to load data from database.

    Args:
        database_filepath: String. the path of the db file.

    Returns:
        X: numpy.ndarray. Feature variable, disaster messages.
        y: numpy.ndarray. target variable, disaster categories for each messages.    

    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("DisasterResponse", con=engine)

    category_names = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[category_names].values

    return X, y, category_names

    
def tokenize(text):

    """Function to Normalize, tokenize, and lemmatize text (message).
    
    Args:
      text: String. Message.
      
    Returns:
      list_tokens: list of strings. A list of clean tokens.
      
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get urls with regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with urlplaceholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    list_tokens = []
    for t in tokens:
        lem_norm = lemmatizer.lemmatize(t).lower().strip()
        list_tokens.append(lem_norm)

    return list_tokens


def build_model():
    
    """Function to build a model.

    Returns:
       cv: gridsearchcv.

    """

    # create a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])

    # parameters
    parameters = {
    'clf__estimator__n_estimators': [10, 20]
    }

    # create GridSearchCV
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """Function to evaluate the model and print scores for each output categories.

    Args:
        model: The Model, that contains a scikit-learn estimador.
        X: numpy.ndarray. Feature variable, disaster messages.
        y: numpy.ndarray. target variable, disaster categories for each messages. 
        category_names: list of strings with category names.
     
    """

    # predict categories of messages
    y_pred = model.predict(X_test)
    
    # print accuracy, precision, recall and f1_score for each categories
    for i,c in enumerate(category_names):
        print(c)
        accuracy = accuracy_score(Y_test[i], y_pred[i])
        precision = precision_score(Y_test[i], y_pred[i])
        recall = recall_score(Y_test[i], y_pred[i])
        f1 = f1_score(Y_test[i], y_pred[i])
        
        print("\tAccuracy: %.2f\tPrecision: %.2f\t Recall: %.2f\t F1 Score: %.2f\n" % (accuracy, precision, recall, f1))



def save_model(model, model_filepath):

    """Function to save the model to a pickle file.

    Args:
        model: The Model, that contains a scikit-learn estimador.
        model_filepath: String. The model is saved into a pickle file here.
        
    """

    with open (model_filepath, 'wb') as file:
        pickle.dump(model, file)

    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()