import pandas as pd
import seaborn as sns
import matplotlib as mt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    
    #Loading the database, setting X and y variables, setting the feature importance model
    df = pd.read_csv('dataR2.csv')

    X = df.drop(['Classification'],axis=1)
    y = df['Classification']
    m = RFECV(RandomForestClassifier(), scoring='accuracy')

    #Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #Training the model and predicting
    m.fit(X_train, y_train)
    m_predictions = m.predict(X_test)
    print(classification_report(y_test,m_predictions))
    print(confusion_matrix(y_test,m_predictions))

