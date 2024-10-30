import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cloud_classifier(X_data,Y_data,class_type = 'log_res'):

    #Split into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data,test_size = 0.25, random_state = 0)

    #Normalize data (this step seem quite strange)
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)

    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)

    #Train classifier
    logistic_classifier = LogisticRegression(random_state = 0)
    logistic_classifier.fit(X_train,Y_train)

    #Test classifier
    predictions = logistic_classifier.predict(X_test)
    cm = confusion_matrix(Y_test, predictions)
    TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)
    accuracy =  (TP + TN) / (TP + FP + TN + FN)
    print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))
    score = logistic_classifier.score(X_test,Y_test)
    print(f'Accuracy score: {score}')


    # fig_test, ax_test = plt.subplots()
    # sc_test = ax_test.scatter(rise_time,peak_to_zero,c = cloud_indicator,cmap = 'inferno')
    # cb_test = fig_test.colorbar(sc_test)

    return 0