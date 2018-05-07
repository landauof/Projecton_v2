# imports
import pandas

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import model_selection

from sklearn.utils import resample
import numpy as np

# vars
selected_features_path =\
    'MFDCA-DATA/FraudedFeatureSelectedOutputs/output'
selected_features_path_other = 'MFDCA-DATA/FraudedFeatureSelectedOutputs/outputALL.csv'
user_number = 5
number_of_features = 225
# Magic

def one_class_svm(n, g):
    data_set = pandas.read_csv(selected_features_path+str(user_number)+'.csv')
    data_set.pop(data_set.columns[0])

    # class distribution
    print(data_set.groupby('Class').size())

    # Split-out validation dataset
    array = data_set.values
    X = array[:, 0:number_of_features]
    Y = array[:, number_of_features]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = \
        X[0:50], X[50:], Y[0:50], Y[50:]
    #    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('OneClassSVM', OneClassSVM()))
    # evaluate each model in turn
    results = []
    names = []
    #for name, model in models:
    #    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #    results.append(cv_results)
    #    names.append(name)
    #    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #    print(msg)

    model = OneClassSVM(nu=n, kernel='rbf', gamma=g)
    model.fit(X_train)
#    print('\n')
#    print(model)
#    print('\n')

    preds = model.predict(X_validation)
    correct_preds = []
    for pred in preds:
        if pred == -1:
            correct_preds.append(1)
        else:
            correct_preds.append(0)
    targs = Y_validation
    print('\n')

    correct_targs = []
    for targ in targs:
        correct_targs.append(targ)
#    print(correct_targs)
#    print(correct_preds)

    print("accuracy: ", metrics.accuracy_score(correct_targs, correct_preds))
#    print("precision: ", metrics.precision_score(correct_targs, correct_preds, average=None))
#    print("recall: ", metrics.recall_score(correct_targs, correct_preds, average=None))
#    print("f1: ", metrics.f1_score(correct_targs, correct_preds, average=None))
    # print("area under curve (auc): ", metrics.roc_auc_score(correct_targs, preds))

    res = metrics.accuracy_score(correct_targs, correct_preds)
#    print(type(np.float64(res).item()))
    fres = np.float64(res).item()
    return fres


def classify():
    data_set_user = pandas.read_csv(selected_features_path + str(user_number) + '.csv')
    data_set_rest = pandas.read_csv(selected_features_path_other)
    data_set_user.pop(data_set_user.columns[0])
    data_set_rest.pop(data_set_rest.columns[0])

    array = data_set_user.values
    X = array[:, 0:number_of_features]
    Y = array[:, number_of_features]

    X_validation, Y_validation = X[50:], Y[50:]

    data_set_train = data_set_user.head(50)
    l = [data_set_train, data_set_rest]
    data_set_train = pandas.concat(l, axis=0)
    data_set_train.to_csv(selected_features_path_other)
    data_set_train = pandas.read_csv(selected_features_path_other)
    data_set_train.pop(data_set_train.columns[0])
    # Separate majority and minority classes
    df_majority = data_set_train.loc[data_set_train["Class"] == 1]
    df_minority = data_set_train.loc[data_set_train["Class"] == 0]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=1950,  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pandas.concat([df_majority, df_minority_upsampled])

    # class distribution#
    print(df_upsampled.groupby('Class').size())

    array = df_upsampled.values
    X_train = array[:, 0:number_of_features]
    Y_train = array[:, number_of_features]

#    one = 0
#    zero = 1
#    for i in Y:
#        if i == 1:
#            one += 1
#        elif i == 0:
#            zero += 1
#        else:
#            print("bug! {}".format(i))
#    print(one)
#    print(zero)
    # if features are in the same order in each CSV, now the arrays should be good
    validation_size = 100/3900
    seed = 7

    for i in Y_validation:
        print(i)

    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # models.append(('OneClassSVM', OneClassSVM()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed )
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    knn = SVC()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(predictions)
    print(Y_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))