import numpy as np

import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def make_pipeline(model):
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', model)
    ])
    return text_clf


def param_tune(text_clf, parameters, dataset_train):
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(dataset_train.data, dataset_train.target)
    return gs_clf


def model_predict(dataset_train, dataset_test, model, parameters, tune_params=True):
    text_clf = make_pipeline(model)
    if tune_params:
        gs_clf = param_tune(text_clf, parameters, dataset_train)
        clf = gs_clf
    else:
        text_clf.fit(dataset_train.data, dataset_train.target)
        clf = text_clf
    predicted = clf.predict(dataset_test.data)
    return clf, predicted


def main():
    newsgroup_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroup_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }


    logistic_params = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
        #"classifier": [LogisticRegression()],
        #"n_estimators": n_estimators,
        "logisticregression__penalty": ["l1", "l2"],
        "logisticregression__C": np.logspace(-3, 3, 7),
    }
    model_estimator_logistic = LogisticRegression()
    clf_logistic, predicted_logistic = model_predict(newsgroup_train, newsgroup_test, model_estimator_logistic, logistic_params)
    accuracy_logistic = np.mean(predicted_logistic == newsgroup_test.target)
    print(f'Accuracy (logistic regression): {accuracy_logistic}')
    print(metrics.classification_report(newsgroup_test.target, predicted_logistic, target_names=newsgroup_test.target_names))
    print(metrics.confusion_matrix(newsgroup_test.target, predicted_logistic))
    print(clf_logistic.best_score_)
    for param_name in sorted(parameters.keys()):
        print(f"{param_name}: {clf_logistic.best_params_[param_name]}")

    model_estimator_tree = DecisionTreeClassifier()
    clf_tree, predicted_tree = model_predict(newsgroup_train, newsgroup_test, model_estimator_tree, parameters)
    accuracy_tree = np.mean(predicted_tree == newsgroup_test.target)
    print(f'Accuracy (decision tree): {accuracy_tree}')
    print(metrics.classification_report(newsgroup_test.target, predicted_tree, target_names=newsgroup_test.target_names))
    print(metrics.confusion_matrix(newsgroup_test.target, predicted_tree))
    print(clf_tree.best_score_)
    for param_name in sorted(parameters.keys()):
        print(f"{param_name}: {clf_tree.best_params_[param_name]}")

    # Could also use SGDClassifier
    model_estimator_svm = LinearSVC(#loss='hinge',
                                    penalty='l2',
                                    #random_state=42,
                                    dual=False,
                                    #max_iter=8000,
                                    tol=1e-3)
    clf_svm, predicted_svm = model_predict(newsgroup_train, newsgroup_test, model_estimator_svm, parameters)
    accuracy_svm = np.mean(predicted_svm == newsgroup_test.target)
    print(f'Accuracy (support vector machine): {accuracy_svm}')
    print(metrics.classification_report(newsgroup_test.target, predicted_svm, target_names=newsgroup_test.target_names))
    print(metrics.confusion_matrix(newsgroup_test.target, predicted_svm))
    print(clf_svm.best_score_)
    for param_name in sorted(parameters.keys()):
        print(f"{param_name}: {clf_svm.best_params_[param_name]}")
    
    # for doc, cat in zip(newsgroup_test.data, predicted):
    #     print(f'{doc[:50]}... => {newsgroup_train.target_names[cat]}')

    # LEARN_RATE = .1
    # MAX_NUM_ITER = 8000
    # DECAY = .96
    # DECAY_RATE = 50
    # EPS = 1e-2
    # REGUL_LAMBDA = .1
    # k = 5 # of folds for cross validation


if __name__ == "__main__":
    main()
