from pprint import pprint
from time import time
import logging
import csv
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.metrics import f1_score


logging.basicConfig(filename='app.log', level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
X_train, y_train = fetch_20newsgroups(
    subset="train", remove=("headers", "footers", "quotes"), return_X_y=True
)
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words="english")

X_test, y_test = fetch_20newsgroups(
    subset="test", remove=("headers", "footers", "quotes"), return_X_y=True
)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train.shape)
print(X_test.shape)

ch2 = SelectKBest(chi2, k=5000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)

#models = {"RF": 0, "AdaBoost": 1, "linearSVM": 2}
models = {"linearSVM": 2}
estimators = [
    RandomForestClassifier(),
    AdaBoostClassifier(),
    LinearSVC(),
]

# Number of trees in random forest
n_estimators = [20,30,40,60,80,100,120,140,180,200]

# Create the random grid
rf_grid = {
    "n_estimators": n_estimators,
    "max_features": ["auto", "sqrt"],
    "max_depth": [1, 5, 10, 20, 40, 60, 80, 100, 120, 150, 200],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

ada_grid = {
    "n_estimators": n_estimators,
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    "algorithm": ["SAMME", "SAMME.R"],
}


svc_grid = {
    "penalty": ["l1", "l2"],
    "loss": ["hinge", "squared_hinge"],
    "multi_class": ["ovr", "crammer_singer"],
}

grids = [rf_grid, ada_grid, svc_grid]


def parameterTuning(clf, random_grid, x, y):
    csvs = []
    tuning_guy = GridSearchCV(
        estimator=clf,
        param_grid=random_grid,
        pre_dispatch=4,
        scoring="f1_weighted",
        return_train_score=True,
        cv=4,
        verbose=2,
        n_jobs=-1,
    )
    tuning_guy.fit(x, y)
    for i in range(len(tuning_guy.cv_results_["mean_test_score"])):
        csvs.append( [
            tuning_guy.cv_results_["mean_test_score"][i],
            tuning_guy.cv_results_["mean_fit_time"][i],
        ] + [tuning_guy.cv_results_["params"][i][pa] for pa in random_grid.keys()])
    best_clf = tuning_guy.best_estimator_
    y_pred = best_clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    return (f1, tuning_guy.best_score_, tuning_guy.best_params_, csvs)


if __name__ == "__main__":
    for m, k in models.items():
        print(m)
        print(k)
        test_f1, best_train_score, best_params, output_csv = parameterTuning(
            estimators[k], grids[k], X_train, y_train
        )
        print(
            """for modle %s, the test f1 is %s, training score is %s and the
          params is:"""
            % (m, test_f1, best_train_score)
        )
        print(best_params)
        with open("output_%s.csv" % (m), "w") as result_file:
            wr = csv.writer(result_file)
            wr.writerows(output_csv)
