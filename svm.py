import numpy as np

from functools import reduce
from sklearn import svm, datasets, linear_model, neighbors, tree, ensemble
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

class SVM:
  @staticmethod
  def get_svm_classifier():
    return svm.SVC()

  @staticmethod
  def get_knn_classifier():
    return neighbors.KNeighborsClassifier()

  @staticmethod
  def get_logistic_classifier():
    return linear_model.LogisticRegression()

  @staticmethod
  def get_decision_tree_classifier():
    return tree.DecisionTreeClassifier()

  @staticmethod
  def get_random_forest_classifier():
    return ensemble.RandomForestClassifier()

  @staticmethod
  def run(inputs, expected_classifiers):
   # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
      inputs,
      expected_classifiers,
      test_size=0.4,
      stratify=expected_classifiers,
      random_state = 0
    )
    parameters_per_kernel = [
      ('svm_linear', { 'kernel': ['linear'], 'C': [0.1] }, SVM.get_svm_classifier),
      ('svm_polynomial', { 'kernel': ['poly'], 'C': [1], 'degree': [6], 'gamma': [0.5] }, SVM.get_svm_classifier),
      ('svm_rbf', { 'kernel': ['rbf'], 'C': [100], 'gamma': [3] }, SVM.get_svm_classifier),
      ('logistic', { 'C': [0.1] }, SVM.get_logistic_classifier),
      ('knn', { 'n_neighbors': [5], 'leaf_size': [8] }, SVM.get_knn_classifier),
      ('decision_tree', { 'max_depth': [12], 'min_samples_split': [2] }, SVM.get_decision_tree_classifier),
      ('random_forest', { 'max_depth': [47], 'min_samples_split': [2] }, SVM.get_random_forest_classifier),
    ]

    results = []
    for label, params, get_classifier_type in parameters_per_kernel:
      # Generate the classifier
      classifier_type = get_classifier_type()
      clf = GridSearchCV(classifier_type, params, cv=5, verbose=0)
      clf.fit(X_train, y_train)

      results.append([
        label,
        max(clf.cv_results_["mean_test_score"]),
        clf.score(X_test, y_test)
      ])
    
    return results
