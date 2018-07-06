import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline
import pandas as pd
import numpy as np
import feature
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


#模型和预测
X_train,Y_train,X_test,train_df,test_df = feature.split_train_test()
#X_train.shape, Y_train.shape, X_test.shape
print(X_train.shape)
print(X_test.shape)
print(X_train.columns.values)

#标准化
x_train = StandardScaler().fit_transform(X_train.values)
x_test = StandardScaler().fit_transform(X_test.values)

'''
# Logistic Regression
#print("------logistic----------")
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
#print(acc_log)


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

#print(coeff_df.sort_values(by='Correlation', ascending=False))

# Support Vector Machines
#print("------svm----------")
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
#print(acc_svc)


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
#print(acc_knn)

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
#print(acc_gaussian)

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
#print(acc_perceptron)

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
#print(acc_linear_svc)

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
#print(acc_sgd)

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#print(acc_decision_tree)

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#print(acc_random_forest)

#model evalution
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
#submission.to_csv('data/submission.csv', index=False)
'''
clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1, bootstrap=False)
kfold = StratifiedKFold(n_splits=10)
param_max_depth = [None]
param_min_samples_split = [2, 3, 10]
param_min_samples_leaf = [1, 3, 10]
param_max_features = [1, 3, 8]
param_n_estimators = [1100]
param_grid = {"max_depth": param_max_depth,
              "max_features": param_max_features,
              "min_samples_split": param_min_samples_split,
              "min_samples_leaf": param_min_samples_leaf,
              "n_estimators": param_n_estimators,
              }
gs = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=kfold, iid=False)
start = time.time()
gs.fit(x_train, Y_train)
end = time.time()
elapsed_train_time = 'Random Forest, elapsed training time: {} min, {} sec '.format(int((end - start) / 60),
                                                                                    int((end - start) % 60))
print(elapsed_train_time)
print('--------------------------------------------')
print(gs.best_estimator_)
print('--------------------------------------------')
print('Random Forest, train best score: {}'.format(gs.best_score_))
print('Random Forest, train best param: {}'.format(gs.best_params_))
random_forest_clf = gs.best_estimator_
pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": random_forest_clf.predict(x_test).astype(int)}).to_csv(
    'submission_rf.csv', header=True, index=False)