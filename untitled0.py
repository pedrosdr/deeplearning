from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

x = pd.read_csv('entradas_breast.csv').to_numpy()
y = pd.read_csv('saidas_breast.csv').to_numpy()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
tree.fit(xtrain, ytrain)
predictions = tree.predict(xtest)

print(accuracy_score(ytest, predictions))
plot_tree(tree)

tree = DecisionTreeClassifier()
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10]
}
gs = GridSearchCV(tree, params, scoring='accuracy', cv=5)
gs.fit(x, y)
best_params = gs.best_params_
best_score = gs.best_score_

tree = gs.best_estimator_
results = cross_val_score(tree, x, y, cv=10, scoring='accuracy')

