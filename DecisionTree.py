from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from UsefulFunctions import plot_decision_regions



#Load the IRIS datasets and prepare model's variables
iris = datasets.load_iris()
X= iris.data[:,[2,3]]
Y = iris.target

#Split our data into train/test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Standerize our features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#Train our decision tree model
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train_std, Y_train)

#Prediction for our test data
Y_predicted = tree.predict(X_test_std)

#Groupe the train and test data
X_full_std = np.vstack((X_train_std, X_test_std))
Y_full = np.hstack((Y_train, Y_test))


print("Decision Tree fails to classify : %d" %(Y_test != Y_predicted).sum())
print('Accuracy: %.2f' %accuracy_score(Y_test, Y_predicted))


plot_decision_regions(X=X_full_std, y=Y_full, classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length std')
plt.ylabel('petal width std')
plt.title('Decision Tree ')
plt.legend(loc='upper left')
plt.show()

