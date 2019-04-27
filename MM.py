# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('F:\\College\\New folder\\MovieDatabase.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Total Revenue vs Days (Training set)')
plt.xlabel('Days')
plt.ylabel('Box Office Collection')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Total Revenue vs Days (Test set)')
plt.xlabel('Days')
plt.ylabel('Box Office Collection')
plt.show()



# MODULE 2

# Importing the dataset
dataset = pd.read_csv('F:\\College\\New folder\\MovieDatabase.csv')
A = dataset.iloc[:, :-2].values
b = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_A = StandardScaler()
A_train = sc_A.fit_transform(A_train)
A_test = sc_A.transform(A_test)
sc_b = StandardScaler()
b_train = sc_b.fit_transform(b_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(A_train, b_train)

# Predicting the Test set results
b_pred = regressor.predict(A_test)

# Visualising the Training set results
plt.scatter(A_train, b_train, color = 'red')
plt.plot(A_train, regressor.predict(A_train), color = 'blue')
plt.title('Revenue/Day vs Days (Training set)')
plt.xlabel('Days')
plt.ylabel('Box Office Collection')
plt.show()

# Visualising the Test set results
plt.scatter(A_test, b_test, color = 'red')
plt.plot(A_train, regressor.predict(A_train), color = 'blue')
plt.title('Revenue/Day vs Days (Test set)')
plt.xlabel('Days')
plt.ylabel('Box Office Collection')
plt.show()

#Module 3


# IMPORTING DATASET
dataset = pd.read_csv('F:\\College\\New folder\\MovieDatabase.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 2].values
dataset.plot()
# Splitting Data into Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Random Forest class to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# for i in range(len(y_pred)):
#     print(X_test[i, :], y_pred[i])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.title('Random forest classification (Test set)')
plt.xlabel('Days')
plt.ylabel('Collection')
plt.legend()
plt.show()




#module 4

# IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv('F:\\College\\New folder\\BBB.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
dataset.plot()
# Splitting Data into Training & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Random Forest class to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# for i in range(len(y_pred)):
#     print(X_test[i, :], y_pred[i])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','yellow'))(i), label = j)
plt.title('Random forest classification (Train set)')
plt.xlabel('Days')
plt.ylabel('Collection')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green','yellow')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'brown','yellow','blue','black'))(i), label = j)
plt.title('Random forest classification (Test set)')
plt.xlabel('Days')
plt.ylabel('Collection')
plt.legend()
plt.show()
