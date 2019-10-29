from random import randrange
from csv import reader
import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image
import pydotplus

col_names = ['Sample code number', 'Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size', 'Bare Nuclei','Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# load dataset
pandas_dataset = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=col_names)
# pandas_dataset.head()

# remove noise data
for index , row in pandas_dataset.iterrows():
	for col in range(len(row)):
		if row[col] == "?":
			pandas_dataset.drop(index, inplace = True)
			break

feature_cols = ['Clump Thickness','Uniformity of Cell Size', 'Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size', 'Bare Nuclei','Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
X = pandas_dataset[feature_cols] 
y = pandas_dataset.Class 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# clf = DecisionTreeClassifier(criterion='entropy')
clf = DecisionTreeClassifier(criterion='gini')

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, average='micro'))
print("Recall:", metrics.recall_score(y_test, y_pred, average='micro'))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['2','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('winconsin_breast_cancer_gini2.png')
Image(graph.create_png())