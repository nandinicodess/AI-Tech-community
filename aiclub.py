from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
 
iris = load_iris()
x= iris.data
y= iris.target
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2 , random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test , y_pred , average="weighted")
f1 = f1_score(y_test , y_pred , average = "weighted")
print("Accuracy: " , accuracy)
print("Precision:" , precision)
print("Recall: " , recall)
print ("F1:" , f1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
