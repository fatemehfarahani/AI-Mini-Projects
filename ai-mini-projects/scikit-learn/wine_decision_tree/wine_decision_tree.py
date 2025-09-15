from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt


wine = datasets.load_wine()
x = wine.data
y = wine.target

x_train , x_test, y_train , y_test = train_test_split(x, y , test_size=0.2 , random_state=42 )

model = DecisionTreeClassifier()

train_sizes , train_scores , test_scores= learning_curve (model , x_train , y_train)


model.fit (x_train, y_train)

y_predict = model.predict(x_test)

accuracy = accuracy_score (y_test , y_predict)
cm = confusion_matrix (y_test , y_predict)
cr = classification_report (y_test , y_predict)

train_scores_mean = np.mean(train_scores , axis=1)
test_scores_mean = np.mean ( test_scores , axis=1)

print("Accuracy: " , accuracy )
print("confusion_matrix :", cm  )
print("classification_report:", cr )

plt.figure(figsize=(5,4))
sns.heatmap(cm , annot=True , fmt='d', cmap='Reds')
plt.title("confusion_matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(5,4))
plt.plot(train_scores_mean , '*-', color='hotpink', label='Training score')
plt.plot(test_scores_mean , '*-' , color='skyblue', label='Test score')
plt.title('learning_curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()