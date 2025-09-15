from sklearn import datasets 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


iris= datasets.load_iris()

x = iris.data
y = iris.target


x_train , x_test  , y_train , y_test = train_test_split( x, y , test_size=0.2 , random_state=42)

model= LogisticRegression()

train_sizes, train_scores, test_scores = learning_curve(model , x_train ,y_train)

model.fit(x_train , y_train)
y_predict = model.predict(x_test)


accuracy = accuracy_score(y_test , y_predict)
cm =  confusion_matrix(y_test , y_predict)
cr = classification_report(y_test , y_predict)

train_scores_mean = np.mean(train_scores , axis=1 )
test_scores_mean = np.mean(test_scores ,  axis=1 )


print('Accuracy :' , accuracy)
print('Confusion Matrix:\n' , cm)
print('classification report:\n', cr)



plt.figure(figsize=(5,4))
sns.heatmap( cm , annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


plt.figure(figsize=(5,4))
plt.plot( train_sizes, train_scores_mean, 'o-' , color='blue' , label='Training score' )
plt.plot(train_sizes, test_scores_mean , 'o-' , color='red', label='Test score' )
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('learning_curve')
plt.legend(loc='best')
plt.show()

#probs = model.predict_proba(x_test)
#print("Probabilities for class 1:\n", probs[:, 1])