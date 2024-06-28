import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('heart.csv')

X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)

y_pred = sgd_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SGD Accuracy: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('SGD Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('sgd_confusion_matrix.png')
plt.show()

print(classification_report(y_test, y_pred))
