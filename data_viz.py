import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('heart.csv')

sns.countplot(x='target', data=data)
plt.title('Distribution of Heart Disease')
plt.savefig('distribution_of_heart_disease.png')
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

features = ['age', 'trestbps', 'chol', 'thalach']
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.savefig(f'distribution_of_{feature}.png')
    plt.show()
