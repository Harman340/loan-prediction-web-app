import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Sample dataset (you can replace with real CSV)
data = {
    'income': [50000, 60000, 30000, 40000, 80000],
    'age': [25, 35, 22, 28, 45],
    'loan': [20000, 25000, 10000, 15000, 30000],
    'approved': [1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[['income', 'age', 'loan']]
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved!")
