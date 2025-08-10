import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(r'AIML-Project//Project 1//iris.data', names=column_names)

# Separate features and labels
X = data.iloc[:, :-1]  # Features
y = data['species']    # Labels (species names)

# Encode labels once
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Show samples
print("Features sample:")
print(X.head())

print("\nLabels sample:")
print(y.head())

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Predict new flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
pred_index = model.predict(new_flower)[0]
pred_species = le.inverse_transform([pred_index])[0]

print(f"Predicted species: {pred_species}")
