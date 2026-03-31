import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv('../data/dataset.csv')

X = data[['rainfall', 'water_level', 'humidity']]
y = data['flood_risk']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'flood_model.pkl')

print("Model trained and saved.")
