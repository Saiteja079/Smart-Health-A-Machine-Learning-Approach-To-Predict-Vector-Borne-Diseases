
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import pickle

# Load your dataset
data = pd.read_csv('data.csv')

# Separate features (X) and target variable (y)
X = data.drop('Diseas', axis=1)  
y = data['Diseas'] 

# Check for missing values 
print(X.isnull().sum())

# Impute missing values 
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (Logistic Regression)
model_lr = LogisticRegression()
model_lr.fit(X_train_scaled, y_train)

# Save the Logistic Regression model as a .pkl file
with open('l_model.pkl', 'wb') as f:
    pickle.dump(model_lr, f)

# Predict on the test data
y_pred_lr = model_lr.predict(X_test_scaled)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr:.2f}')
#print(classification_report(y_test, y_pred_lr))




# Train the model (Random Forest Classifier)
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_scaled, y_train)

with open('lg.pkl', 'wb') as f:
    pickle.dump(model_rf, f)

# Predict on the test data
y_pred_rf = model_rf.predict(X_test_scaled)

# Evaluate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
#print(classification_report(y_test, y_pred_rf))
