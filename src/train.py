# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Load the dataset
df = pd.read_csv('..././data/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
# Start an MLflow run
with mlflow.start_run():
    # Define and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    # Output results
    print(f"Training completed with accuracy: {accuracy:.2f}")
'''   
# Define and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
    
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"The Accuracy is  {accuracy}")
 
# Save the model to a file
joblib.dump(model, 'model_diabetes_prediction.joblib')
