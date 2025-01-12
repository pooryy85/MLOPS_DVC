# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('../data/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLflow run
mlflow.set_experiment("diabetes_experiment")  # Set a named experiment
with mlflow.start_run():
    # Define and train the model
    model = LogisticRegression(max_iter=500)
    
    # Assuming X and y are defined
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)  # Fit the model

    y_pred = model.predict(X_test)  # Make predictions
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Log parameters, metrics, and model
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("mse",mse)
    mlflow.sklearn.log_model(model, "model", input_example=X_test[:1])
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Output results
    print(f"Training completed with accuracy: {accuracy:.2f}")   
    print(f"Training completed with mse: {mse:.2f}") 

''' This is without MLFlow Experiment
# Define and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
    
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"The Accuracy is  {accuracy}")
'''

# Save the model to a file
joblib.dump(model, "../model/logi-default.joblib")

