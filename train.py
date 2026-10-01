import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Create/select MLflow experiment
mlflow.set_experiment("MLflow_Assignment")


# 2. Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Start an MLflow run
with mlflow.start_run():

    # 5. Define model
    model = LogisticRegression(
        max_iter=500
    )

    # 6. Train model
    model.fit(X_train, y_train)

    # 7. Make predictions
    predictions = model.predict(X_test)

    # 8. Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # 9. Log parameters
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.2)

    # 10. Log metric
    mlflow.log_metric("accuracy", accuracy)

    # 11. Save model to MLflow
    mlflow.sklearn.log_model(
        model,
        "iris_model"
    )

    print("Model trained successfully!")
    print("Accuracy:", accuracy)
    print("Run ID:", mlflow.active_run().info.run_id)