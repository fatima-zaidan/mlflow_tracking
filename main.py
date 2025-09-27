import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("mlops-course")

MODEL_NAME = "mlops-course-model"

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different values for n_estimators
n_estimators_list = [10, 50, 100]
max_depth = 5
random_state = 42

for n in [10,50,100]:
    with mlflow.start_run(run_name=f"RF_{n}_estimators",tags={"model":"RandomForest", "dataset":"Iris"}) :
        # Train model:
        clf = RandomForestClassifier(
            n_estimators=n,
            max_depth=max_depth,
            random_state=random_state
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log parameters
        mlflow.log_param("n_estimators", n)
        mlflow.log_param("max_depth", max_depth)
       

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        signature= infer_signature(X_train, clf.predict(X_train))

        # Log model
        info=mlflow.sklearn.log_model(clf, artifact_path="model", signature=signature, registered_model_name=MODEL_NAME)
        
        print(f"logged run-id={info.run_id}|registered_model_name={MODEL_NAME}")
        print(f"Run with n_estimators={n}: "
              f"Accuracy={acc:.4f}, F1-score={f1:.4f}")
        
