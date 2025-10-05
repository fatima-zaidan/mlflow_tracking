import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#
import os
import dagshub



def run_training(
    n_values=(10, 50, 100),
    max_depth=5,
    experiment_name="mlops-demo",
    model_name="mlops-demo-model",
    use_registry=True,
    dagshub_owner=os.getenv("DAGSHUB_OWNER", "fatima-zaidan"),
    dagshub_repo=os.getenv("DAGSHUB_REPO", "mlflow_tracking"),
):
    # If env MLFLOW_TRACKING_URI isn’t set, init DagsHub automatically
    if mlflow.get_tracking_uri().startswith("file:") and dagshub:
        dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)

    mlflow.set_experiment(experiment_name)

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    for n in n_values:
        with mlflow.start_run(run_name=f"rf_n={n}", tags={"model": "rf", "dataset": "iris"}):
            clf = RandomForestClassifier(n_estimators=int(n), max_depth=int(max_depth), random_state=42)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            acc = accuracy_score(y_te, preds)

            mlflow.log_param("n_estimators", int(n))
            mlflow.log_param("max_depth", int(max_depth))
            mlflow.log_metric("accuracy", float(acc))

            signature = infer_signature(X_tr, clf.predict(X_tr))

            run_id = mlflow.active_run().info.run_id
            try:
                mlflow.sklearn.log_model(
                    sk_model=clf,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=model_name if use_registry else None,
                )
                registered = True if use_registry else False
            except Exception as e:
                # If registry is unsupported on remote (e.g., MLflow 3 API), still log the model artifact
                mlflow.sklearn.log_model(sk_model=clf, artifact_path="model", signature=signature)
                registered = False

            results.append({
                "run_id": run_id,
                "n_estimators": int(n),
                "accuracy": float(acc),
                "registered": registered,
            })
    return results


if __name__ == "__main__":
    run_training()