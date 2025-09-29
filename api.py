# api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from train import run_training
import mlflow

app = FastAPI(title="Iris Trainer API")

class TrainRequest(BaseModel):
    n_values: List[int] = Field(default_factory=lambda: [10, 50, 100])
    max_depth: int = 5
    experiment: str = "mlops-course"
    model_name: str = "mlops-course_model"
    use_registry: bool = True

class TrainResponse(BaseModel):
    tracking_uri: str
    results: list


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    results = run_training(
        n_values=req.n_values,
        max_depth=req.max_depth,
        experiment_name=req.experiment,
        model_name=req.model_name,
        use_registry=req.use_registry,
    )
    return {"tracking_uri": mlflow.get_tracking_uri(), "results": results}


@app.get("/")
def main():
    return {"Hello" : "world"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)