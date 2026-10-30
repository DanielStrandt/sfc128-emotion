from __future__ import annotations
from fastapi.testclient import TestClient

from sfc128_emotion.api import create_app


def test_api_health_info_predict(synthetic_export):
    app = create_app(model_dir=synthetic_export)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200 and response.json()["model_loaded"] is True
        info = client.get("/model-info").json()
        assert info["depth"] == 2 and len(info["labels"]) == 6
        prediction = client.post("/predict", json={"text": "So happy!", "top_k": 3})
        assert prediction.status_code == 200
        assert len(prediction.json()["top_predictions"]) == 3
        batch = client.post("/predict-batch", json={"texts": ["Sad!", "Happy!"]})
        assert batch.status_code == 200 and len(batch.json()) == 2
        assert client.post("/predict", json={"text": "x🙂"}).status_code == 422
        assert client.post("/predict", json={"text": ""}).status_code == 422
