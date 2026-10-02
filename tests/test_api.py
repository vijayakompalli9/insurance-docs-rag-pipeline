from fastapi.testclient import TestClient

from rag_pipeline.api import create_app


def test_health_and_query(indexed_settings):
    with TestClient(create_app(indexed_settings)) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["chunks"] > 40
        resp = client.post("/query", json={"question": "Does the policy cover earthquake or landslide?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["sources"][0]["doc_id"] == "ho-perils-exclusions"
        assert body["citations"] and not body["refused"]


def test_query_validation_and_refusal(indexed_settings):
    with TestClient(create_app(indexed_settings)) as client:
        assert client.post("/query", json={"question": ""}).status_code == 422
        body = client.post("/query", json={"question": "Best pizza toppings in Naples?"}).json()
        assert body["refused"] is True


def test_query_without_index_returns_503(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.post("/query", json={"question": "anything here"}).status_code == 503
