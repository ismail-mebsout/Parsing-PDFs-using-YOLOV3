"""API tests using FastAPI's TestClient with the pipeline mocked out."""

import pytest
from fastapi.testclient import TestClient

from pdf_table_extractor import api
from pdf_table_extractor.pipeline import ExtractionResult


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_extract_success(client, monkeypatch, fake_tables):
    def fake_extract(pdf_path, page):
        return ExtractionResult(
            pdf_path=pdf_path,
            page=page,
            areas=["1,2,3,4", "5,6,7,8"],
            tables=fake_tables,
        )

    monkeypatch.setattr(api, "extract_tables", fake_extract)

    resp = client.post(
        "/extract",
        params={"page": 2},
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_tables"] == 2
    assert body["page"] == 2
    assert body["tables"][0]["index"] == 0
    assert body["tables"][0]["area"] == "1,2,3,4"
    assert body["tables"][0]["rows"] == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


def test_extract_rejects_non_pdf(client):
    resp = client.post(
        "/extract",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_extract_rejects_empty_file(client):
    resp = client.post(
        "/extract",
        files={"file": ("doc.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400


def test_extract_page_out_of_range(client, monkeypatch):
    def boom(pdf_path, page):
        raise IndexError("page out of range")

    monkeypatch.setattr(api, "extract_tables", boom)

    resp = client.post(
        "/extract",
        params={"page": 99},
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 400
    assert "out of range" in resp.json()["detail"]


def test_extract_invalid_page_number(client):
    # page must be >= 1 (validated by FastAPI).
    resp = client.post(
        "/extract",
        params={"page": 0},
        files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )
    assert resp.status_code == 422
