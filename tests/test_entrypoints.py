"""公开入口的参数和任务状态契约，不访问实际 Milvus。"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import cli
from app.api import admin
from app.config import settings
from app.main import app
from app.models import IngestReport


def test_cli_ingest_passes_supported_recreate_keyword(monkeypatch) -> None:
    from app.ingest import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "run_ingest", lambda **kwargs: calls.append(kwargs) or IngestReport(ok=True))
    args = argparse.Namespace(book="design", dry_run=True, force_reembed=False, recreate=True)

    assert cli.cmd_ingest(args) == 0
    assert calls == [{"book": "design", "dry_run": True, "force_reembed": False, "recreate": True}]


def test_admin_ingest_passes_supported_keyword_and_reports_failure(monkeypatch) -> None:
    from app.ingest import pipeline

    calls = []
    monkeypatch.setattr(settings, "admin_api_key", "test-admin-key")
    monkeypatch.setattr(
        pipeline,
        "run_ingest",
        lambda **kwargs: calls.append(kwargs) or IngestReport(ok=False, warnings=["未写入"]),
    )

    class ImmediateThread:
        def __init__(self, target, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(admin.threading, "Thread", ImmediateThread)
    with TestClient(app) as client:
        response = client.post(
            "/api/admin/ingest",
            json={"book": "deep", "recreate_collection": True},
            headers={"Authorization": "Bearer test-admin-key"},
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        status = client.get(
            f"/api/admin/ingest/{job_id}",
            headers={"Authorization": "Bearer test-admin-key"},
        )
    assert status.json()["status"] == "failed"
    assert calls == [{"book": "deep", "dry_run": False, "force_reembed": False, "recreate": True}]


def test_cli_check_requires_real_milvus_operation(monkeypatch, tmp_path) -> None:
    from app.ingest import milvus_store

    class UnavailableStore:
        def has_collection(self):
            raise ConnectionError("offline")

    dummy = SimpleNamespace(
        data_dir=tmp_path,
        logs_dir=tmp_path,
        pdf_paths={},
        milvus_uri="localhost",
        milvus_collection="agent_design_kb",
        embedding_model="test",
        embedding_dimension=3,
        embedding_api_key="test-key",
        embedding_mock=False,
        embedding_enabled=True,
        llm_base_url="localhost",
        llm_model="test",
        llm_api_key="test-key",
        llm_enabled=True,
    )
    (tmp_path / "chunks.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "settings", dummy)
    monkeypatch.setattr(milvus_store, "get_store", lambda: UnavailableStore())

    assert cli.cmd_check(argparse.Namespace()) == 1
