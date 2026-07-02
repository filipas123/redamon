"""Integration tests for the /ai-attack-surface/* API route layer (Step 3).

The route coroutines are awaited directly against a faked container_manager
(no httpx/TestClient in the image). Verifies run_config assembly, delegation,
the 503 guards (uninitialized / source not mounted).
"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

import api
from models import (
    AiAttackSurfaceListResponse,
    AiAttackSurfaceState,
    AiAttackSurfaceStartRequest,
    AiAttackSurfaceStatus,
)


def run(coro):
    return asyncio.run(coro)


class TestAiAttackRoutes(unittest.TestCase):
    def setUp(self):
        self._saved_mgr = api.container_manager
        self._saved_path = api.AI_ATTACK_SURFACE_PATH
        api.AI_ATTACK_SURFACE_PATH = "/host/ai_attack_surface_scan"
        self.mgr = MagicMock()
        self.mgr.start_ai_attack_surface = AsyncMock(
            return_value=AiAttackSurfaceState(project_id="p", run_id="r1",
                                              status=AiAttackSurfaceStatus.RUNNING))
        self.mgr.stop_ai_attack_surface = AsyncMock(
            return_value=AiAttackSurfaceState(project_id="p", run_id="r1",
                                              status=AiAttackSurfaceStatus.IDLE))
        self.mgr.get_ai_attack_surface_status = AsyncMock(
            return_value=AiAttackSurfaceState(project_id="p", run_id="r1",
                                              status=AiAttackSurfaceStatus.RUNNING))
        self.mgr.get_all_ai_attack_surface_statuses = AsyncMock(return_value=[])
        api.container_manager = self.mgr

    def tearDown(self):
        api.container_manager = self._saved_mgr
        api.AI_ATTACK_SURFACE_PATH = self._saved_path

    def _req(self, **kw):
        base = dict(project_id="p", user_id="u", tool="skeleton",
                    targets=[{"baseurl": "http://h", "path": "/c"}],
                    bounds={"judge_model": "m"}, roe_confirmed=True)
        base.update(kw)
        return AiAttackSurfaceStartRequest(**base)

    def _http(self, grader_key=None):
        """A fake FastAPI Request exposing only the headers the route reads."""
        req = MagicMock()
        req.headers = {"x-grader-key": grader_key} if grader_key else {}
        return req

    def test_start_builds_run_config_and_delegates(self):
        out = run(api.start_ai_attack_surface("p", self._req(), self._http()))
        self.assertEqual(out.status, AiAttackSurfaceStatus.RUNNING)
        kwargs = self.mgr.start_ai_attack_surface.call_args.kwargs
        rc = kwargs["run_config"]
        self.assertEqual(rc["tool"], "skeleton")
        self.assertTrue(rc["roe_confirmed"])
        self.assertEqual(rc["bounds"], {"judge_model": "m"})
        self.assertEqual(len(rc["targets"]), 1)
        self.assertEqual(kwargs["ai_attack_path"], "/host/ai_attack_surface_scan")
        # local grader default -> no key forwarded
        self.assertIsNone(kwargs["grader_api_key"])

    def test_start_503_when_uninitialized(self):
        api.container_manager = None
        with self.assertRaises(HTTPException) as ctx:
            run(api.start_ai_attack_surface("p", self._req(), self._http()))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_start_503_when_source_not_mounted(self):
        api.AI_ATTACK_SURFACE_PATH = ""
        with self.assertRaises(HTTPException) as ctx:
            run(api.start_ai_attack_surface("p", self._req(), self._http()))
        self.assertEqual(ctx.exception.status_code, 503)

    def test_start_value_error_becomes_409(self):
        self.mgr.start_ai_attack_surface = AsyncMock(side_effect=ValueError("limit"))
        with self.assertRaises(HTTPException) as ctx:
            run(api.start_ai_attack_surface("p", self._req(), self._http()))
        self.assertEqual(ctx.exception.status_code, 409)

    # --- external grader gate (grader fidelity) ---------------------------

    def test_external_grader_without_consent_is_400(self):
        req = self._req(bounds={"judge_model": "m", "grader_provider": "openai"},
                        roe_confirmed=True)
        with self.assertRaises(HTTPException) as ctx:
            run(api.start_ai_attack_surface("p", req, self._http("k")))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_external_grader_without_roe_is_400(self):
        req = self._req(bounds={"judge_model": "m", "grader_provider": "openai",
                                "grader_consent": True}, roe_confirmed=False)
        with self.assertRaises(HTTPException) as ctx:
            run(api.start_ai_attack_surface("p", req, self._http("k")))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_external_grader_without_key_header_is_400(self):
        req = self._req(bounds={"judge_model": "m", "grader_provider": "openai",
                                "grader_consent": True}, roe_confirmed=True)
        with self.assertRaises(HTTPException) as ctx:
            run(api.start_ai_attack_surface("p", req, self._http(None)))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_external_grader_forwards_key_env_only_not_in_run_config(self):
        req = self._req(bounds={"judge_model": "m", "grader_provider": "openai",
                                "grader_model": "gpt-4o", "grader_consent": True},
                        roe_confirmed=True)
        run(api.start_ai_attack_surface("p", req, self._http("sk-grader")))
        kwargs = self.mgr.start_ai_attack_surface.call_args.kwargs
        # key forwarded to the container manager, NOT persisted in run_config
        self.assertEqual(kwargs["grader_api_key"], "sk-grader")
        rc = kwargs["run_config"]
        self.assertNotIn("grader_api_key", rc)
        self.assertNotIn("sk-grader", json.dumps(rc))   # secret never serialized
        # non-secret grader routing stamped into run_config for the scan container
        self.assertEqual(rc["grader_provider"], "openai")
        self.assertEqual(rc["grader_model"], "gpt-4o")

    def test_status_delegates(self):
        out = run(api.get_ai_attack_surface_status("p", "r1"))
        self.assertEqual(out.run_id, "r1")
        self.mgr.get_ai_attack_surface_status.assert_awaited_once_with("p", "r1")

    def test_stop_delegates(self):
        out = run(api.stop_ai_attack_surface("p", "r1"))
        self.assertEqual(out.status, AiAttackSurfaceStatus.IDLE)

    def test_list_returns_response(self):
        out = run(api.list_ai_attack_surface("p"))
        self.assertIsInstance(out, AiAttackSurfaceListResponse)

    def test_logs_404_when_idle(self):
        self.mgr.get_ai_attack_surface_status = AsyncMock(
            return_value=AiAttackSurfaceState(project_id="p", run_id="r1",
                                              status=AiAttackSurfaceStatus.IDLE))
        with self.assertRaises(HTTPException) as ctx:
            run(api.stream_ai_attack_surface_logs("p", "r1"))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
