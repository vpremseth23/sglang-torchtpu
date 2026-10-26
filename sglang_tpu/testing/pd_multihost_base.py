"""Class fixture for P/D engines that each span an independent TPU slice."""

import os
import signal
import sys
from pathlib import Path

from sglang.test.test_utils import CustomTestCase
from sglang.test.tpu import RecipeLauncher, wait_for_health
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase
from sglang.test.tpu.utils.pd_singlehost_base import peer_hostname


class PDMultiHostTestCase(TPULauncherTestCase, CustomTestCase):
    """Start one local shard everywhere, and a router on the prefill leader.

    The recipe owns slice-local nnodes/TP settings. TPU_WORKER_HOSTNAMES must
    keep describing that slice, including on the leader running the tests.
    Worker test processes wait until the JobSet tears down their services.
    """

    health_timeout = 7200
    log_path = "/tmp/pd_multihost_server.log"
    base_url = "http://127.0.0.1:8000"

    @classmethod
    def setUpClass(cls):
        cls.role = os.environ.get("CI_PD_ROLE")
        cls.rank = int(os.environ.get("TPU_WORKER_ID", "0"))
        hosts = os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",")
        if cls.role not in ("prefill", "decode") or not all(hosts):
            raise ValueError("Provide CI_PD_ROLE and slice-local TPU_WORKER_HOSTNAMES")
        cls.is_test_leader = cls.role == "prefill" and cls.rank == 0
        cls.prefill_host = hosts[0]
        cls.decode_host = (
            peer_hostname(hosts[0], from_role="prefill", to_role="decode")
            if cls.role == "prefill"
            else hosts[0]
        )
        cls._launcher = cls._start_launcher(
            cls.config_module, cls.get_runtime_envs(), log_path=cls.log_path
        )
        if not cls.is_test_leader:
            return

        if str(cls.repo_root) not in sys.path:
            sys.path.insert(0, str(cls.repo_root))
        from benchmark.tpu.launch_server import router_command, wait_healthy

        cls.prefill_url = f"http://{cls.prefill_host}:30000"
        cls.decode_url = f"http://{cls.decode_host}:30000"

        def check_local_server():
            if cls._launcher.process.poll() is not None:
                raise RuntimeError(
                    "Prefill launcher exited before the P/D pair was ready"
                )

        for url in (cls.prefill_url, cls.decode_url):
            wait_healthy(url, cls.health_timeout, check_local_server)
        cls.check_server_configuration()
        router = RecipeLauncher(
            command=router_command(
                [cls.prefill_url], [cls.decode_url], bootstrap_port=8998, port=8000
            ),
            cwd=str(cls.repo_root),
            start_new_session=True,
            tag=cls.tag + "-router",
        )
        cls.addClassCleanup(router.shutdown)
        router.start()
        wait_for_health(
            cls.base_url,
            cls.health_timeout,
            process=router.process,
            endpoint="/health_generate",
            tag=cls.tag,
        )

    def wait_for_worker_shutdown(self):
        """Accept the JobSet's expected teardown, while surfacing engine crashes."""
        code = self._launcher.wait()
        if code in (-signal.SIGTERM, 128 + signal.SIGTERM):
            return
        if code in (-signal.SIGKILL, 128 + signal.SIGKILL):
            log = Path(self.log_path).read_text(errors="replace")
            if "Connection closed by peer" in log or "Connection reset by peer" in log:
                return
        self.fail(f"{self.role} rank {self.rank} exited before teardown: {code}")
