"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import contextlib
import ipaddress
import logging
import random
import urllib
import warnings
from http import HTTPStatus
from itertools import chain
from typing import Awaitable, Optional, Tuple

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang_router.mini_lb_policy import (
    EngineOverloaded,
    LoadPoller,
    SessionAffinityPolicy,
)
from sglang_router.router_args import MINI_LB_POLICY_CHOICES, RouterArgs

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_SECS = 10.0

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error

# A server's /health and /health_generate run a one-token generation by default
# and give up after SGLANG_HEALTH_CHECK_TIMEOUT (20 s by default) with a 503;
# wait a little longer so the server's own answer arrives.
HEALTH_CHECK_TIMEOUT_SECS = 30


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class _UpstreamConnectionError(aiohttp.ClientConnectionError):
    """A connection failure recorded in the affinity load table."""


def _response_failed(task: asyncio.Task) -> bool:
    return task.exception() is not None or task.result().status != 200


def _sse_events(body: bytes):
    """Yield the JSON payload of each `data:` line of an SSE body, skipping
    `[DONE]` and non-JSON payloads."""
    for line in body.split(b"\n"):
        line = line.rstrip(b"\r")
        if not line.startswith(b"data:"):
            continue
        try:
            yield orjson.loads(line[len(b"data:") :].strip())
        except orjson.JSONDecodeError:
            continue


def _prefill_stream_failure(body: bytes) -> Optional[str]:
    """Classify a streaming prefill's body, which the engine sends with status
    200 even when the request failed (#1706).

    Returns "error_event" if the body has an `error` event (request rejected
    before dispatch, e.g. longer than the context length), "aborted" if every
    request in it (by `meta_info.id`, using its last non-null finish reason)
    was aborted by the scheduler, and None otherwise. A prefill that finished
    normally ends with finish reason `length`, never `abort`. A partially
    aborted batch is not a failure, so decode still serves its other requests.
    Same rule as `prefill_stream_failure` in the Rust PD router (#1704, PR
    #1705).
    """
    aborted_by_id = {}
    for event in _sse_events(body):
        if not isinstance(event, dict):
            continue
        if "error" in event:
            return "error_event"
        meta_info = event.get("meta_info")
        if not isinstance(meta_info, dict):
            continue
        finish_reason = meta_info.get("finish_reason")
        if finish_reason is None:
            continue
        aborted_by_id[str(meta_info.get("id"))] = (
            isinstance(finish_reason, dict) and finish_reason.get("type") == "abort"
        )
    if aborted_by_id and all(aborted_by_id.values()):
        return "aborted"
    return None


async def _read_first_sse_event(content: aiohttp.StreamReader) -> bytes:
    """Read lines up to and including the blank line that ends the first SSE
    event (or until EOF), and return them unchanged."""
    lines = []
    while True:
        line = await content.readline()
        lines.append(line)
        if not line or (not line.strip() and any(seen.strip() for seen in lines)):
            return b"".join(lines)


def _sse_event_ends_request(event_bytes: bytes) -> bool:
    """True if a decode's first SSE event shows that decode already gave up on
    the request: an `error` event (rejected before dispatch) or a finish reason
    of type `abort`. Decode then never asks prefill for its KV."""
    for event in _sse_events(event_bytes):
        if not isinstance(event, dict):
            continue
        if "error" in event:
            return True
        meta_info = event.get("meta_info")
        finish_reason = (
            meta_info.get("finish_reason") if isinstance(meta_info, dict) else None
        )
        if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
            return True
    return False


async def _post_prefill_and_decode(
    *,
    prefill_request: Awaitable[aiohttp.ClientResponse],
    decode_request: Awaitable[aiohttp.ClientResponse],
) -> Tuple[Optional[aiohttp.ClientResponse], Optional[aiohttp.ClientResponse]]:
    """Send a request's prefill and decode POSTs concurrently.

    Returns both responses once both have answered. A PD request needs both
    halves, so if one side fails (non-200 or transport error) while the other
    is still pending, the pending side can never succeed -- e.g. a prefill
    whose decode rejected the request would otherwise wait until its 300s
    bootstrap timeout, and the client with it (#658). The pending request is
    then cancelled, which closes its connection so that engine aborts the
    request, and ``None`` is returned in its place. A side is therefore
    ``None`` only when the other failed. A transport error of the failed side
    is re-raised, after closing the other side's response if it had arrived.
    """
    prefill_task = asyncio.ensure_future(prefill_request)
    decode_task = asyncio.ensure_future(decode_request)
    tasks = (prefill_task, decode_task)
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            if pending and any(_response_failed(task) for task in done):
                break
    finally:
        # Also runs if this coroutine is cancelled (client disconnect).
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    answered = [task for task in tasks if not task.cancelled()]
    error = next((t.exception() for t in answered if t.exception() is not None), None)
    if error is not None:
        for task in answered:
            if task.exception() is None:
                task.result().close()
        raise error
    return (
        None if prefill_task.cancelled() else prefill_task.result(),
        None if decode_task.cancelled() else decode_task.result(),
    )


class MiniLoadBalancer:
    """Debug P/D router with opt-in prefill affinity and sampled load placement."""

    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.test_external_dp_routing = router_args.test_external_dp_routing
        self.prefill_dp_size = None
        self.decode_dp_size = None
        self.affinity_policy = (
            SessionAffinityPolicy()
            if router_args.mini_lb_policy == "session_affinity"
            else None
        )
        self._load_poller = (
            LoadPoller(
                loads=self.affinity_policy.loads,
                metric_names=self.affinity_policy.metric_names,
                interval_secs=1.0,
                timeout_secs=5.0,
            )
            if self.affinity_policy is not None
            else None
        )

    def _client_timeout(self) -> aiohttp.ClientTimeout:
        affinity = self.affinity_policy is not None
        return aiohttp.ClientTimeout(
            total=self.timeout,
            # The total deadline already bounds affinity reads.
            sock_read=None if affinity else self.timeout,
            connect=_CONNECT_TIMEOUT_SECS if affinity else None,
            sock_connect=_CONNECT_TIMEOUT_SECS if affinity else 300,
        )

    async def _engine_request(self, engine, operation, *args, **kwargs):
        try:
            return await operation(*args, **kwargs)
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as error:
            if self.affinity_policy is None:
                raise
            self.affinity_policy.loads.mark_unreachable(engine)
            raise _UpstreamConnectionError(f"{engine}: {error!r}") from error

    def _validate_router_args(self, router_args: RouterArgs):
        logger.warning(
            "\x1b[33mMiniLB is for debugging only. "
            "Routing policy comes from --mini-lb-policy.\033[0m"
        )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Ignoring --policy, see --mini-lb-policy")
            router_args.policy = "random"
        if router_args.mini_lb_policy not in MINI_LB_POLICY_CHOICES:
            raise ValueError(
                f"Unknown MiniLB policy {router_args.mini_lb_policy!r}, "
                f"expected one of {MINI_LB_POLICY_CHOICES}"
            )

        if not router_args.pd_disaggregation:
            raise ValueError("MiniLB only supports PD disaggregation mode")

        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError(
                "MiniLB requires at least one prefill and one decode server"
            )

    def start(self):
        global lb
        lb = self
        if self.affinity_policy is not None:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                force=True,
            )
        uvicorn.run(app, host=self.host, port=self.port)

    def start_load_polling(self) -> Optional[asyncio.Task]:
        """Poll engine loads only when session affinity is enabled."""
        if self._load_poller is None:
            return None
        return asyncio.create_task(
            self._load_poller.run(lambda: self.prefill_urls + self.decode_urls)
        )

    async def _ensure_dp_sizes(self):
        if self.prefill_dp_size is not None:
            return
        session_args = (
            {"timeout": self._client_timeout()}
            if self.affinity_policy is not None
            else {}
        )
        async with aiohttp.ClientSession(**session_args) as session:
            async with await self._engine_request(
                self.prefill_urls[0], session.get, f"{self.prefill_urls[0]}/server_info"
            ) as resp:
                info = await self._engine_request(self.prefill_urls[0], resp.json)
                prefill_dp_size = len(info.get("internal_states", [1]))
            async with await self._engine_request(
                self.decode_urls[0], session.get, f"{self.decode_urls[0]}/server_info"
            ) as resp:
                info = await self._engine_request(self.decode_urls[0], resp.json)
                decode_dp_size = len(info.get("internal_states", [1]))
        self.prefill_dp_size, self.decode_dp_size = prefill_dp_size, decode_dp_size
        logger.info(
            f"[MiniLB] DP sizes: prefill={self.prefill_dp_size}, decode={self.decode_dp_size}"
        )

    def _fork_dp_requests(self, request):
        p_rank = random.randint(0, self.prefill_dp_size - 1)
        d_rank = random.randint(0, self.decode_dp_size - 1)

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req["routed_dp_rank"] = p_rank
        decode_req["routed_dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank

        return prefill_req, decode_req, d_rank

    def select_pair(self, routing_key=None, room=None):
        """Return a random or affinity-selected (prefill, bootstrap port, decode)."""
        prefill_urls = self.prefill_urls
        decode_urls = self.decode_urls
        assert prefill_urls, "No prefill servers available"
        assert decode_urls, "No decode servers available"
        if self.affinity_policy is None:
            prefill = prefill_urls[random.randint(0, len(prefill_urls) - 1)]
            decode = decode_urls[random.randint(0, len(decode_urls) - 1)]
        else:
            decisions = self.affinity_policy.select(
                prefill_urls=prefill_urls,
                decode_urls=decode_urls,
                routing_key=routing_key,
            )
            for decision in decisions:
                logger.info(
                    "[MiniLB] routing decision %s",
                    decision.to_log_json(room=room, routing_key=routing_key),
                )
            prefill, decode = (decision.engine for decision in decisions)
        pidx = self.prefill_urls.index(prefill)
        return prefill, self.prefill_bootstrap_ports[pidx], decode

    def current_role_and_port(self, worker_url):
        """Return (role, bootstrap_port) of a registered server, or (None, None)
        if it is not in either routing list."""
        if worker_url in self.prefill_urls:
            return (
                "prefill",
                self.prefill_bootstrap_ports[self.prefill_urls.index(worker_url)],
            )
        if worker_url in self.decode_urls:
            return "decode", None
        return None, None

    def remove_worker(self, worker_url):
        """Drop a server from both routing lists so no new requests are sent to
        it (used to quiesce it before a role switch)."""
        if worker_url in self.decode_urls:
            self.decode_urls.remove(worker_url)
        if worker_url in self.prefill_urls:
            idx = self.prefill_urls.index(worker_url)
            self.prefill_urls.pop(idx)
            self.prefill_bootstrap_ports.pop(idx)

    def add_worker(self, worker_url, role, bootstrap_port=None):
        """Register a server under a role in the routing lists."""
        if role == "prefill":
            self.prefill_urls.append(worker_url)
            self.prefill_bootstrap_ports.append(bootstrap_port or 8998)
        elif role == "decode":
            self.decode_urls.append(worker_url)

    def apply_role_switch(self, worker_url, new_role, bootstrap_port=None):
        """Move a server between the prefill and decode routing lists after its
        role has been switched on the backend. Idempotent."""
        self.remove_worker(worker_url)
        self.add_worker(worker_url, new_role, bootstrap_port)

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> Response:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        expected_decode_dp_rank = None
        if self.test_external_dp_routing:
            await self._ensure_dp_sizes()
            prefill_req, decode_req, expected_decode_dp_rank = self._fork_dp_requests(
                modified_request
            )
        else:
            prefill_req = modified_request
            decode_req = modified_request

        async with aiohttp.ClientSession(timeout=self._client_timeout()) as session:
            prefill_response, decode_response = await _post_prefill_and_decode(
                prefill_request=self._engine_request(
                    prefill_server,
                    session.post,
                    f"{prefill_server}/{endpoint}",
                    json=prefill_req,
                ),
                decode_request=self._engine_request(
                    decode_server,
                    session.post,
                    f"{decode_server}/{endpoint}",
                    json=decode_req,
                ),
            )
            # Also covers a cancelled (None) side: the other one failed. Both
            # can be non-None with one failed when it answered last.
            for response, engine in (
                (prefill_response, prefill_server),
                (decode_response, decode_server),
            ):
                if response is not None and response.status != 200:
                    return Response(
                        content=await self._engine_request(engine, response.read),
                        status_code=response.status,
                        media_type=response.content_type,
                    )

            if "return_logprob" in modified_request:
                prefill_json = await self._engine_request(
                    prefill_server, prefill_response.json
                )
                ret_json = await self._engine_request(
                    decode_server, decode_response.json
                )

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await self._engine_request(
                    decode_server, decode_response.json
                )

            if expected_decode_dp_rank is not None:
                actual = ret_json.get("meta_info", {}).get("dp_rank")
                if actual != expected_decode_dp_rank:
                    return ORJSONResponse(
                        content={
                            "error": f"DP rank mismatch: expected {expected_decode_dp_rank}, got {actual}"
                        },
                        status_code=500,
                    )

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):

        if self.test_external_dp_routing:
            warnings.warn("--test-external-dp-routing is not supported with streaming")

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        session = aiohttp.ClientSession(timeout=self._client_timeout())
        responses = []
        # Reads of the two upstream bodies, started once both have answered.
        body_tasks = []

        async def close_upstreams():
            for task in body_tasks:
                task.cancel()
            await asyncio.gather(*body_tasks, return_exceptions=True)
            for response in responses:
                response.close()
            await session.close()

        async def decode_lines():
            # Decode's first event was read while racing prefill's body.
            for line in (await decode_first_event).splitlines(keepends=True):
                yield line
            async for line in decode_response.content:
                yield line

        async def decode_chunks():
            first_event = await decode_first_event
            if first_event:
                yield first_event
            async for chunk in decode_response.content.iter_chunked(
                AIOHTTP_STREAM_READ_CHUNK_SIZE
            ):
                yield chunk

        async def stream_results():
            try:
                # prefill_body is None when decode failed first; relay it as is.
                if (
                    modified_request.get("return_logprob", False)
                    and prefill_body is not None
                ):
                    first_prefill_chunk_json = next(_sse_events(prefill_body))

                    async for chunk in decode_lines():
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"][
                                    "input_token_logprobs"
                                ]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )
                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_chunks():
                        yield chunk
            finally:
                await close_upstreams()

        # The session and both upstream responses stay open only for an
        # accepted request, whose stream_results() closes them when the
        # stream ends; every other exit closes them here.
        streaming = False
        try:
            # Both servers answer the headers before any token is produced, so
            # a request either server rejects is known here, before the
            # streaming response (and its 200 status) is committed. A side
            # still pending when the other fails is cancelled (None).
            prefill_response, decode_response = await _post_prefill_and_decode(
                prefill_request=self._engine_request(
                    prefill_server,
                    session.post,
                    f"{prefill_server}/{endpoint}",
                    json=modified_request,
                ),
                decode_request=self._engine_request(
                    decode_server,
                    session.post,
                    f"{decode_server}/{endpoint}",
                    json=modified_request,
                ),
            )
            answered = [
                (response, engine)
                for response, engine in (
                    (prefill_response, prefill_server),
                    (decode_response, decode_server),
                )
                if response is not None
            ]
            responses.extend(response for response, _ in answered)
            for response, engine in answered:
                if response.status != 200:
                    body = await self._engine_request(engine, response.read)
                    return Response(
                        content=body,
                        status_code=response.status,
                        media_type=response.content_type,
                    )
            # Both bodies report failures in-band with status 200. Race
            # prefill's body (short in PD mode) against decode's first event:
            # - A failed streaming prefill means decode would wait for KV that
            #   never comes (#1706), so prefill's stream is returned instead.
            # - A decode that gave up never asks prefill for its KV, so prefill
            #   never finishes until its bootstrap timeout (#658); decode's
            #   stream is relayed at once and prefill is closed when it ends.
            # Otherwise decode can't produce a token before prefill's KV
            # arrives, so waiting for prefill's body costs no real latency.
            prefill_body_read = asyncio.create_task(
                self._engine_request(prefill_server, prefill_response.read)
            )
            decode_first_event = asyncio.create_task(
                self._engine_request(
                    decode_server, _read_first_sse_event, decode_response.content
                )
            )
            body_tasks.extend([prefill_body_read, decode_first_event])
            await asyncio.wait(body_tasks, return_when=asyncio.FIRST_COMPLETED)
            # Decode finished first only if prefill's body is still pending.
            decode_ended_first = not prefill_body_read.done() and (
                decode_first_event.exception() is not None
                # EOF before any event: decode closed its stream.
                or not decode_first_event.result().strip()
                or _sse_event_ends_request(decode_first_event.result())
            )
            if decode_ended_first:
                logger.warning(
                    "Decode stream failed, relaying it without waiting for "
                    "prefill decode_url=%s prefill_url=%s",
                    decode_server,
                    prefill_server,
                )
                prefill_body = None
                streaming = True
                return StreamingResponse(
                    stream_results(),
                    media_type="text/event-stream",
                )
            prefill_body = await prefill_body_read
            failure = _prefill_stream_failure(prefill_body)
            if failure is not None:
                logger.warning(
                    "Prefill stream failed (%s), aborting paired decode request "
                    "decode_url=%s prefill_url=%s",
                    failure,
                    decode_server,
                    prefill_server,
                )
                # Closing decode (in the finally below) disconnects it, which
                # aborts its request. The client gets prefill's stream
                # unchanged, as it would from a single engine.
                return Response(
                    content=prefill_body,
                    status_code=prefill_response.status,
                    media_type="text/event-stream",
                )
            streaming = True
            return StreamingResponse(
                stream_results(),
                media_type="text/event-stream",
            )
        finally:
            if not streaming:
                await close_upstreams()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    load_polling = lb.start_load_polling() if lb is not None else None
    try:
        yield
    finally:
        if load_polling is not None:
            load_polling.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await load_polling


app = FastAPI(lifespan=_lifespan)
lb: Optional[MiniLoadBalancer] = None


async def _probe_backends(path: str) -> Response:
    """GET `path` on every registered server; 200 only if all answer 200.

    MiniLB routes to every registered server and does not track their health,
    so a single dead server already fails part of the traffic. This is stricter
    than the Rust gateway's /readiness, which skips unhealthy workers.
    """
    timeout = HEALTH_CHECK_TIMEOUT_SECS

    async def probe(session, server):
        try:
            async with session.get(f"{server}{path}") as response:
                return response.status, None
        except asyncio.TimeoutError:
            return None, f"no answer in {timeout} s"
        except aiohttp.ClientError as e:
            return None, f"{type(e).__name__}: {e}"

    roles = {"prefill": list(lb.prefill_urls), "decode": list(lb.decode_urls)}
    backends = [(role, url) for role, urls in roles.items() for url in urls]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        results = await asyncio.gather(*(probe(session, url) for _, url in backends))
    unhealthy = [
        {"role": role, "url": url, "status": status, "error": error}
        for (role, url), (status, error) in zip(backends, results)
        if status != 200
    ]
    # A role switch can briefly leave a role with no registered server.
    unhealthy += [
        {"role": role, "url": None, "status": None, "error": "no server registered"}
        for role, urls in roles.items()
        if not urls
    ]
    if not unhealthy:
        return Response(status_code=200)
    logger.warning(f"[MiniLB] {path}: unhealthy backends {unhealthy}")
    return ORJSONResponse(
        content={"unhealthy": unhealthy},
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


@app.get("/liveness")
async def liveness():
    """200 while the MiniLB process serves requests; ignores the backends."""
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    return await _probe_backends("/health")


@app.get("/health_generate")
async def health_generate():
    return await _probe_backends("/health_generate")


async def _post_role_switch(worker_url, body):
    """POST the role switch to a backend server; return (status, json)."""
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=lb.timeout)
        ) as session:
            async with session.post(f"{worker_url}/pd_role_switch", json=body) as resp:
                return resp.status, await resp.json()
    except Exception as e:  # transport error -> report as a failure
        return 502, {"success": False, "message": str(e)}


@app.post("/pd_role_switch")
async def pd_role_switch(request_data: dict):
    """Switch a running server's PD role (prefill<->decode) at runtime and
    update the LB's routing lists. Body: {"worker_url", "new_role":
    "prefill"|"decode", "bootstrap_port"?, "decode_cuda_graph_bs"?,
    "decode_cuda_graph_memory_gb"?, "drain"?, "drain_timeout_secs"?}.

    The backend rejects a switch unless the instance is idle. To make this
    safe while serving, by default the LB first removes the server from its
    routing lists (so no new requests arrive), then retries the switch while
    the server drains its in-flight requests, and only then registers it
    under the new role. A failed server is restored only when the backend
    confirms that no role state changed."""
    worker_url = request_data.get("worker_url")
    new_role = request_data.get("new_role")
    if worker_url is None:
        raise HTTPException(status_code=400, detail="worker_url is required")
    if new_role not in ("prefill", "decode"):
        raise HTTPException(status_code=400, detail=f"invalid new_role={new_role!r}")

    drain = request_data.get("drain", True)
    drain_timeout = request_data.get("drain_timeout_secs", 300)
    old_role, old_port = lb.current_role_and_port(worker_url)

    body = {"new_role": new_role}
    for field in ("decode_cuda_graph_bs", "decode_cuda_graph_memory_gb"):
        if request_data.get(field) is not None:
            body[field] = request_data[field]

    # Stop routing new requests to this server so it can drain to idle.
    if drain and old_role is not None:
        lb.remove_worker(worker_url)

    deadline = asyncio.get_event_loop().time() + drain_timeout
    while True:
        status, result = await _post_role_switch(worker_url, body)
        if status == 200 and result.get("success", False):
            break
        # The backend rejects while not idle; keep retrying as it drains.
        not_idle = "not idle" in (result.get("message", "") or "").lower()
        if drain and not_idle and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(1.0)
            continue
        if drain and old_role is not None and result.get("safe_to_restore", False):
            lb.add_worker(worker_url, old_role, old_port)
        return ORJSONResponse(content=result, status_code=status)

    lb.apply_role_switch(worker_url, new_role, request_data.get("bootstrap_port"))
    return ORJSONResponse(content=result, status_code=200)


@app.post("/flush_cache")
async def flush_cache(timeout: Optional[float] = None):
    # `timeout` must reach the workers. The scheduler treats a missing or
    # non-positive timeout as "flush now, skip the idle check", so dropping it
    # here frees KV buffers while a PD KV transfer is still reading them: the
    # transfer then fails for real and the peer session gets blacklisted.
    # Forwarding it keeps the scheduler on its deferred, drain-first path.
    params = None if timeout is None else {"timeout": timeout}
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.post(f"{server}/flush_cache", params=params))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


# TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
@app.get("/server_info")
@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
            server_info = await session.get(f"{server}/server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


async def _get_model_info_impl():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


@app.get("/model_info")
async def model_info():
    return await _get_model_info_impl()


@app.get("/get_model_info")
async def get_model_info():
    return await _get_model_info_impl()


@app.post("/generate")
async def handle_generate_request(
    request_data: dict, x_routing_key: Optional[str] = Header(default=None)
):
    return await _forward_to_backend(request_data, "generate", x_routing_key)


async def _forward_to_backend(
    request_data: dict, endpoint_name: str, routing_key: Optional[str]
):
    batch_size = _get_request_batch_size(request_data)
    bootstrap_rooms = [
        _generate_bootstrap_room()
        for _ in range(batch_size if batch_size is not None else 1)
    ]
    try:
        prefill_server, bootstrap_port, decode_server = lb.select_pair(
            routing_key=routing_key,
            room=bootstrap_rooms[0] if bootstrap_rooms else None,
        )
    except EngineOverloaded as error:
        logger.warning(f"[MiniLB] POST /{endpoint_name} failed: {error}")
        return ORJSONResponse(
            content={"error": {"message": str(error)}},
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
        )

    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": (
                [hostname] * batch_size if batch_size is not None else hostname
            ),
            "bootstrap_port": (
                [bootstrap_port] * batch_size
                if batch_size is not None
                else bootstrap_port
            ),
            "bootstrap_room": (
                bootstrap_rooms if batch_size is not None else bootstrap_rooms[0]
            ),
        }
    )
    forward = lb.generate_stream if request_data.get("stream", False) else lb.generate
    try:
        return await forward(
            modified_request, prefill_server, decode_server, endpoint=endpoint_name
        )
    except _UpstreamConnectionError as error:
        logger.warning("[MiniLB] upstream unreachable: %s", error)
        return ORJSONResponse(
            content={"error": {"message": f"upstream unreachable: {error}"}},
            status_code=HTTPStatus.BAD_GATEWAY,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(
    request_data: dict, x_routing_key: Optional[str] = Header(default=None)
):
    return await _forward_to_backend(request_data, "v1/chat/completions", x_routing_key)


@app.post("/v1/completions")
async def handle_completion_request(
    request_data: dict, x_routing_key: Optional[str] = Header(default=None)
):
    return await _forward_to_backend(request_data, "v1/completions", x_routing_key)


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
