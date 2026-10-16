# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the production prefix-cache fixture against the supplied KVConductor.

Usage: python tests/core/kv_cache/events/test_motor_kv_events.py
    --fixture /path/to/kv_cache_event_fixture --conductor /path/to/kv-conductor
Requires pyzmq and msgpack. Does not allocate NPU tensors or load a model.
"""

import argparse
import concurrent.futures
import contextlib
import json
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TextIO

import msgpack
import zmq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger


def _rank_endpoint(endpoint: str, rank: int) -> str:
    address, _, port = endpoint.rpartition(":")
    return f"{address}:{int(port) + rank}"


def _post_json(base_url: str, route: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        base_url.rstrip("/") + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _register(args: argparse.Namespace) -> None:
    for rank in range(args.dp_size):
        body: dict[str, Any] = {
            "instance_id": args.instance_id,
            "type": "vllm",
            "modelname": args.model,
            "block_size": args.block_size,
            "dp_rank": rank,
            "tenant_id": args.tenant_id,
        }
        if args.unregister:
            _post_json(args.conductor, "/unregister", body)
            continue
        # Bind each local Host pool to its exact DP, rather than sharing it
        # across every DP on the node as a CPU-only pool registration would.
        body["store_backend"] = "xllm"
        body["medium_endpoints"] = {"npu": _rank_endpoint(args.engine_endpoint, rank)}
        if args.pool_endpoint:
            body["medium_endpoints"]["cpu"] = _rank_endpoint(args.pool_endpoint, rank)
        if args.replay_endpoint:
            body["replay_endpoint"] = _rank_endpoint(args.replay_endpoint, rank)
        _post_json(args.conductor, "/register", body)


def _free_ports(count: int, excluded: set[int]) -> int:
    for _ in range(100):
        sockets = []
        try:
            first = socket.socket()
            sockets.append(first)
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port + count > 65535 or any(p in excluded for p in range(port, port + count)):
                continue
            for offset in range(1, count):
                extra = socket.socket()
                sockets.append(extra)
                extra.bind(("127.0.0.1", port + offset))
            excluded.update(range(port, port + count))
            return port
        except OSError:
            continue
        finally:
            for bound in sockets:
                bound.close()
    raise RuntimeError("Could not find free adjacent ports")


def _read_line(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 10)
    if not ready:
        raise TimeoutError("Fixture did not respond within 10 seconds")
    line = process.stdout.readline().strip()
    if not line:
        raise RuntimeError(f"Fixture exited: {process.poll()}")
    return line


def _command(process: subprocess.Popen[str], **body: Any) -> str:
    assert process.stdin is not None
    process.stdin.write(json.dumps(body) + "\n")
    process.stdin.flush()
    return _read_line(process)


def _start_conductor(binary: str, port: int, log: TextIO) -> subprocess.Popen[str]:
    process = subprocess.Popen([binary, "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=log, text=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"KVConductor exited: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return process
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise TimeoutError("KVConductor did not become ready")


def _terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _verify_lifecycle(fixture_binary: str, context: zmq.Context, ports: set[int], log: TextIO) -> None:
    for operation in ("disable_events", "stop"):
        engine_port = _free_ports(2, ports)
        pool_port = _free_ports(2, ports)
        config = {
            "endpoint": f"tcp://127.0.0.1:{engine_port}",
            "motor_pool_endpoint": f"tcp://127.0.0.1:{pool_port}",
            "snapshot_interval_seconds": 60,
        }
        fixture = subprocess.Popen(
            [fixture_binary, json.dumps(config)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        try:
            assert _read_line(fixture) == "ready"
            for rank in range(2):
                tokens = [100 + rank, 200 + rank]
                assert _command(fixture, op="store", rank=rank, tokens=tokens) == "ok"
                assert _command(fixture, op="copy", rank=rank, tokens=tokens) == "ok"
            with contextlib.ExitStack() as stack:
                subscribers = []
                for rank in range(2):
                    for base_port, source in ((engine_port, "engine"), (pool_port, "pool")):
                        subscriber = stack.enter_context(context.socket(zmq.SUB))
                        subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                        subscriber.setsockopt(zmq.LINGER, 0)
                        subscriber.connect(f"tcp://127.0.0.1:{base_port + rank}")
                        subscribers.append((subscriber, source))
                for subscriber, source in subscribers:
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        if not subscriber.poll(100):
                            continue
                        payload = msgpack.unpackb(subscriber.recv_multipart()[2], raw=False)
                        events = payload[1] if source == "engine" else payload["events"]
                        key = "type" if source == "engine" else "event_type"
                        stored = "BlockStored" if source == "engine" else "stored"
                        if any(event[key] == stored for event in events):
                            break
                    else:
                        raise AssertionError(f"{operation}: no initial {source} residency")
                # Drain join snapshots before checking the lifecycle clear.
                for subscriber, _ in subscribers:
                    while subscriber.poll(100):
                        subscriber.recv_multipart()
                if operation == "stop":
                    started = time.monotonic()
                    assert _command(fixture, op="stop") == "stopped"
                    assert time.monotonic() - started < 2
                else:
                    assert _command(fixture, op=operation) == "ok"
                    assert _command(fixture, op="store", rank=0, tokens=[300, 301]) == "ok"
                    assert not json.loads(_command(fixture, op="status"))["journal_enabled"]
                for subscriber, source in subscribers:
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        if not subscriber.poll(100):
                            continue
                        payload = msgpack.unpackb(subscriber.recv_multipart()[2], raw=False)
                        events = payload[1] if source == "engine" else payload["events"]
                        key = "type" if source == "engine" else "event_type"
                        cleared = "AllBlocksCleared" if source == "engine" else "cleared"
                        if any(event[key] == cleared for event in events):
                            assert len(events) == 1, f"{operation}: clear contains stale residency"
                            break
                    else:
                        raise AssertionError(f"{operation}: no {source} clear")
                if operation == "stop":
                    # Cache destructors have their own grace periods after
                    # the publisher has stopped and flushed its clear events.
                    assert fixture.wait(timeout=10) == 0
        finally:
            _terminate(fixture)


def _verify_non_caching_draft(fixture_binary: str, ports: set[int], log: TextIO) -> None:
    port = _free_ports(2, ports)
    fixture = subprocess.Popen(
        [fixture_binary, json.dumps({"endpoint": f"tcp://127.0.0.1:{port}"}), "--pool"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
    )
    try:
        assert _read_line(fixture) == "ready"
        draft = json.loads(_command(fixture, op="create_draft", prefix_cache=False))
        assert not draft["draft_journal_attached"] and not draft["draft_publisher_healthy"]
        assert _command(fixture, op="store", rank=0, tokens=[800, 801]) == "ok"
        state = json.loads(_command(fixture, op="state"))
        assert len(state["events"]) == 1 and state["events"][0]["model_id"] == "fixture"
        status = json.loads(_command(fixture, op="status"))
        assert status["publisher_healthy"] and status["journal_enabled"]
    finally:
        _terminate(fixture)


def _flood_subscriptions(
    context: zmq.Context,
    endpoints: list[str],
    ready: threading.Event,
    stopped: threading.Event,
    filters: tuple[bytes, ...] = (b"noise",),
) -> list[int]:
    # Each socket belongs to this worker. XSUB sends commands directly, avoiding
    # SUB's local option deduplication. Both engine and pool receive sustained input.
    with contextlib.ExitStack() as stack:
        senders = [stack.enter_context(context.socket(zmq.XSUB)) for _ in endpoints]
        counts = [0] * len(senders)
        monitors = []
        for sender, endpoint in zip(senders, endpoints):
            sender.setsockopt(zmq.LINGER, 0)
            sender.setsockopt(zmq.SNDHWM, 1000)
            monitors.append(stack.enter_context(sender.get_monitor_socket(events=zmq.EVENT_HANDSHAKE_SUCCEEDED)))
            sender.connect(endpoint)
        for sender, monitor in zip(senders, monitors):
            assert monitor.poll(3000), "subscription sender did not complete its handshake"
            monitor.recv_multipart()
            sender.disable_monitor()
        deadline = time.monotonic() + 20
        while not stopped.is_set() and time.monotonic() < deadline:
            for index, sender in enumerate(senders):
                subscription_filter = filters[(counts[index] // 2) % len(filters)]
                command = bytes([1 if counts[index] % 2 == 0 else 0]) + subscription_filter
                try:
                    sender.send(command, zmq.DONTWAIT)
                    counts[index] += 1
                except zmq.Again:
                    stopped.wait(0.001)
            if min(counts) >= 512:
                ready.set()
            # XSUB can drop commands before connection or at HWM. Exercise a
            # sustained connected sender without saturating its local pipe.
            stopped.wait(0.001)
        return counts


def _collect_snapshots(subscribers: list[tuple[zmq.Socket, str]], seconds: float) -> list[int]:
    poller = zmq.Poller()
    for subscriber, _ in subscribers:
        poller.register(subscriber, zmq.POLLIN)
    counts = [0] * len(subscribers)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        readable = dict(poller.poll(50))
        for index, (subscriber, source) in enumerate(subscribers):
            if subscriber not in readable:
                continue
            payload = msgpack.unpackb(subscriber.recv_multipart()[2], raw=False)
            events = payload[1] if source == "engine" else payload["events"]
            key, cleared = ("type", "AllBlocksCleared") if source == "engine" else ("event_type", "cleared")
            counts[index] += any(event[key] == cleared for event in events)
    return counts


def _verify_subscription_snapshots(
    fixture_binary: str, context: zmq.Context, ports: set[int], log: TextIO
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, topic in (("empty", ""), ("binary_long", "kv/缓存\x00" + "x" * 300)):
        engine_port = _free_ports(2, ports)
        pool_port = _free_ports(2, ports)
        endpoints = [
            (f"tcp://127.0.0.1:{port + rank}", source)
            for rank in range(2)
            for port, source in ((engine_port, "engine"), (pool_port, "pool"))
        ]
        config = {
            "endpoint": f"tcp://127.0.0.1:{engine_port}",
            "motor_pool_endpoint": f"tcp://127.0.0.1:{pool_port}",
            "topic": topic,
            "snapshot_interval_seconds": 60,
        }
        fixture = subprocess.Popen(
            [fixture_binary, json.dumps(config)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        try:
            assert _read_line(fixture) == "ready"
            for rank in range(2):
                tokens = [970 + rank, 980 + rank]
                assert _command(fixture, op="store", rank=rank, tokens=tokens) == "ok"
                assert _command(fixture, op="copy", rank=rank, tokens=tokens) == "ok"
            topic_bytes = topic.encode()
            # Cover empty, short prefix, exact and >256-byte matching filters.
            matching = (b"", topic_bytes[:8], topic_bytes, topic_bytes[:280])
            with contextlib.ExitStack() as stack:
                subscribers = []
                for (endpoint, source), subscription_filter in zip(endpoints, matching):
                    subscriber = stack.enter_context(context.socket(zmq.SUB))
                    subscriber.setsockopt(zmq.SUBSCRIBE, subscription_filter)
                    subscriber.setsockopt(zmq.LINGER, 0)
                    subscriber.connect(endpoint)
                    subscribers.append((subscriber, source))
                initial = _collect_snapshots(subscribers, 2.2)
                assert all(count > 0 for count in initial), f"{name}: matching subscriber did not recover"
                stages: dict[str, Any] = {}
                for stage, filters in (
                    ("unmatched", (topic_bytes + b"suffix", topic_bytes[:-1] + b"y")),
                    ("matched", matching),
                ):
                    ready = threading.Event()
                    stopped = threading.Event()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        flood = executor.submit(
                            _flood_subscriptions, context, [ep for ep, _ in endpoints], ready, stopped, filters
                        )
                        try:
                            assert ready.wait(3), f"{name}/{stage}: subscription load did not start"
                            # Commands sent while waiting for ready can already
                            # produce a snapshot. Measure only the next window.
                            for subscriber, _ in subscribers:
                                while subscriber.poll(0):
                                    subscriber.recv_multipart()
                            counts = _collect_snapshots(subscribers, 2.5)
                            assert not flood.done(), f"{name}/{stage}: load ended before observation"
                            status = json.loads(_command(fixture, op="status"))
                            assert status["publisher_healthy"] and status["journal_enabled"]
                        finally:
                            stopped.set()
                        sent = flood.result(timeout=3)
                    if stage == "unmatched":
                        assert counts == [0] * len(subscribers), f"{name}: unmatched filters forced snapshots: {counts}"
                    else:
                        assert all(1 <= count <= 3 for count in counts), f"{name}: snapshot rate unbounded: {counts}"
                    stages[stage] = {"snapshots": counts, "commands_sent": sent}
                # A subscriber joining during the cooldown still gets cached state.
                late = stack.enter_context(context.socket(zmq.SUB))
                late.setsockopt(zmq.SUBSCRIBE, topic_bytes)
                late.setsockopt(zmq.LINGER, 0)
                late.connect(endpoints[0][0])
                assert late.poll(3000), f"{name}: coalescing lost a late subscription"
                frames = late.recv_multipart()
                assert frames[0] == topic_bytes
                events = msgpack.unpackb(frames[2], raw=False)[1]
                assert any(event["type"] == "BlockStored" and event["token_ids"] == [970, 980] for event in events)
                results[name] = stages
        finally:
            _terminate(fixture)
    return results


def _verify_subscription_load(
    fixture_binary: str, context: zmq.Context, ports: set[int], log: TextIO
) -> dict[str, Any]:
    engine_port = _free_ports(2, ports)
    pool_port = _free_ports(2, ports)
    engine = f"tcp://127.0.0.1:{engine_port}"
    pool = f"tcp://127.0.0.1:{pool_port}"
    fixture = subprocess.Popen(
        [fixture_binary, json.dumps({"endpoint": engine, "motor_pool_endpoint": pool}), "--pool"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
    )
    stopped = threading.Event()
    ready = threading.Event()
    try:
        assert _read_line(fixture) == "ready"
        for rank in range(2):
            tokens = [950 + rank, 960 + rank]
            assert _command(fixture, op="store", rank=rank, tokens=tokens) == "ok"
            assert _command(fixture, op="copy", rank=rank, tokens=tokens) == "ok"
        with contextlib.ExitStack() as stack:
            subscribers = []
            for rank in range(2):
                for port, source in ((engine_port, "engine"), (pool_port, "pool")):
                    subscriber = stack.enter_context(context.socket(zmq.SUB))
                    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                    subscriber.setsockopt(zmq.LINGER, 0)
                    subscriber.connect(f"tcp://127.0.0.1:{port + rank}")
                    subscribers.append((subscriber, source, rank))
            # Establish all four subscriptions before starting the command load.
            for subscriber, _, _ in subscribers:
                assert subscriber.poll(3000), "subscription did not receive initial reset"
                subscriber.recv_multipart()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                flood = executor.submit(_flood_subscriptions, context, [engine, pool], ready, stopped)
                try:
                    assert ready.wait(3), "subscription load did not start"
                    tokens = [900, 901]
                    assert _command(fixture, op="store", rank=0, tokens=tokens) == "ok"
                    assert _command(fixture, op="copy", rank=0, tokens=tokens) == "ok"
                    media: set[str] = set()
                    pool_hashes: set[int] = set()
                    host_hash: int | None = None
                    deadline = time.monotonic() + 3
                    engine_subscriber = subscribers[0][0]
                    pool_subscriber = subscribers[1][0]
                    while time.monotonic() < deadline:
                        if engine_subscriber.poll(50):
                            events = msgpack.unpackb(engine_subscriber.recv_multipart()[2], raw=False)[1]
                            for event in events:
                                if event["type"] == "BlockStored" and event["token_ids"] == tokens:
                                    media.add(event["medium"])
                                    if event["medium"] == "cpu":
                                        host_hash = event["block_hashes"][0]
                        if pool_subscriber.poll(50):
                            events = msgpack.unpackb(pool_subscriber.recv_multipart()[2], raw=False)["events"]
                            for event in events:
                                if event["event_type"] == "stored":
                                    pool_hashes.update(event["seq_hashes"])
                        if media == {"gpu", "cpu"} and host_hash in pool_hashes:
                            break
                    assert media == {"gpu", "cpu"}, "subscription load starved residency publication"
                    assert host_hash in pool_hashes, "subscription load starved Host pool publication"
                    assert not flood.done(), "subscription load ended before shutdown"
                    status = json.loads(_command(fixture, op="status"))
                    assert status["publisher_healthy"] and status["journal_enabled"]
                    started = time.monotonic()
                    assert _command(fixture, op="stop") == "stopped"
                    shutdown_seconds = time.monotonic() - started
                    assert shutdown_seconds < 2, "subscription load blocked publisher shutdown"
                finally:
                    stopped.set()
                commands = flood.result(timeout=3)
            for subscriber, source, _ in subscribers:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if not subscriber.poll(100):
                        continue
                    payload = msgpack.unpackb(subscriber.recv_multipart()[2], raw=False)
                    events = payload[1] if source == "engine" else payload["events"]
                    key = "type" if source == "engine" else "event_type"
                    cleared = "AllBlocksCleared" if source == "engine" else "cleared"
                    if len(events) == 1 and events[0][key] == cleared:
                        break
                else:
                    raise AssertionError(f"subscription load: missing final {source} clear")
            assert fixture.wait(timeout=10) == 0
            return {"commands_sent": commands, "shutdown_seconds": shutdown_seconds, "result": "passed"}
    finally:
        stopped.set()
        _terminate(fixture)


def _flood_replays(
    context: zmq.Context, endpoints: list[str], ready: threading.Event, stopped: threading.Event
) -> list[dict[str, Any]]:
    with contextlib.ExitStack() as stack:
        senders = [stack.enter_context(context.socket(zmq.DEALER)) for _ in endpoints]
        pending = [0] * len(senders)
        sent = [0] * len(senders)
        completed = [0] * len(senders)
        generations: list[dict[float, int]] = [{} for _ in senders]
        monitors = []
        for sender, endpoint in zip(senders, endpoints):
            sender.setsockopt(zmq.LINGER, 0)
            sender.setsockopt(zmq.SNDHWM, 64)
            monitors.append(stack.enter_context(sender.get_monitor_socket(events=zmq.EVENT_HANDSHAKE_SUCCEEDED)))
            sender.connect(endpoint)
        for sender, monitor in zip(senders, monitors):
            assert monitor.poll(3000), "replay sender did not complete its handshake"
            monitor.recv_multipart()
            sender.disable_monitor()
        deadline = time.monotonic() + 20
        while not stopped.is_set() and time.monotonic() < deadline:
            for rank, sender in enumerate(senders):
                while pending[rank] < 32:
                    try:
                        sender.send_multipart([b"", bytes(8)], zmq.DONTWAIT)
                    except zmq.Again:
                        break
                    pending[rank] += 1
                    sent[rank] += 1
                while sender.poll(0):
                    frames = sender.recv_multipart()
                    assert len(frames) == 3 and frames[0] == b"" and len(frames[1]) == 8
                    sequence = int.from_bytes(frames[1], "big")
                    if sequence == 2**64 - 1:
                        assert frames[2] == b""
                        pending[rank] -= 1
                        completed[rank] += 1
                        assert pending[rank] >= 0
                        continue
                    payload = msgpack.unpackb(frames[2], raw=False)
                    assert payload[2] == rank
                    assert payload[1][0]["type"] == "AllBlocksCleared"
                    generated_at = payload[0]
                    assert generations[rank].get(generated_at, sequence) == sequence
                    generations[rank][generated_at] = sequence
            if min(completed) > 0:
                ready.set()
            stopped.wait(0.01)
        return [
            {"sent": sent[rank], "completed": completed[rank], "generations": sorted(generations[rank])}
            for rank in range(len(senders))
        ]


def _query_replay(dealer: zmq.Socket) -> tuple[list[Any], int]:
    dealer.send_multipart([b"", bytes(8)])
    frames = dealer.recv_multipart()
    assert len(frames) == 3 and frames[0] == b"" and len(frames[1]) == 8
    sequence = int.from_bytes(frames[1], "big")
    assert sequence != 2**64 - 1
    payload = msgpack.unpackb(frames[2], raw=False)
    end = dealer.recv_multipart()
    assert end == [b"", (2**64 - 1).to_bytes(8, "big"), b""]
    return payload, sequence


def _verify_replay_load(fixture_binary: str, context: zmq.Context, ports: set[int], log: TextIO) -> dict[str, Any]:
    engine_port = _free_ports(2, ports)
    pool_port = _free_ports(2, ports)
    replay_port = _free_ports(2, ports)
    config = {
        "endpoint": f"tcp://127.0.0.1:{engine_port}",
        "motor_pool_endpoint": f"tcp://127.0.0.1:{pool_port}",
        "replay_endpoint": f"tcp://127.0.0.1:{replay_port}",
        "snapshot_interval_seconds": 60,
    }
    fixture = subprocess.Popen(
        [fixture_binary, json.dumps(config)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
    )
    ready = threading.Event()
    stopped = threading.Event()
    try:
        assert _read_line(fixture) == "ready"
        tokens = [list(range(1200 + rank * 200, 1320 + rank * 200)) for rank in range(2)]
        for rank in range(2):
            assert _command(fixture, op="store", rank=rank, tokens=tokens[rank]) == "ok"
            assert _command(fixture, op="copy", rank=rank, tokens=tokens[rank]) == "ok"
        with contextlib.ExitStack() as stack:
            subscribers = []
            for rank in range(2):
                for port, source in ((engine_port, "engine"), (pool_port, "pool")):
                    subscriber = stack.enter_context(context.socket(zmq.SUB))
                    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                    subscriber.setsockopt(zmq.LINGER, 0)
                    subscriber.connect(f"tcp://127.0.0.1:{port + rank}")
                    subscribers.append((subscriber, source))
            assert all(count > 0 for count in _collect_snapshots(subscribers, 3.2))
            endpoints = [f"tcp://127.0.0.1:{replay_port + rank}" for rank in range(2)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                flood = executor.submit(_flood_replays, context, endpoints, ready, stopped)
                try:
                    assert ready.wait(3), "valid replay traffic did not start"
                    # Concurrent normal clients must recover within Motor's
                    # receive timeout despite pipelined requests on both DPs.
                    clients = []
                    for rank, endpoint in enumerate(endpoints):
                        for _ in range(2):
                            dealer = stack.enter_context(context.socket(zmq.DEALER))
                            dealer.setsockopt(zmq.LINGER, 0)
                            dealer.setsockopt(zmq.RCVTIMEO, 2000)
                            dealer.connect(endpoint)
                            clients.append((dealer, rank))
                    normal_started = time.monotonic()
                    for dealer, _ in clients:
                        dealer.send_multipart([b"", bytes(8)])
                    for dealer, rank in clients:
                        frames = dealer.recv_multipart()
                        payload = msgpack.unpackb(frames[2], raw=False)
                        assert payload[2] == rank
                        assert any(event["type"] == "BlockStored" for event in payload[1])
                        assert dealer.recv_multipart()[1] == (2**64 - 1).to_bytes(8, "big")
                    normal_seconds = time.monotonic() - normal_started
                    assert normal_seconds < 2, "replay traffic starved normal clients"
                    assert _command(fixture, op="clear", rank=0, medium="gpu") == "ok"
                    new_tokens = [1800, 1801]
                    assert _command(fixture, op="store", rank=0, tokens=new_tokens) == "ok"
                    assert _command(fixture, op="copy", rank=0, tokens=new_tokens) == "ok"
                    media: set[str] = set()
                    deadline = time.monotonic() + 2
                    while media != {"gpu", "cpu"} and time.monotonic() < deadline:
                        engine = subscribers[0][0]
                        if not engine.poll(50):
                            continue
                        events = msgpack.unpackb(engine.recv_multipart()[2], raw=False)[1]
                        for event in events:
                            if event["type"] == "BlockStored" and event["token_ids"] == new_tokens:
                                media.add(event["medium"])
                    assert media == {"gpu", "cpu"}, "replay traffic starved live residency events"
                    for dealer, rank in clients[:2] + clients[2:3]:
                        payload, sequence = _query_replay(dealer)
                        assert payload[2] == rank and sequence > 0
                        gpu = {
                            tuple(event["token_ids"])
                            for event in payload[1]
                            if event["type"] == "BlockStored" and event["medium"] == "gpu"
                        }
                        expected = (
                            {tuple(new_tokens)}
                            if rank == 0
                            else {tuple(tokens[rank][index : index + 2]) for index in range(0, len(tokens[rank]), 2)}
                        )
                        assert gpu == expected, "replay reused a stale or foreign-DP snapshot"
                    assert not flood.done(), "valid replay traffic ended early"
                    status = json.loads(_command(fixture, op="status"))
                    assert status["publisher_healthy"] and status["journal_enabled"]
                    started = time.monotonic()
                    assert _command(fixture, op="stop") == "stopped"
                    shutdown_seconds = time.monotonic() - started
                    assert shutdown_seconds < 2, "replay traffic blocked publisher shutdown"
                finally:
                    stopped.set()
                traffic = flood.result(timeout=3)
            for rank, counts in enumerate(traffic):
                generated = counts["generations"]
                assert len(generated) >= 2, f"DP {rank}: no sustained replay recovery"
                assert counts["completed"] > len(generated), f"DP {rank}: replay payloads were not coalesced"
                assert all(b - a >= 0.9 for a, b in zip(generated, generated[1:])), (
                    f"DP {rank}: replay snapshots regenerated without cooldown: {generated}"
                )
            for subscriber, source in subscribers:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if not subscriber.poll(100):
                        continue
                    payload = msgpack.unpackb(subscriber.recv_multipart()[2], raw=False)
                    events = payload[1] if source == "engine" else payload["events"]
                    key, cleared = ("type", "AllBlocksCleared") if source == "engine" else ("event_type", "cleared")
                    if len(events) == 1 and events[0][key] == cleared:
                        break
                else:
                    raise AssertionError(f"replay traffic: no final {source} clear")
            assert fixture.wait(timeout=10) == 0
            return {"traffic": traffic, "normal_client_seconds": normal_seconds, "shutdown_seconds": shutdown_seconds}
    finally:
        stopped.set()
        _terminate(fixture)


def _run(fixture_binary: str, conductor_binary: str, report_path: Path) -> None:
    ports: set[int] = set()
    motor_port = _free_ports(1, ports)
    engine_port = _free_ports(2, ports)
    pool_port = _free_ports(2, ports)
    replay_port = _free_ports(2, ports)
    engine = f"tcp://127.0.0.1:{engine_port}"
    pool = f"tcp://127.0.0.1:{pool_port}"
    replay = f"tcp://127.0.0.1:{replay_port}"
    url = f"http://127.0.0.1:{motor_port}"
    registration = argparse.Namespace(
        conductor=url,
        instance_id="xllm-fixture",
        model="fixture",
        block_size=2,
        dp_size=2,
        tenant_id="default",
        engine_endpoint=engine,
        pool_endpoint=pool,
        replay_endpoint=replay,
        unregister=False,
    )
    observed: dict[str, Any] = {}
    tokens = [10, 11, 12, 13, 14, 15]
    other = [20, 21, 22, 23, 24, 25]
    fixture: subprocess.Popen[str] | None = None
    conductor: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="xllm-motor-test-") as directory:
        fixture_log_path = Path(directory) / "fixture.log"
        motor_log_path = Path(directory) / "motor.log"
        with fixture_log_path.open("w+") as fixture_log, motor_log_path.open("w+") as motor_log:
            context = zmq.Context()
            context.setsockopt(zmq.LINGER, 0)
            try:
                conductor = _start_conductor(conductor_binary, motor_port, motor_log)
                config = {
                    "publisher": "zmq",
                    "endpoint": engine,
                    "motor_pool_endpoint": pool,
                    "replay_endpoint": replay,
                    "snapshot_interval_seconds": 60,
                }
                fixture = subprocess.Popen(
                    [fixture_binary, json.dumps(config), "--pool"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=fixture_log,
                    text=True,
                    bufsize=1,
                )
                assert _read_line(fixture) == "ready"
                # Populate caches before subscription to exercise snapshot recovery.
                assert _command(fixture, op="store", rank=0, tokens=tokens) == "ok"
                assert _command(fixture, op="store", rank=1, tokens=other) == "ok"
                draft = json.loads(_command(fixture, op="create_draft", prefix_cache=True))
                assert not draft["draft_journal_attached"] and not draft["draft_publisher_healthy"]
                state = json.loads(_command(fixture, op="state"))
                assert all(event["model_id"] == "fixture" for event in state["events"])
                _register(registration)

                def _expect(name: str, query_tokens: list[int], rank: int, npu: int, cpu: int) -> None:
                    deadline = time.monotonic() + 8
                    result: dict[str, Any] = {}
                    while time.monotonic() < deadline:
                        result = _post_json(
                            url, "/query", {"model": "fixture", "block_size": 2, "token_ids": query_tokens}
                        )
                        counts = result.get("default", {}).get("xllm-fixture", {}).get("DP", {}).get(str(rank), {})
                        if counts.get("npu_blocks", 0) == npu and counts.get("cpu_blocks", 0) == cpu:
                            observed[name] = result
                            logger.info("%s passed: DP %d, NPU %d, CPU %d", name, rank, npu, cpu)
                            return
                        time.sleep(0.05)
                    raise AssertionError(f"{name}: expected NPU={npu}, CPU={cpu}, got {result}")

                _expect("device", tokens, 0, 3, 0)
                _expect("dp_isolation", tokens, 1, 0, 0)
                _expect("other_dp", other, 1, 3, 0)

                # Inspect the real multipart wire and replay termination marker.
                with context.socket(zmq.SUB) as subscriber:
                    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                    subscriber.setsockopt(zmq.RCVTIMEO, 3000)
                    subscriber.connect(engine)
                    frames = subscriber.recv_multipart()
                    assert len(frames) == 3 and len(frames[1]) == 8
                    batch = msgpack.unpackb(frames[2], raw=False)
                    assert batch[2] == 0
                    stored = next(event for event in batch[1] if event["type"] == "BlockStored")
                    assert stored["token_ids"] == tokens[:2] and stored["block_size"] == 2
                    observed["multi_subscriber_immediate_snapshot"] = "passed"
                    while subscriber.poll(100):
                        frames = subscriber.recv_multipart()
                    live_sequence = int.from_bytes(frames[1], "big")
                    assert live_sequence > 0
                    with context.socket(zmq.DEALER) as dealer:
                        dealer.setsockopt(zmq.RCVTIMEO, 3000)
                        dealer.setsockopt(zmq.LINGER, 0)
                        dealer.connect(replay)

                        def _expect_replay() -> None:
                            dealer.send_multipart([b"", live_sequence.to_bytes(8, "big")])
                            replay_frames = dealer.recv_multipart()
                            assert len(replay_frames) == 3 and replay_frames[0] == b""
                            assert int.from_bytes(replay_frames[1], "big") == live_sequence
                            assert any(
                                event["type"] == "BlockStored"
                                for event in msgpack.unpackb(replay_frames[2], raw=False)[1]
                            )
                            assert int.from_bytes(dealer.recv_multipart()[1], "big") == 2**64 - 1

                        _expect_replay()
                        observed["nonzero_replay_sequence"] = "passed"
                        for malformed in (
                            [b""] * 17,
                            [b""] * 257,
                            [b"bad delimiter", b"12345678"],
                            [b"", b"short"],
                            [b"x" * 2048] * 3,
                        ):
                            dealer.send_multipart(malformed)
                            _expect_replay()
                            assert json.loads(_command(fixture, op="status"))["publisher_healthy"]
                        observed["malformed_replay_isolated"] = "passed"
                    assert _command(fixture, op="store", rank=0, tokens=[90, 91]) == "ok"
                    frames = subscriber.recv_multipart()
                    assert int.from_bytes(frames[1], "big") == live_sequence + 1
                    assert any(
                        event["type"] == "BlockStored" and event["token_ids"] == [90, 91]
                        for event in msgpack.unpackb(frames[2], raw=False)[1]
                    )
                    observed["live_sequence_after_replay"] = "passed"
                observed["wire_and_replay"] = "passed"

                assert _command(fixture, op="copy", rank=0, tokens=tokens) == "ok"
                state = json.loads(_command(fixture, op="state"))
                assert sum(event["medium"] == "cpu" for event in state["events"]) == 3
                _expect("replicas_counted_once", tokens, 0, 3, 0)
                assert _command(fixture, op="clear", rank=0, medium="gpu") == "ok"
                _expect("host_after_device_eviction", tokens, 0, 0, 3)
                _expect("host_dp_isolation", tokens, 1, 0, 0)
                assert _command(fixture, op="store", rank=0, tokens=tokens[:2]) == "ok"
                _expect("mixed_tiers", tokens, 0, 1, 2)
                assert _command(fixture, op="clear", rank=0, medium="gpu") == "ok"
                _expect("host_before_restart", tokens, 0, 0, 3)

                _terminate(conductor)
                conductor = _start_conductor(conductor_binary, motor_port, motor_log)
                _register(registration)
                _expect("conductor_restart", tokens, 0, 0, 3)
                _expect("other_dp_after_restart", other, 1, 3, 0)
                assert _command(fixture, op="clear", rank=0, medium="cpu") == "ok"
                _expect("host_eviction", tokens, 0, 0, 0)
                deadline = time.monotonic() + 2.2
                while time.monotonic() < deadline:
                    result = _post_json(url, "/query", {"model": "fixture", "block_size": 2, "token_ids": tokens})
                    counts = result.get("default", {}).get("xllm-fixture", {}).get("DP", {}).get("0", {})
                    assert counts.get("npu_blocks", 0) == 0 and counts.get("cpu_blocks", 0) == 0
                    time.sleep(0.1)
                observed["host_eviction_during_repair"] = "passed"
                _expect("unaffected_dp", other, 1, 3, 0)
                registration.unregister = True
                _register(registration)
                _expect("unregister", other, 1, 0, 0)
                assert _command(fixture, op="state")
                _verify_lifecycle(fixture_binary, context, ports, fixture_log)
                observed["disabled_directory_all_streams"] = "passed"
                observed["shutdown_clear_all_streams"] = "passed"
                observed["external_draft_target_reporting"] = "passed"
                _verify_non_caching_draft(fixture_binary, ports, fixture_log)
                observed["non_caching_draft_isolation"] = "passed"
                observed["subscription_load"] = _verify_subscription_load(fixture_binary, context, ports, fixture_log)
                observed["subscription_snapshots"] = _verify_subscription_snapshots(
                    fixture_binary, context, ports, fixture_log
                )
                observed["replay_load"] = _verify_replay_load(fixture_binary, context, ports, fixture_log)
                report_path.write_text(json.dumps(observed, indent=2) + "\n")
                logger.info("Motor integration passed; report: %s", report_path)
            except Exception:
                fixture_log.flush()
                motor_log.flush()
                logger.error("Fixture log:\n%s", fixture_log_path.read_text()[-8000:])
                logger.error("Motor log:\n%s", motor_log_path.read_text()[-12000:])
                raise
            finally:
                _terminate(fixture)
                _terminate(conductor)
                context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--conductor", required=True)
    parser.add_argument("--report", type=Path, default=Path("motor_kv_events_report.json"))
    args = parser.parse_args()
    _run(args.fixture, args.conductor, args.report)


if __name__ == "__main__":
    main()
