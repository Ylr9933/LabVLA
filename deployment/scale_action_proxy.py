#!/usr/bin/env python3
"""WebSocket proxy that scales LabVLA arm deltas before they reach a client.

The LabVLA server returns denormalized absolute arm targets even when the
checkpoint action mode is ``delta``.  This proxy therefore scales
``target - current_state`` and then reconstructs the absolute target.  Gripper
and action-pad dimensions are forwarded unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

import msgpack
import numpy as np
import websockets


LOGGER = logging.getLogger("scale_action_proxy")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise TypeError(f"unsupported numpy dtype: {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot pack object of type {type(value).__name__}")


def _unpack_numpy(value: dict) -> Any:
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        shape = tuple(int(item) for item in value[b"shape"])
        data = value[b"data"]
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(data) != expected:
            raise ValueError(
                f"invalid ndarray payload: got {len(data)} bytes, expected {expected}"
            )
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_message(value: Any) -> bytes:
    return msgpack.packb(value, default=_pack_numpy)


def unpack_message(payload: bytes) -> Any:
    return msgpack.unpackb(payload, object_hook=_unpack_numpy)


def scale_response(
    request: dict[str, Any],
    response: dict[str, Any],
    gain: float,
    arm_dims: int,
) -> dict[str, Any]:
    if "error" in response or "actions" not in response:
        return response

    state = np.asarray(request.get("state"), dtype=np.float32)
    actions = np.asarray(response["actions"], dtype=np.float32)
    if state.ndim != 1 or state.shape[0] < arm_dims:
        raise ValueError(
            f"request state must be 1-D with at least {arm_dims} values, "
            f"got {state.shape}"
        )
    if actions.ndim != 2 or actions.shape[1] < arm_dims:
        raise ValueError(
            f"response actions must be 2-D with at least {arm_dims} columns, "
            f"got {actions.shape}"
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(actions)):
        raise ValueError("state/actions contain NaN or infinity")

    current = state[:arm_dims]
    original_delta = actions[:, :arm_dims] - current[None, :]
    actions[:, :arm_dims] = current[None, :] + gain * original_delta
    response["actions"] = actions

    LOGGER.info(
        "scaled %d actions: gain=%.3f, delta_abs_max %.4f -> %.4f deg",
        actions.shape[0],
        gain,
        float(np.max(np.abs(original_delta))),
        float(np.max(np.abs(actions[:, :arm_dims] - current[None, :]))),
    )
    return response


async def proxy_connection(
    downstream,
    *,
    upstream_uri: str,
    upstream_token: str,
    gain: float,
    arm_dims: int,
) -> None:
    headers = {"Authorization": f"Bearer {upstream_token}"} if upstream_token else None
    connect_kwargs: dict[str, Any] = {
        "open_timeout": 10,
        "ping_timeout": 120,
        "max_size": 16 * 1024 * 1024,
    }
    if headers:
        # websockets <=13 calls this argument ``extra_headers``; websockets
        # 14+ renamed it to ``additional_headers``.
        try:
            ws_major = int(websockets.__version__.split(".", 1)[0])
        except (AttributeError, ValueError):
            ws_major = 16
        header_arg = "additional_headers" if ws_major >= 14 else "extra_headers"
        connect_kwargs[header_arg] = headers

    async with websockets.connect(upstream_uri, **connect_kwargs) as upstream:
        metadata = unpack_message(await upstream.recv())
        if not isinstance(metadata, dict):
            raise ValueError("upstream metadata is not a map")
        if metadata.get("action_mode") != "delta":
            raise ValueError(
                "this proxy expects a delta-mode LabVLA server, got "
                f"{metadata.get('action_mode')!r}"
            )
        if int(metadata.get("state_dim", 0)) < arm_dims:
            raise ValueError("upstream state_dim is smaller than arm_dims")
        if int(metadata.get("action_dim", 0)) < arm_dims:
            raise ValueError("upstream action_dim is smaller than arm_dims")

        await downstream.send(pack_message(metadata))
        LOGGER.info("connected upstream=%s gain=%.3f", upstream_uri, gain)

        async for raw_request in downstream:
            request = unpack_message(raw_request)
            if not isinstance(request, dict):
                raise ValueError("request is not a map")
            await upstream.send(pack_message(request))
            raw_response = await asyncio.wait_for(upstream.recv(), timeout=120)
            response = unpack_message(raw_response)
            if not isinstance(response, dict):
                raise ValueError("upstream response is not a map")
            response = scale_response(request, response, gain, arm_dims)
            await downstream.send(pack_message(response))


async def run(args: argparse.Namespace) -> None:
    async def handler(websocket) -> None:
        try:
            await proxy_connection(
                websocket,
                upstream_uri=args.upstream_uri,
                upstream_token=args.upstream_token,
                gain=args.gain,
                arm_dims=args.arm_dims,
            )
        except websockets.exceptions.ConnectionClosed:
            LOGGER.info("client disconnected")
        except Exception:
            LOGGER.exception("proxy connection failed")
            await websocket.close(code=1011, reason="action proxy failure")

    async with websockets.serve(
        handler,
        args.host,
        args.port,
        max_size=16 * 1024 * 1024,
        ping_timeout=120,
    ):
        LOGGER.info(
            "listening on ws://%s:%d -> %s, gain=%.3f",
            args.host,
            args.port,
            args.upstream_uri,
            args.gain,
        )
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scale JAKA arm deltas returned by a LabVLA WebSocket server."
    )
    parser.add_argument("--upstream-uri", default="ws://127.0.0.1:31003")
    parser.add_argument("--upstream-token", default="labvla_test_2026")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31002)
    parser.add_argument("--gain", type=float, default=4.0)
    parser.add_argument("--arm-dims", type=int, default=6)
    args = parser.parse_args()
    if not np.isfinite(args.gain) or args.gain <= 0.0 or args.gain > 20.0:
        parser.error("--gain must be finite and in (0, 20]")
    if args.arm_dims != 6:
        parser.error("this JAKA proxy expects exactly 6 arm dimensions")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run(parse_args()))
