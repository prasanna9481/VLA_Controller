#!/usr/bin/env python3
"""
Purpose:
Provides a lightweight client interface for sending robot observations to the
local Duet-DINO inference server and receiving predicted Cartesian actions.

Workflow:
1. Connect to the locally hosted Duet-DINO TCP server.
2. Initialize the task using external and wrist goal images.
3. Send the current external image, wrist image, and end-effector pose.
4. Receive the predicted 7D relative Cartesian action.
5. Return the action and inference time.

Technical details:
- Default server address: 127.0.0.1:8000.
- Communication uses TCP sockets with an 8-byte unsigned length header.
- Payloads are serialized using Python pickle.
- The server supports reset, set_goal_images, infer, and metadata requests.
- The returned action contains 7 values:
  [dx, dy, dz, rx, ry, rz, gripper].
- Rotation values are already converted to rotation vectors by the server.
- This class performs inference communication only; it does not control
  the Franka robot or Robotiq gripper.
"""

import socket
import struct
import pickle
import time

import numpy as np


HEADER = struct.Struct("!Q")


class DuetDINOClient:
    def __init__(self, host="127.0.0.1", port=8000, timeout=120):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None

    def connect(self):
        self.socket = socket.create_connection(
            (self.host, self.port),
            timeout=self.timeout,
        )

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _send(self, payload):
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        self.socket.sendall(HEADER.pack(len(data)) + data)

    def _recv_exact(self, size):
        chunks = []
        while size:
            chunk = self.socket.recv(size)
            if not chunk:
                raise ConnectionError("Duet-DINO server disconnected.")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def _receive(self):
        size = HEADER.unpack(self._recv_exact(HEADER.size))[0]
        return pickle.loads(self._recv_exact(size))

    def _request(self, payload):
        if self.socket is None:
            self.connect()

        self._send(payload)
        response = self._receive()

        if "error" in response:
            raise RuntimeError(response["error"])

        return response

    def reset(self):
        return self._request({"method": "reset"})

    def set_goal_images(
        self,
        external_image,
        wrist_image,
        instruction="goal",
        env_id=0,
        run_idx=None,
    ):
        payload = {
            "method": "set_goal_images",
            "external_image": external_image,
            "wrist_image": wrist_image,
            "instruction": instruction,
            "env_id": env_id,
            "run_idx": run_idx,
        }

        return self._request(payload)

    def predict(
        self,
        external_image,
        wrist_image,
        ee_pose,
        instruction="",
        env_id=0,
    ):
        """
        external_image:
            External camera observation in the format expected by Duet-DINO.

        wrist_image:
            Wrist camera observation in the format expected by Duet-DINO.

        ee_pose:
            Current end-effector pose in the format expected by Duet-DINO.

        instruction:
            Task identifier used by the server.

        Returns:
            action:
                np.ndarray float32, shape (7,)

            inference_time_ms:
                Client-observed inference request latency in milliseconds.
        """

        ee_pose = np.asarray(ee_pose, dtype=np.float32)

        payload = {
            "method": "infer",
            "obs": {
                "external_image": external_image,
                "wrist_image": wrist_image,
                "ee_pose": ee_pose,
            },
            "instruction": instruction,
            "env_id": env_id,
        }

        start_time = time.perf_counter()
        result = self._request(payload)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        action = np.asarray(result["action"], dtype=np.float32)

        if action.shape != (7,):
            raise ValueError(
                f"Expected Duet-DINO action shape (7,), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("Duet-DINO returned non-finite action values.")

        return action, inference_time_ms

    def metadata(self):
        return self._request({"method": "metadata"})

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()