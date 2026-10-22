
#!/usr/bin/env python3
"""
Purpose:
Provides a lightweight client interface for sending robot observations to the
local π₀.₅-DROID inference server and receiving predicted action chunks.

Workflow:
1. Receive the external RGB image, wrist RGB image, language instruction,
   and current robot state.
2. Convert images to uint8 and robot state to float32 NumPy arrays.
3. Build the π₀.₅-DROID observation payload.
4. Send the observation to the OpenPI WebSocket inference server.
5. Decode the server response and return the predicted action trajectory
   together with the measured inference round-trip time.

Technical details:
- Default server endpoint: ws://127.0.0.1:8000.
- Camera inputs must be RGB uint8 arrays with shape (H, W, 3).
- Robot state must contain 8 float32 values:
  [q1, q2, q3, q4, q5, q6, q7, gripper_state].
- The OpenPI client handles observation serialization automatically.
- The returned actions array contains the continuous action chunk
  produced by π₀.₅-DROID.
- This class performs inference communication only; it does not control
  the Franka robot or Robotiq gripper directly.
"""

import time
import numpy as np
from openpi_client import websocket_client_policy


class Pi05Client:
    def __init__(
        self,
        server_url="ws://127.0.0.1:8000",
        timeout=120,
    ):
        self.server_url = server_url
        self.timeout = timeout

        from urllib.parse import urlparse

        parsed_url = urlparse(self.server_url)

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=parsed_url.hostname,
            port=parsed_url.port or 8000,
        )

    def predict(
        self,
        external_image,
        wrist_image,
        instruction,
        robot_state,
    ):
        """
        external_image:
            np.ndarray RGB uint8, shape (H, W, 3)

        wrist_image:
            np.ndarray RGB uint8, shape (H, W, 3)

        instruction:
            string

        robot_state:
            np.ndarray float32 shape (8,)
            [q1, q2, q3, q4, q5, q6, q7, gripper_state]
        """

        external_image = np.asarray(
            external_image,
            dtype=np.uint8,
        )

        wrist_image = np.asarray(
            wrist_image,
            dtype=np.uint8,
        )

        robot_state = np.asarray(
            robot_state,
            dtype=np.float32,
        )

        if robot_state.shape != (8,):
            raise ValueError(
                f"Expected robot_state shape (8,), "
                f"got {robot_state.shape}"
            )

        payload = {
            "observation/exterior_image_1_left": external_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": robot_state[:7],
            "observation/gripper_position": robot_state[7:8],
            "prompt": instruction,
        }

        start_time = time.perf_counter()

        result = self.client.infer(payload)

        dt_ms = (time.perf_counter() - start_time) * 1000.0

        actions = np.asarray(
            result["actions"],
            dtype=np.float32,
        )

        return actions, dt_ms
