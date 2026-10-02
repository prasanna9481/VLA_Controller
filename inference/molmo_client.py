#!/usr/bin/env python3
"""
Purpose:
Provides a lightweight client interface for sending robot observations to the
local MolmoAct2-DROID inference server and receiving predicted action chunks.

Workflow:
1. Receive the external RGB image, wrist RGB image, language instruction,
   and current robot state.
2. Convert images to uint8 and robot state to float32 NumPy arrays.
3. Build the MolmoAct2 observation payload.
4. Serialize NumPy data using `json_numpy`.
5. Send the observation to the `/act` HTTP endpoint.
6. Decode the server response and return the predicted action trajectory
   together with the reported inference time.

Technical details:
- Default server endpoint: `http://127.0.0.1:8000/act`.
- Camera inputs must be RGB uint8 arrays with shape `(H, W, 3)`.
- Robot state must contain 8 float32 values:
  `[q1, q2, q3, q4, q5, q6, q7, gripper_state]`.
- NumPy arrays are serialized using `json_numpy` because the MolmoAct2
  server expects this request format.
- The returned `actions` array contains the continuous robot action chunk
  produced by MolmoAct2-DROID.
- This class performs inference communication only; it does not control
  the Franka robot or Robotiq gripper directly.
"""

import numpy as np
import requests
import json_numpy


class MolmoActClient:
    def __init__(
        self,
        server_url="http://127.0.0.1:8000/act",
        timeout=120,
    ):
        self.server_url = server_url
        self.timeout = timeout

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
            "external_cam": external_image,
            "wrist_cam": wrist_image,
            "instruction": instruction,
            "state": robot_state,
        }

        body = json_numpy.dumps(payload)

        response = requests.post(
            self.server_url,
            data=body,
            headers={
                "Content-Type": "application/json"
            },
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"MolmoAct server returned "
                f"{response.status_code}: "
                f"{response.text}"
            )

        result = json_numpy.loads(response.text)

        actions = np.asarray(
            result["actions"],
            dtype=np.float32,
        )

        return actions, result.get("dt_ms")