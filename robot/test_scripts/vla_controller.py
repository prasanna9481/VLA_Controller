#!/usr/bin/env python3
"""
MolmoAct2 Franka VLA Client

Workflow:
1. Connect to the Franka, Robotiq gripper, side ZED, and wrist ZED.
2. Read RGB images, Franka joint positions, and gripper state.
3. Build the MolmoAct2 observation:
   [side image, wrist image, instruction, q1..q7, gripper].
4. Send the observation to the local MolmoAct2-DROID server over HTTP.
5. Receive a continuous action chunk with shape (15, 8).
6. In DRY_RUN mode, only print predictions; no robot or gripper motion is executed.

Technical notes:
- MolmoAct2 server: http://127.0.0.1:8000/act
- Camera images are RGB uint8 NumPy arrays.
- Robot state is float32 with 8 values.
- Gripper-to-DROID state mapping is not yet confirmed, so the gripper model state remains 0.0.
"""

import numpy as np
from pylibfranka import Robot, RealtimeConfig
from cameras.zed_camera import DualZEDCameras
from gripper.robotiq_gripper import RobotiqGripper
from inference.molmo_client import MolmoActClient

# ============================================================
# CONFIGURATION
# ============================================================

ROBOT_IP = "192.168.103.1"
MOLMO_SERVER = "http://127.0.0.1:8000/act"
INSTRUCTION = "pick up the object"
DRY_RUN = True

# ============================================================
# VLA CONTROLLER
# ============================================================

class FrankaVLAController:
    def __init__(self):
        print(f"Connecting to Franka at {ROBOT_IP}...")
        self.robot = Robot(
            ROBOT_IP,
            RealtimeConfig.kIgnore,
        )
        print("Franka connected.")

        print("Connecting to Robotiq gripper...")
        self.gripper = RobotiqGripper()
        print("Robotiq gripper connected.")

        print("Opening cameras...")
        self.cameras = DualZEDCameras()
        print("Cameras ready.")

        self.molmo = MolmoActClient(server_url=MOLMO_SERVER)
        print("MolmoAct client ready.")

    # ========================================================
    # FRANKA STATE
    # ========================================================

    def get_joint_state(self):
        state = self.robot.read_once()
        q = np.asarray(state.q, dtype=np.float32)

        if q.shape != (7,):
            raise RuntimeError(f"Expected 7 joints, got {q.shape}")

        return q

    # ========================================================
    # GRIPPER STATE
    # ========================================================

    def get_gripper_model_state(self):
        """
        Temporary placeholder.

        The exact MolmoAct2-DROID gripper-state convention has
        not yet been mapped to the Robotiq Modbus position.
        """
        return np.float32(0.0)

    # ========================================================
    # MODEL STATE
    # ========================================================

    def get_model_state(self):
        q = self.get_joint_state()
        gripper_state = self.get_gripper_model_state()

        state = np.concatenate([
            q,
            np.array([gripper_state], dtype=np.float32),
        ])

        if state.shape != (8,):
            raise RuntimeError(f"Expected model state shape (8,), got {state.shape}")

        return state

    # ========================================================
    # OBSERVATION
    # ========================================================

    def get_observation(self):
        side_rgb, wrist_rgb = self.cameras.get_frames_rgb()
        robot_state = self.get_model_state()

        return side_rgb, wrist_rgb, robot_state

    # ========================================================
    # MOLMOACT2 INFERENCE
    # ========================================================

    def infer(self, instruction):
        side_rgb, wrist_rgb, robot_state = self.get_observation()

        print()
        print("=" * 60)
        print("OBSERVATION")
        print("=" * 60)
        print("Side image:", side_rgb.shape, side_rgb.dtype)
        print("Wrist image:", wrist_rgb.shape, wrist_rgb.dtype)
        print("Robot state:", robot_state)
        print("Instruction:", instruction)

        actions, inference_ms = self.molmo.predict(
            external_image=side_rgb,
            wrist_image=wrist_rgb,
            instruction=instruction,
            robot_state=robot_state,
        )

        print()
        print("=" * 60)
        print("MOLMOACT2 OUTPUT")
        print("=" * 60)

        if inference_ms is not None:
            print(f"Inference time: {inference_ms:.1f} ms")

        print("Action shape:", actions.shape)
        print("Action dtype:", actions.dtype)

        if actions.ndim != 2 or actions.shape[1] != 8:
            raise RuntimeError(f"Unexpected MolmoAct2 action shape: {actions.shape}")

        for i, action in enumerate(actions):
            print(f"{i:02d}: {np.array2string(action, precision=4)}")

        return actions

    # ========================================================
    # RUN
    # ========================================================

    def run(self, instruction):
        print()
        print("=" * 60)
        print("MolmoAct2 Franka Client")
        print("=" * 60)
        print(f"Instruction: {instruction}")
        print(f"DRY_RUN: {DRY_RUN}")

        try:
            while True:
                actions = self.infer(instruction)

                if DRY_RUN:
                    print()
                    print("DRY RUN: actions were NOT sent to the robot or gripper.")

                input("\nPress ENTER for next inference or Ctrl+C to quit...")

        except KeyboardInterrupt:
            print("\nStopping.")

        finally:
            print("Closing cameras...")
            self.cameras.close()

            print("Closing Robotiq connection...")
            self.gripper.close_connection()

            print("Shutdown complete.")

# ============================================================
# MAIN
# ============================================================

def main():
    controller = FrankaVLAController()
    controller.run(instruction=INSTRUCTION)

if __name__ == "__main__":
    main()