
#!/usr/bin/env python3
"""
Purpose:
Tests the complete π₀.₅-DROID inference pipeline using the Franka robot,
dual ZED cameras, and Robotiq gripper without executing robot actions.

Technical flow:
1. Connect to Franka and read the current seven joint positions.
2. Read the Robotiq gripper position and convert it to a binary state.
3. Capture external and wrist RGB images from the ZED cameras.
4. Send observations and language instruction to the π₀.₅ server.
5. Print the predicted action chunk and inference time.

This script does not send robot or gripper movement commands.
"""

import numpy as np
from pylibfranka import Robot, RealtimeConfig

from cameras.zed_camera import DualZEDCameras
from gripper.robotiq_gripper import RobotiqGripper
from inference.pi05_client import Pi05Client


ROBOT_IP = "192.168.103.1"
PI05_SERVER = "ws://127.0.0.1:8000"
INSTRUCTION = "pick up the object"


def main():
    robot = Robot(ROBOT_IP)
    cameras = None
    gripper = None

    try:
        cameras = DualZEDCameras()
        gripper = RobotiqGripper()
        client = Pi05Client(server_url=PI05_SERVER)

        # Read current Franka joint positions.
        robot_state = robot.read_once()
        joint_positions = np.asarray(robot_state.q, dtype=np.float32)

        # Read the current gripper position.
        gripper_position = gripper.get_position_normalized()
        gripper_position = float(gripper_position > 0.5)

        # Build the eight-dimensional robot state.
        robot_state = np.concatenate([
            joint_positions,
            np.array([gripper_position], dtype=np.float32),
        ])

        # Capture RGB observations.
        external_image, wrist_image = cameras.get_frames_rgb()

        # Run π₀.₅ inference.
        predicted_actions, inference_ms = client.predict(
            external_image=external_image,
            wrist_image=wrist_image,
            instruction=INSTRUCTION,
            robot_state=robot_state,
        )

        print("\n" + "=" * 60)
        print("π₀.₅ INFERENCE RESULT")
        print("=" * 60)
        print("Robot state:", robot_state)
        print("External image:", external_image.shape)
        print("Wrist image:", wrist_image.shape)
        print("Action shape:", predicted_actions.shape)
        print("Inference time:", f"{inference_ms:.1f} ms")
        print("Predicted actions:\n", predicted_actions)
        print("=" * 60)
        print("Inference completed. No robot actions executed.")

    finally:
        if cameras is not None:
            cameras.close()

        if gripper is not None:
            gripper.close_connection()


if __name__ == "__main__":
    main()
