#!/usr/bin/env python3

import numpy as np
from pylibfranka import (
    Robot,
    ControllerMode,
    JointPositions,
    RealtimeConfig,
)

ROBOT_IP = "192.168.103.1"


def main():
    robot = None

    try:
        print(f"Connecting to Franka at {ROBOT_IP}...")

        robot = Robot(
            ROBOT_IP,
            RealtimeConfig.kIgnore,
        )

        control = robot.start_joint_position_control(
            ControllerMode.JointImpedance
        )

        # Read state only after starting the active controller.
        state, _ = control.readOnce()

        # Start from Franka's current desired joint position.
        start_q = np.asarray(
            getattr(state, "q_d", state.q),
            dtype=np.float64,
        )

        command_q = start_q.copy()

        elapsed = 0.0
        duration = 5.0

        # Very small and slow test motion.
        amplitude = 0.03      # rad
        frequency = 0.10      # Hz

        print("Starting slow test motion...")

        while elapsed < duration:
            state, period = control.readOnce()
            dt = float(period.to_sec())
            elapsed += dt

            command_q[:] = start_q

            # Smooth cosine trajectory:
            # starts with zero velocity.
            delta = amplitude * (
                1.0 - np.cos(
                    2.0 * np.pi * frequency * elapsed
                )
            )

            command_q[1] = start_q[1] + delta

            command = JointPositions(
                command_q.tolist()
            )

            if elapsed >= duration:
                command.motion_finished = True

            control.writeOnce(command)

        print("Test completed.")

    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()