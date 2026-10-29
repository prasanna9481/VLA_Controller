#!/usr/bin/env python3

import numpy as np
import mujoco
from pylibfranka import Robot, RealtimeConfig

ROBOT_IP = "192.168.103.1"
MODEL_PATH = "droid/droid/robot_ik/franka/fr3.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Robot(ROBOT_IP, RealtimeConfig.kIgnore)

wrist_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SITE, "wrist_site"
)

state = robot.read_once()
joint_positions = np.asarray(state.q, dtype=float)

# Calculate MuJoCo forward kinematics
data.qpos[:7] = joint_positions
mujoco.mj_forward(model, data)

wrist_position = data.site_xpos[wrist_id].copy()
wrist_rotation = data.site_xmat[wrist_id].reshape(3, 3).copy()

# Read Franka end-effector transform
franka_transform = np.asarray(
    state.O_T_EE, dtype=float
).reshape(4, 4, order="F")

franka_position = franka_transform[:3, 3]
franka_rotation = franka_transform[:3, :3]

# Relative transform between MuJoCo wrist and Franka EE
local_offset = wrist_rotation.T @ (franka_position - wrist_position)
relative_rotation = wrist_rotation.T @ franka_rotation

print("\n========== FR3 FK VALIDATION ==========")
print("Joint positions:", np.round(joint_positions, 6))
print("MuJoCo wrist position:", np.round(wrist_position, 6))
print("Franka EE position:", np.round(franka_position, 6))
print("Local offset (m):", np.round(local_offset, 6))
print("Relative rotation:\n", np.round(relative_rotation, 6))
