#!/usr/bin/env python3
"""FR3 MuJoCo FK/IK without a robot connection."""
from pathlib import Path
import numpy as np
import yaml
from dm_control import mjcf
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class FrankaIK:
    def __init__(self, config=None, max_joint_change=0.50):
        if config is None:
            with (Path(__file__).resolve().parent / "config.yaml").open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)

        franka = config["franka"]
        self.lower = np.asarray(franka["joint_lower_limit"], dtype=float) + 0.02
        self.upper = np.asarray(franka["joint_upper_limit"], dtype=float) - 0.02
        self.max_joint_change = float(max_joint_change)
        # Fixed wrist-site-to-EE translation, expressed in the wrist-site frame.
        self.wrist_to_ee = np.array([0.000095, 0.000038, 0.149868])

        model_path = Path(__file__).resolve().parents[1] / "droid/droid/robot_ik/franka/fr3.xml"
        self.model = mjcf.from_path(str(model_path))
        self.physics = mjcf.Physics.from_mjcf_model(self.model)
        self.joints = self.model.find_all("joint")
        # self.wrist = self.model.find("site", "wrist_site")
        self.wrist = self.model.find("body", "panda_link8")
        if len(self.joints) != 7 or self.wrist is None:
            raise RuntimeError("Invalid FR3 kinematic model.")

    # def forward_kinematics(self, joints):
    #     self.physics.bind(self.joints).qpos[:] = joints
    #     self.physics.forward()
    #     wrist = self.physics.bind(self.wrist)
    #     rotation = wrist.xmat.reshape(3, 3).copy()
    #     return wrist.xpos.copy() + rotation @ self.wrist_to_ee, rotation

    def forward_kinematics(self, joints):
            self.physics.bind(self.joints).qpos[:] = joints
            self.physics.forward()
            wrist = self.physics.bind(self.wrist)
            rotation = wrist.xmat.reshape(3, 3).copy()
            return wrist.xpos.copy(), rotation

    def solve(self, current_q, target_position, target_rotation):
        current_q = np.asarray(current_q, dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        target_rotation = np.asarray(target_rotation, dtype=float)
        if current_q.shape != (7,) or target_position.shape != (3,) or target_rotation.shape != (3, 3):
            raise ValueError("Expected joint (7,), position (3,), rotation (3,3).")
        if not all(np.all(np.isfinite(value)) for value in (current_q, target_position, target_rotation)):
            raise ValueError("Non-finite IK input.")
        if np.any(current_q < self.lower) or np.any(current_q > self.upper):
            raise RuntimeError("Initial joints violate IK limits with 0.02 rad margin.")

        def residual(joints):
            position, rotation = self.forward_kinematics(joints)
            orientation_error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
            return np.concatenate((target_position - position, 0.1 * orientation_error,
                                   0.01 * (joints - current_q)))

        result = least_squares(residual, current_q, bounds=(self.lower, self.upper),
                               max_nfev=300, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        position, rotation = self.forward_kinematics(result.x)
        position_error = np.linalg.norm(target_position - position)
        orientation_error = np.linalg.norm(Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec())
        joint_change = np.max(np.abs(result.x - current_q))

        print(f"IK converged: {result.success} | position: {position_error * 1000:.3f} mm | "
              f"orientation: {np.rad2deg(orientation_error):.3f} deg | joint change: {joint_change:.4f} rad")
        if not result.success:
            raise RuntimeError("IK failed to converge.")
        # if position_error > 0.002:
        #     raise RuntimeError("IK position tolerance exceeded (2 mm).")
        # if orientation_error > np.deg2rad(2.0):
        #     raise RuntimeError("IK orientation tolerance exceeded (2 deg).")
        # if joint_change > self.max_joint_change:
        #     raise RuntimeError("IK joint displacement exceeds configured limit.")
        return result.x