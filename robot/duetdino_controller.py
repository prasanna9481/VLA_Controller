#!/usr/bin/env python3
"""
Purpose:
Runs the complete Duet-DINO goal-image control pipeline for a Franka robot
with dual ZED cameras and a Robotiq gripper.

Technical flow:
1. Load experiment, robot, model, IK, safety and runtime parameters from config.yaml.
2. Run the Franka/Ruckig joint controller in a separate multiprocessing process.
3. Exchange joint targets and measured states through versioned shared memory.
4. Capture ZED observations and read the current robot and gripper states.
5. Request base-frame Cartesian translation and rotation-vector deltas from Duet-DINO.
6. Check action bounds, convert the target pose to joints using FrankaIK,
   and validate the sampled Ruckig trajectory in Cartesian and joint space.
7. Publish arm and gripper targets, wait for motion to settle, and compare the
   measured pose with the absolute predicted target.
8. Save the experiment configuration, action logs, images and optional videos.

Important libraries:
- pylibfranka: Franka state access and joint-position control.
- ruckig: online joint trajectory generation and path validation.
- multiprocessing: separates Franka control from inference and camera capture.
- scipy Rotation/Slerp: Cartesian orientation updates and path interpolation.
- FrankaIK: Cartesian-to-joint conversion and forward-kinematics checks.
- pyzed/OpenCV: dual-camera observations and video recording.
- PyTorch/NumPy/PyYAML: inference input, numeric data and configuration.
- DuetDINOClient: model inference server communication.
- RobotiqGripper: gripper feedback and control.

Safety:
No automatic robot error recovery. The controller checks Cartesian action
bounds, FK consistency, IK validity, joint limits, joint displacement and
Ruckig path deviation. These checks do NOT provide collision avoidance.
"""

import ctypes
import json
import multiprocessing as mp
import shutil
import time
import traceback
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pylibfranka import ControllerMode, JointPositions, Robot
from ruckig import InputParameter, OutputParameter, Ruckig, Trajectory
from scipy.spatial.transform import Rotation, Slerp

from cameras.zed_camera import DualZEDCameras
from gripper.robotiq_gripper import RobotiqGripper
from inference.duetdino_client import DuetDINOClient
from robot.test_cartesian_IK import FrankaIK


# Configuration
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "robot" / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

FRANKA_CONFIG = CONFIG["franka"]
GRIPPER_CONFIG = CONFIG["gripper"]
DUET_CONFIG = CONFIG["duet_dino"]
RUNTIME_CONFIG = CONFIG.get("runtime", {})
EXPERIMENT_CONFIG = CONFIG.get("experiment", {})

EXPERIMENT_NAME = EXPERIMENT_CONFIG.get("name", "duet_dino")
EXPERIMENT_ROOT = Path(EXPERIMENT_CONFIG.get("root_dir", ROOT / "robot" / "experiments"))
if not EXPERIMENT_ROOT.is_absolute():
    EXPERIMENT_ROOT = ROOT / EXPERIMENT_ROOT
INSTRUCTION = DUET_CONFIG.get("instruction", CONFIG.get("task", {}).get("instruction", "goal"))

ROBOT_IP = FRANKA_CONFIG["robot_ip"]
JOINT_LOWER_LIMIT = np.asarray(FRANKA_CONFIG["joint_lower_limit"], dtype=np.float64)
JOINT_UPPER_LIMIT = np.asarray(FRANKA_CONFIG["joint_upper_limit"], dtype=np.float64)
MAX_JOINT_VELOCITY = [float(FRANKA_CONFIG["max_joint_velocity"])] * 7
MAX_JOINT_ACCELERATION = [float(FRANKA_CONFIG["max_joint_acceleration"])] * 7
MAX_JOINT_JERK = [float(FRANKA_CONFIG["max_joint_jerk"])] * 7

MAX_VLA_STEPS = int(DUET_CONFIG["max_steps"])
MAX_CONSECUTIVE_REJECTIONS = int(RUNTIME_CONFIG.get("max_consecutive_rejections", 3))
MAX_TRANSLATION_PER_ACTION = float(DUET_CONFIG["max_translation_per_action"])
MAX_ROTATION_PER_ACTION = float(DUET_CONFIG["max_rotation_per_action_rad"])
MAX_JOINT_TARGET_DELTA = float(DUET_CONFIG["max_joint_target_delta_rad"])
MAX_PATH_DEVIATION = float(DUET_CONFIG["max_path_deviation_m"])
MAX_PATH_ROTATION_DEVIATION = float(DUET_CONFIG["max_path_rotation_deviation_rad"])

ENABLE_GRIPPER = bool(GRIPPER_CONFIG.get("enabled", True))
GRIPPER_SPEED = int(GRIPPER_CONFIG["speed"])
GRIPPER_FORCE = int(GRIPPER_CONFIG["force"])
DRY_RUN = bool(RUNTIME_CONFIG.get("dry_run", False))
SAVE_OBSERVATIONS = bool(DUET_CONFIG.get("save_observations", True))
SAVE_VIDEOS = bool(DUET_CONFIG.get("save_videos", True))
VIDEO_FPS = float(DUET_CONFIG.get("video_fps", 2.0))


# Experiment management

def create_experiment_dir():
    """Create an iteration directory and snapshot the active configuration."""
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    iteration = 1
    while (EXPERIMENT_ROOT / f"iteration_{iteration}").exists():
        iteration += 1
    experiment_dir = EXPERIMENT_ROOT / f"iteration_{iteration}"
    experiment_dir.mkdir()
    (experiment_dir / "images").mkdir()
    (experiment_dir / "videos").mkdir()
    shutil.copy2(CONFIG_PATH, experiment_dir / "config.yaml")
    return experiment_dir


class ModelTrajectoryRejected(RuntimeError):
    """Raised when a Duet-DINO action or its planned motion is invalid."""


# Shared-memory helpers: odd/even version counters avoid partially read vectors.

def write_shared_vector(shared_array, version_counter, values):
    values = np.asarray(values, dtype=np.float64)
    version_counter.value += 1
    for index in range(7):
        shared_array[index] = float(values[index])
    version_counter.value += 1


def read_shared_vector(shared_array, version_counter):
    while True:
        version_before = version_counter.value
        if version_before % 2:
            continue
        values = np.array([shared_array[index] for index in range(7)], dtype=np.float64)
        version_after = version_counter.value
        if version_before == version_after and version_after % 2 == 0:
            return values


def read_shared_vector_if_updated(shared_array, version_counter, previous_version, previous_values):
    current_version = version_counter.value
    if current_version == previous_version or current_version % 2:
        return previous_values, previous_version
    values = np.array([shared_array[index] for index in range(7)], dtype=np.float64)
    version_after = version_counter.value
    if current_version == version_after and version_after % 2 == 0:
        return values, version_after
    return previous_values, previous_version


# Independent Franka process: deliberately no automatic error recovery.

def franka_control_process(shared_target, target_version, shared_state, state_version,
                           completed_version, state_ready, stop_event, error_queue):
    robot = control = None
    try:
        robot = Robot(ROBOT_IP)
        control = robot.start_joint_position_control(ControllerMode.JointImpedance)
        robot_state, _ = control.readOnce()
        measured_joints = np.asarray(robot_state.q, dtype=np.float64)
        write_shared_vector(shared_state, state_version, measured_joints)

        trajectory_generator = Ruckig(7, 0.001)
        ruckig_input, ruckig_output = InputParameter(7), OutputParameter(7)
        ruckig_input.current_position = measured_joints.tolist()
        ruckig_input.current_velocity = [0.0] * 7
        ruckig_input.current_acceleration = [0.0] * 7
        ruckig_input.target_position = measured_joints.tolist()
        ruckig_input.target_velocity = [0.0] * 7
        ruckig_input.target_acceleration = [0.0] * 7
        ruckig_input.max_velocity = MAX_JOINT_VELOCITY
        ruckig_input.max_acceleration = MAX_JOINT_ACCELERATION
        ruckig_input.max_jerk = MAX_JOINT_JERK

        target_joints = measured_joints.copy()
        last_target_version = target_version.value
        settled_cycles = 0
        control.writeOnce(JointPositions(measured_joints.tolist()))
        state_ready.set()
        print("Franka control ready.", flush=True)

        while not stop_event.is_set():
            robot_state, _ = control.readOnce()
            measured_joints = np.asarray(robot_state.q, dtype=np.float64)
            write_shared_vector(shared_state, state_version, measured_joints)
            new_target, new_version = read_shared_vector_if_updated(
                shared_target, target_version, last_target_version, target_joints
            )
            if new_version != last_target_version:
                target_joints, last_target_version = new_target.copy(), new_version
                settled_cycles = 0
                if (not np.all(np.isfinite(target_joints)) or
                    np.any(target_joints < JOINT_LOWER_LIMIT) or
                    np.any(target_joints > JOINT_UPPER_LIMIT)):
                    raise RuntimeError("Franka target is nonfinite or outside joint limits.")
                ruckig_input.target_position = target_joints.tolist()
                ruckig_input.target_velocity = [0.0] * 7
                ruckig_input.target_acceleration = [0.0] * 7

            result = trajectory_generator.update(ruckig_input, ruckig_output)
            if result < 0:
                raise RuntimeError(f"Ruckig trajectory generation failed: {result}")
            command_joints = np.asarray(ruckig_output.new_position, dtype=np.float64)
            if (not np.all(np.isfinite(command_joints)) or
                np.any(command_joints < JOINT_LOWER_LIMIT) or
                np.any(command_joints > JOINT_UPPER_LIMIT)):
                raise RuntimeError("Ruckig command violates Franka joint limits.")
            control.writeOnce(JointPositions(command_joints.tolist()))
            ruckig_output.pass_to_input(ruckig_input)

            # Acknowledge only after Ruckig finishes AND measured joints settle.
            # Requiring consecutive cycles avoids acknowledging transient proximity.
            if (result == 1 and
                np.max(np.abs(measured_joints - target_joints)) < 0.005 and
                np.max(np.abs(np.asarray(robot_state.dq, dtype=np.float64))) < 0.02):
                settled_cycles += 1
                if settled_cycles >= 100:
                    completed_version.value = last_target_version
            else:
                settled_cycles = 0

    except Exception:
        try:
            error_queue.put_nowait(traceback.format_exc())
        except Exception:
            pass
        stop_event.set()
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if control is not None:
            del control


# Cartesian conversion and safety checks

def read_franka_pose(robot):
    robot_state = robot.read_once()
    return (np.asarray(robot_state.q, dtype=np.float64),
            np.asarray(robot_state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F"))


def wrap_euler_angles(angles):
    """Wrap Euler observations to [-pi, pi]; do not alias orientations modulo pi."""
    return (np.asarray(angles, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def model_pose(transform, gripper_position):
    """Absolute base-frame EE pose (xyz, roll, pitch, yaw) and absolute grip [0, 1]."""
    euler_angles = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    
    return np.concatenate((transform[:3, 3], euler_angles,
                           [np.clip(gripper_position, 0.0, 1.0)])).astype(np.float32)


def target_from_action(current_pose, gripper_position, predicted_action):
    """Apply Duet-DINO base-frame XYZ and rotvec deltas to the current EE pose."""
    predicted_action = np.asarray(predicted_action, dtype=np.float64)
    if predicted_action.shape != (7,) or not np.all(np.isfinite(predicted_action)):
        raise ModelTrajectoryRejected(f"Invalid Duet-DINO action: {predicted_action}")
    if np.max(np.abs(predicted_action[:3])) > MAX_TRANSLATION_PER_ACTION:
        raise ModelTrajectoryRejected("Translation exceeds per-axis limit.")

    # Server converts internal Euler XYZ deltas to rotation vectors before sending.
    delta_rotation_angle = float(np.linalg.norm(predicted_action[3:6]))
    if delta_rotation_angle > MAX_ROTATION_PER_ACTION:
        raise ModelTrajectoryRejected("Rotation exceeds per-action limit.")

    desired_pose = current_pose.copy()
    desired_pose[:3, 3] += predicted_action[:3]
    desired_pose[:3, :3] = (
        Rotation.from_rotvec(predicted_action[3:6]).as_matrix()
        @ current_pose[:3, :3]
    )
    # Apply the predicted rotation-vector delta.
    # target_rotation = (
    #     Rotation.from_rotvec(predicted_action[3:6])
    #     * Rotation.from_matrix(current_pose[:3, :3])
    # )

    # # Convert to Euler XYZ and wrap each angle to [-pi/2, pi/2).
    # target_euler = target_rotation.as_euler("xyz")
    # target_euler = (target_euler + np.pi / 2) % np.pi - np.pi / 2

    # desired_pose[:3, :3] = Rotation.from_euler("xyz", target_euler).as_matrix()

    # Model's seventh action component is a relative normalized gripper delta.
    desired_gripper = float(np.clip(gripper_position + predicted_action[6], 0.0, 1.0))
    return desired_pose, desired_gripper


def validate_path(ik, current_joints, target_joints, current_pose, desired_pose):
    """Sample the Ruckig joint path and check Cartesian deviation; NOT collision detection."""
    trajectory_generator = Ruckig(7, 0.001)
    ruckig_input, ruckig_output = InputParameter(7), OutputParameter(7)
    ruckig_input.current_position = current_joints.tolist()
    ruckig_input.target_position = target_joints.tolist()
    ruckig_input.current_velocity = [0.0] * 7
    ruckig_input.current_acceleration = [0.0] * 7
    ruckig_input.target_velocity = [0.0] * 7
    ruckig_input.target_acceleration = [0.0] * 7
    ruckig_input.max_velocity = MAX_JOINT_VELOCITY
    ruckig_input.max_acceleration = MAX_JOINT_ACCELERATION
    ruckig_input.max_jerk = MAX_JOINT_JERK

    segment = desired_pose[:3, 3] - current_pose[:3, 3]
    distance_squared = float(segment @ segment)
    rotations = Rotation.from_matrix(np.stack([current_pose[:3, :3], desired_pose[:3, :3]]))
    rotation_interpolator = Slerp([0.0, 1.0], rotations)
    max_position_error = max_rotation_error = 0.0

    # This is a stationary-start path check; it is not a guarantee for mid-motion replanning.
    # Derive sample count from the planned duration, not a YAML timeout.
    planned_trajectory = Trajectory(7)
    if trajectory_generator.calculate(ruckig_input, planned_trajectory) < 0:
        raise ModelTrajectoryRejected("Ruckig could not calculate the joint trajectory.")
    for _ in range(max(1, int(np.ceil(planned_trajectory.duration / 0.001)) + 2)):
        result = trajectory_generator.update(ruckig_input, ruckig_output)
        if result < 0:
            raise ModelTrajectoryRejected(f"Trajectory planning failed: {result}")
        sampled_joints = np.asarray(ruckig_output.new_position, dtype=np.float64)
        if (not np.all(np.isfinite(sampled_joints)) or
            np.any(sampled_joints < JOINT_LOWER_LIMIT) or
            np.any(sampled_joints > JOINT_UPPER_LIMIT)):
            raise ModelTrajectoryRejected("Planned trajectory exceeds joint limits.")

        sampled_position, sampled_rotation = ik.forward_kinematics(sampled_joints)
        if distance_squared > 1e-12:
            progress = float(np.clip(
                (sampled_position - current_pose[:3, 3]) @ segment / distance_squared, 0.0, 1.0
            ))
        else:
            progress = float(np.clip(
                np.linalg.norm(sampled_joints - current_joints) /
                max(np.linalg.norm(target_joints - current_joints), 1e-9), 0.0, 1.0
            ))
        expected_position = current_pose[:3, 3] + progress * segment
        expected_rotation = rotation_interpolator([progress]).as_matrix()[0]
        max_position_error = max(max_position_error,
                                 float(np.linalg.norm(sampled_position - expected_position)))
        max_rotation_error = max(max_rotation_error, float(np.linalg.norm(
            Rotation.from_matrix(expected_rotation @ sampled_rotation.T).as_rotvec()
        )))
        ruckig_output.pass_to_input(ruckig_input)
        if result == 1:
            break
    else:
        raise ModelTrajectoryRejected("Ruckig trajectory did not complete as calculated.")

    print(f"Planned path deviation: {max_position_error * 1000:.2f} mm, "
          f"rotation deviation: {np.rad2deg(max_rotation_error):.2f} deg")
    if max_position_error > MAX_PATH_DEVIATION:
        raise ModelTrajectoryRejected(f"Cartesian path deviation: {max_position_error:.4f} m")
    if max_rotation_error > MAX_PATH_ROTATION_DEVIATION:
        raise ModelTrajectoryRejected(f"Orientation path deviation: {max_rotation_error:.4f} rad")
    return max_position_error, max_rotation_error


def as_tensor(rgb_frame):
    return torch.from_numpy(np.array(rgb_frame, dtype=np.uint8, copy=True, order="C"))


class VLAController:
    """Coordinate goal images, Duet-DINO inference, Cartesian IK and Franka motion."""

    def __init__(self, shared_target, target_version, shared_state, state_version,
                 stop_event, error_queue, experiment_dir, robot=None, franka_process=None,
                 completed_version=None):
        self.shared_target = shared_target
        self.target_version = target_version
        self.completed_version = completed_version
        self.shared_state = shared_state
        self.state_version = state_version
        self.stop_event = stop_event
        self.error_queue = error_queue
        self.experiment_dir = experiment_dir
        self.robot, self.franka_process = robot, franka_process
        self.cameras = self.gripper = self.client = None
        self.video_writers = {}
        self.motion_pending = False
        self.ik = FrankaIK(CONFIG, max_joint_change=MAX_JOINT_TARGET_DELTA)
        self.log_path = experiment_dir / "robot_run.txt"
        self.jsonl_path = experiment_dir / "robot_run.jsonl"
        self.current_vla_state = None

    def initialize(self):
        self.cameras = DualZEDCameras()
        self.gripper = RobotiqGripper(port=GRIPPER_CONFIG.get("port", "/dev/ttyUSB0"))
        if not DRY_RUN and GRIPPER_CONFIG.get("activate_on_start", False):
            self.gripper.activate()
        self.client = DuetDINOClient(
            host=DUET_CONFIG.get("host", "127.0.0.1"),
            port=int(DUET_CONFIG.get("port", 8000)),
            timeout=float(DUET_CONFIG.get("timeout_seconds", 120)),
        )
        self.client.connect()
        goal_dir = ROOT / "robot" / "goal_images"
        external_goal = np.asarray(Image.open(goal_dir / "external_goal.png").convert("RGB"))
        wrist_goal = np.asarray(Image.open(goal_dir / "wrist_goal.png").convert("RGB"))
        self.client.set_goal_images(as_tensor(external_goal), as_tensor(wrist_goal),
                                    env_id=0, instruction=INSTRUCTION)

    def check_franka_error(self):
        if self.stop_event.is_set() or (
            self.franka_process is not None and not self.franka_process.is_alive()
        ):
            try:
                error_text = self.error_queue.get_nowait()
            except Empty:
                error_text = "Franka control process stopped unexpectedly."
            raise RuntimeError(error_text)

    def get_joint_state(self):
        if DRY_RUN:
            return read_franka_pose(self.robot)
        current_joints = read_shared_vector(self.shared_state, self.state_version)
        position, rotation = self.ik.forward_kinematics(current_joints)
        current_pose = np.eye(4)
        current_pose[:3, :3], current_pose[:3, 3] = rotation, position
        return current_joints, current_pose

    def get_observation(self):
        current_joints, current_pose = self.get_joint_state()
        gripper_position = float(np.clip(self.gripper.get_position_normalized(), 0.0, 1.0))
        external_rgb, wrist_rgb = self.cameras.get_frames_rgb()
        #convert the images sendign to the model each time into the video writer
        self._write_rgb_to_video("external", external_rgb)
        self._write_rgb_to_video("wrist", wrist_rgb)
        return external_rgb, wrist_rgb, current_joints, current_pose, gripper_position

    def _write_rgb_to_video(self, camera_name, rgb_frame):
        if not SAVE_VIDEOS or rgb_frame is None:
            return
        writer = self.video_writers.get(camera_name)
        if writer is None:
            height, width = rgb_frame.shape[:2]
            video_path = self.experiment_dir / "videos" / f"{camera_name}_trajectory.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     VIDEO_FPS, (width, height), True)
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {video_path}")
            self.video_writers[camera_name] = writer
        writer.write(cv2.cvtColor(np.asarray(rgb_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    def save_observation(self, step, external_rgb, wrist_rgb):
        if SAVE_OBSERVATIONS:
            Image.fromarray(external_rgb).save(self.experiment_dir / "images" /
                                               f"{step:04d}_external.png")
            Image.fromarray(wrist_rgb).save(self.experiment_dir / "images" /
                                            f"{step:04d}_wrist.png")
        self._write_rgb_to_video("external", external_rgb)
        self._write_rgb_to_video("wrist", wrist_rgb)

    def get_model_state(self, current_pose, gripper_position):
        return model_pose(current_pose, gripper_position)

    def write_log(self, text):
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(text + "\n")

    def infer(self, external_rgb, wrist_rgb, current_pose, gripper_position):
        observation = self.get_model_state(current_pose, gripper_position)
        self.current_vla_state = observation.copy()
        predicted_action, latency_ms = self.client.predict(
            external_image=as_tensor(external_rgb), wrist_image=as_tensor(wrist_rgb),
            ee_pose=observation, instruction=INSTRUCTION, env_id=0,
        )
        return np.asarray(predicted_action, dtype=np.float64), float(latency_ms), observation

    def validate_action_trajectory(self, current_joints, current_pose, gripper_position, predicted_action):
        desired_pose, desired_gripper = target_from_action(
            current_pose, gripper_position, predicted_action
        )
        fk_position, fk_rotation = self.ik.forward_kinematics(current_joints)
        position_error = np.linalg.norm(fk_position - current_pose[:3, 3])
        rotation_error = np.linalg.norm(Rotation.from_matrix(
            fk_rotation @ current_pose[:3, :3].T
        ).as_rotvec())
        if position_error > 0.002 or rotation_error > np.deg2rad(2.0):
            raise ModelTrajectoryRejected("Franka/MuJoCo forward-kinematics mismatch.")

        try:
            target_joints = np.asarray(self.ik.solve(
                current_joints, desired_pose[:3, 3], desired_pose[:3, :3]
            ), dtype=np.float64)
        except RuntimeError as error:
            # IK feasibility failures are rejected predictions, not controller faults.
            raise ModelTrajectoryRejected(f"IK rejected target: {error}") from error
        if target_joints.shape != (7,) or not np.all(np.isfinite(target_joints)):
            raise ModelTrajectoryRejected("IK returned an invalid joint target.")
        if np.any(target_joints < JOINT_LOWER_LIMIT) or np.any(target_joints > JOINT_UPPER_LIMIT):
            raise ModelTrajectoryRejected("IK target exceeds Franka joint limits.")
        if np.max(np.abs(target_joints - current_joints)) > MAX_JOINT_TARGET_DELTA:
            raise ModelTrajectoryRejected("Joint target displacement exceeds configured limit.")

        path_position_error, path_rotation_error = validate_path(
            self.ik, current_joints, target_joints, current_pose, desired_pose
        )
        return (target_joints, desired_gripper, desired_pose,
                path_position_error, path_rotation_error)

    def set_joint_target(self, target_joints):
        target_joints = np.asarray(target_joints, dtype=np.float64)
        if (target_joints.shape != (7,) or not np.all(np.isfinite(target_joints)) or
            np.any(target_joints < JOINT_LOWER_LIMIT) or
            np.any(target_joints > JOINT_UPPER_LIMIT)):
            raise ModelTrajectoryRejected("Invalid Franka joint target.")
        write_shared_vector(self.shared_target, self.target_version, target_joints)

    def execute_gripper_action(self, gripper_position):
        if ENABLE_GRIPPER:
            gripper_value = int(round(float(np.clip(gripper_position, 0.0, 1.0)) * 255))
            self.gripper.set_position(gripper_value, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)

    def execute_action(self, target_joints, desired_gripper):
        if DRY_RUN:
            print("DRY RUN: no robot/gripper motion")
            return
        self.check_franka_error()
        self.set_joint_target(target_joints)
        self.motion_pending = True
        # Command the gripper immediately, without waiting for arm completion.
        self.execute_gripper_action(desired_gripper)

        # Wait for the Franka process to acknowledge this exact target version.
        # This prevents the next camera observation/inference until the arm settles.
        requested_version = self.target_version.value
        timeout = float(RUNTIME_CONFIG.get("motion_timeout_seconds", 30.0))
        deadline = time.monotonic() + timeout
        while self.completed_version.value < requested_version:
            self.check_franka_error()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Franka did not settle within {timeout:.1f} seconds.")
            time.sleep(0.01)
        self.motion_pending = False

    def run(self):
        print(f"Running '{EXPERIMENT_NAME}' | action frame: base | instruction: {INSTRUCTION} | "
              f"max steps: {MAX_VLA_STEPS} | DRY_RUN={DRY_RUN}", flush=True)
        consecutive_rejections = 0
        with self.jsonl_path.open("w", encoding="utf-8") as log:
            for step in range(1, MAX_VLA_STEPS + 1):
                if not DRY_RUN:
                    self.check_franka_error()
                external_rgb, wrist_rgb, current_joints, current_pose, gripper_position = (
                    self.get_observation()
                )
                predicted_action, latency_ms, observation = self.infer(
                    external_rgb, wrist_rgb, current_pose, gripper_position
                )
                print(f"Duet-DINO action: {np.round(predicted_action, 6)}")
                print(f"Translation magnitude: {np.linalg.norm(predicted_action[:3]) * 100:.2f} cm")
                print(f"Maximum axis displacement: "
                      f"{np.max(np.abs(predicted_action[:3])) * 100:.2f} cm")

                # Save even rejected observations for experiment analysis.
                self.save_observation(step, external_rgb, wrist_rgb)
                try:
                    (target_joints, desired_gripper, desired_pose,
                     path_position_error, path_rotation_error) = (
                        self.validate_action_trajectory(current_joints, current_pose,
                                                        gripper_position, predicted_action)
                    )
                except ModelTrajectoryRejected as error:
                    consecutive_rejections += 1
                    reason = str(error)
                    message = (f"Step {step}: action rejected ({consecutive_rejections}/"
                               f"{MAX_CONSECUTIVE_REJECTIONS} consecutive): {reason}")
                    print(message, flush=True)
                    self.write_log(message)
                    log.write(json.dumps(dict(
                        step=step, status="rejected", reason=reason,
                        state=observation.tolist(), action=predicted_action.tolist(),
                        latency_ms=latency_ms, dry_run=DRY_RUN,
                    )) + "\n")
                    log.flush()
                    if consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
                        print("Consecutive rejection limit reached; stopping episode.", flush=True)
                        break
                    continue
                consecutive_rejections = 0
                record = dict(
                    step=step, state=observation.tolist(), action=predicted_action.tolist(),
                    target_q=target_joints.tolist(), target_gripper=desired_gripper,
                    latency_ms=latency_ms, path_error_m=path_position_error,
                    path_rotation_error_rad=path_rotation_error, dry_run=DRY_RUN,
                )
                self.write_log("=" * 80)
                self.write_log(f"VLA STEP = {step}")
                self.write_log("Robot current state = " + np.array2string(
                    self.current_vla_state, precision=6, separator=", "))
                predicted_absolute_state = self.get_model_state(desired_pose, desired_gripper)
                print("Robot current absolute state before action "
                      "[x, y, z, roll, pitch, yaw, gripper] = "
                      f"{np.round(self.current_vla_state, 6)}", flush=True)
                print("Model-predicted absolute target state "
                      "[x, y, z, roll, pitch, yaw, gripper] = "
                      f"{np.round(predicted_absolute_state, 6)}", flush=True)
                self.write_log("Predicted Cartesian action = " + np.array2string(
                    predicted_action, precision=6, separator=", "))
                self.write_log("Predicted absolute target state = " + np.array2string(
                    predicted_absolute_state, precision=6, separator=", "))
                self.write_log("IK target joints = " + np.array2string(
                    target_joints, precision=6, separator=", "))
                self.write_log(f"Gripper target = {desired_gripper:.6f}")
                self.write_log(f"Inference latency = {latency_ms:.2f} ms")
                self.write_log(f"Cartesian path deviation = {path_position_error:.6f} m")
                self.write_log(f"Rotation path deviation = {path_rotation_error:.6f} rad")
                self.write_log("")
                print(f"Step {step}: action={np.round(predicted_action, 4)} | "
                      f"target_q={np.round(target_joints, 4)}", flush=True)
                self.execute_action(target_joints, desired_gripper)
                actual_joints, actual_pose = self.get_joint_state()
                actual_gripper = float(np.clip(
                    self.gripper.get_position_normalized(), 0.0, 1.0
                ))
                actual_state = self.get_model_state(actual_pose, actual_gripper)
                position_error = float(np.linalg.norm(
                    desired_pose[:3, 3] - actual_pose[:3, 3]
                ))
                orientation_error = float(np.linalg.norm(Rotation.from_matrix(
                    desired_pose[:3, :3] @ actual_pose[:3, :3].T
                ).as_rotvec()))
                state_difference = actual_state.astype(np.float64) - predicted_absolute_state
                state_difference[3:6] = wrap_euler_angles(state_difference[3:6])
                print("Robot measured absolute state after action "
                      "[x, y, z, roll, pitch, yaw, gripper] = "
                      f"{np.round(actual_state, 6)}", flush=True)
                print("Executed minus predicted state "
                      "[x, y, z, roll, pitch, yaw, gripper] = "
                      f"{np.round(state_difference, 6)}", flush=True)
                print(f"Robot measured joints after action = "
                      f"{np.round(actual_joints, 6)}", flush=True)
                print(f"Pose check: position error={position_error * 1000:.2f} mm, "
                      f"orientation error={np.rad2deg(orientation_error):.2f} deg",
                      flush=True)
                self.write_log("Robot measured absolute state after action = " +
                               np.array2string(actual_state, precision=6, separator=", "))
                self.write_log("Executed minus predicted state = " +
                               np.array2string(state_difference, precision=6, separator=", "))
                self.write_log(f"Pose check = position {position_error:.6f} m, "
                               f"orientation {orientation_error:.6f} rad")
                record.update(
                    predicted_absolute_state=predicted_absolute_state.tolist(),
                    executed_absolute_state=actual_state.tolist(),
                    executed_joints=actual_joints.tolist(),
                    state_difference=state_difference.tolist(),
                    pose_error_position_m=position_error,
                    pose_error_orientation_rad=orientation_error,
                )
                log.write(json.dumps(record) + "\n")
                log.flush()

    def close(self):
        """Release video writers, cameras, inference connection and gripper."""
        for writer in self.video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        for resource, method in ((self.client, "close"),
                                 (self.cameras, "close"),
                                 (self.gripper, "close_connection")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as exception:
                    print(f"Shutdown warning: {exception}")


def main():
    mp.set_start_method("spawn", force=True)
    experiment_dir = create_experiment_dir()
    print(f"Experiment output: {experiment_dir} | DRY_RUN={DRY_RUN}", flush=True)

    shared_target = mp.RawArray(ctypes.c_double, 7)
    target_version = mp.RawValue(ctypes.c_uint64, 0)
    shared_state = mp.RawArray(ctypes.c_double, 7)
    state_version = mp.RawValue(ctypes.c_uint64, 0)
    completed_version = mp.RawValue(ctypes.c_uint64, 0)
    state_ready, stop_event = mp.Event(), mp.Event()
    error_queue = mp.Queue(maxsize=1)
    franka_process = controller = robot = None

    try:
        if DRY_RUN:
            robot = Robot(ROBOT_IP)
        else:
            print("WARNING: No table/object/self-collision checks. Confirm calibrated "
                  "frames, workspace clearance and tested controller behavior.")
            franka_process = mp.Process(
                target=franka_control_process,
                args=(shared_target, target_version, shared_state, state_version,
                      completed_version, state_ready, stop_event, error_queue),
                daemon=False,
            )
            franka_process.start()
            if not state_ready.wait(timeout=10.0) or stop_event.is_set():
                raise RuntimeError("Franka control process failed to initialize.")

        controller = VLAController(
            shared_target, target_version, shared_state, state_version,
            stop_event, error_queue, experiment_dir, robot, franka_process,
            completed_version,
        )
        controller.initialize()
        controller.run()
    except KeyboardInterrupt:
        print("Interrupted by operator; requesting Franka stop.", flush=True)
    finally:
        stop_event.set()
        if controller is not None:
            if controller.motion_pending:
                print("WARNING: stopping while a commanded trajectory may still be active.",
                      flush=True)
            controller.close()
        if franka_process is not None:
            franka_process.join(timeout=5.0)
            if franka_process.is_alive():
                franka_process.terminate()
                franka_process.join(timeout=2.0)
        if robot is not None:
            del robot
        try:
            while True:
                print(f"Franka process error:\n{error_queue.get_nowait()}")
        except Empty:
            pass
        print(f"Experiment output: {experiment_dir}", flush=True)


if __name__ == "__main__":
    main()