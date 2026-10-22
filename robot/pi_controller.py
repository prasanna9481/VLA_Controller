#!/usr/bin/env python3
"""Pi05-DROID Franka controller with separate Ruckig control process.

Motion commands are permitted only after explicit authorization and safety checks.
The gripper is read for observation but not actuated until mapping is verified.
"""

import ctypes
import multiprocessing as mp
import os
import shutil
import time
from queue import Empty
import numpy as np
import yaml
from pylibfranka import ControllerMode, JointPositions, Robot
from ruckig import InputParameter, OutputParameter, Ruckig

from cameras.zed_camera import DualZEDCameras
from gripper.robotiq_gripper import RobotiqGripper
from inference.pi05_client import Pi05Client


# Configuration
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

EXPERIMENT_NAME = CONFIG["experiment"]["name"]
EXPERIMENT_ROOT = CONFIG["experiment"]["root_dir"]
INSTRUCTION = CONFIG["task"]["instruction"]

ROBOT_IP = CONFIG["franka"]["robot_ip"]
JOINT_LOWER_LIMIT = np.asarray(CONFIG["franka"]["joint_lower_limit"], dtype=np.float64)
JOINT_UPPER_LIMIT = np.asarray(CONFIG["franka"]["joint_upper_limit"], dtype=np.float64)
MAX_JOINT_VELOCITY = np.full(7, float(CONFIG["franka"]["max_joint_velocity"]), dtype=np.float64)
MAX_JOINT_ACCELERATION = np.full(7, float(CONFIG["franka"]["max_joint_acceleration"]), dtype=np.float64)
MAX_JOINT_JERK = np.full(7, float(CONFIG["franka"]["max_joint_jerk"]), dtype=np.float64)

PI_SERVER = CONFIG["pi05"]["server_url"]
MAX_VLA_STEPS = int(CONFIG["pi05"]["max_vla_steps"])
ACTIONS_PER_CHUNK = int(CONFIG["pi05"]["actions_per_chunk"])
ACTION_RATE_HZ = float(CONFIG["pi05"]["action_rate_hz"])
if ACTION_RATE_HZ != 15.0:
    raise ValueError("The DROID reference policy runs at 15 Hz")
ACTION_DT = 1.0 / ACTION_RATE_HZ
DROID_JOINT_DELTA_SCALE_RAD = 0.2

ENABLE_GRIPPER = bool(CONFIG["gripper"]["enabled"])
MAX_JOINT_DISPLACEMENT_RAD = float(CONFIG["safety"]["max_action_joint_displacement_rad"])


def create_experiment_dir():
    """Create a new experiment iteration directory and save its configuration."""

    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

    iteration = 1
    while os.path.exists(os.path.join(EXPERIMENT_ROOT, f"iteration_{iteration}")):
        iteration += 1

    experiment_dir = os.path.join(EXPERIMENT_ROOT, f"iteration_{iteration}")

    os.makedirs(experiment_dir, exist_ok=False)
    shutil.copy2(CONFIG_PATH, os.path.join(experiment_dir, "config.yaml"))

    return experiment_dir


class ModelTrajectoryRejected(RuntimeError):
    """Raised when a model trajectory is unsafe or malformed."""


def convert_droid_action_to_joint_delta(action):
    """DROID joint_velocity_to_delta mapping for 7 action dimensions.

    DROID first clips the policy action to [-1, 1]. Its IK solver then
    scales any vector with infinity norm > 1 and multiplies by 0.2.
    The result is a joint POSITION INCREMENT, not a velocity in rad/s.
    """
    velocity_action = np.asarray(action, dtype=np.float64)
    if velocity_action.shape != (7,) or not np.all(np.isfinite(velocity_action)):
        raise ModelTrajectoryRejected("Expected seven finite DROID joint actions")
    velocity_action = np.clip(velocity_action, -1.0, 1.0)
    max_norm = float(np.max(np.abs(velocity_action)))
    if max_norm > 1.0:
        velocity_action /= max_norm
    return velocity_action * DROID_JOINT_DELTA_SCALE_RAD


def validate_joint_target(target, measured):
    target = np.asarray(target, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if target.shape != (7,) or measured.shape != (7,) or not np.all(np.isfinite(target)):
        raise ModelTrajectoryRejected("Invalid joint target")
    if np.any(target < JOINT_LOWER_LIMIT) or np.any(target > JOINT_UPPER_LIMIT):
        raise ModelTrajectoryRejected("Joint target exceeds configured Franka position limits")
    displacement = float(np.max(np.abs(target - measured)))
    if displacement > MAX_JOINT_DISPLACEMENT_RAD:
        raise ModelTrajectoryRejected(
            f"Joint displacement {displacement:.4f} rad exceeds {MAX_JOINT_DISPLACEMENT_RAD:.4f} rad limit"
        )
    return displacement


# Shared-memory helpers use an odd/even version counter to avoid partial reads.
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

        values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
        version_after = version_counter.value

        if version_before == version_after and version_after % 2 == 0:
            return values


def read_shared_vector_if_updated(
    shared_array, version_counter, previous_version, previous_values
):
    current_version = version_counter.value
    if current_version == previous_version or current_version % 2:
        return previous_values, previous_version

    values = np.array([shared_array[i] for i in range(7)], dtype=np.float64)
    version_after = version_counter.value

    if current_version == version_after and version_after % 2 == 0:
        return values, version_after

    return previous_values, previous_version


def franka_control_process(
    shared_target,
    target_version,
    shared_state,
    state_version,
    state_ready,
    stop_event,
    error_queue,
):
    """Run Franka joint control in a separate process with Ruckig smoothing."""
    robot = None
    control = None

    try:
        robot = Robot(ROBOT_IP)

        while not stop_event.is_set():
            try:
                control = robot.start_joint_position_control(ControllerMode.JointImpedance)

                robot_state, _ = control.readOnce()
                measured_joints = np.asarray(robot_state.q, dtype=np.float64)
                write_shared_vector(shared_state, state_version, measured_joints)

                # Franka runs at roughly 1 kHz, so Ruckig is configured for 1 ms steps.
                trajectory_generator = Ruckig(7, 0.001)
                ruckig_input = InputParameter(7)
                ruckig_output = OutputParameter(7)

                ruckig_input.current_position = measured_joints.tolist()
                ruckig_input.current_velocity = [0.0] * 7
                ruckig_input.current_acceleration = [0.0] * 7
                ruckig_input.target_position = measured_joints.tolist()
                ruckig_input.target_velocity = [0.0] * 7
                ruckig_input.target_acceleration = [0.0] * 7
                ruckig_input.max_velocity = MAX_JOINT_VELOCITY.tolist()
                ruckig_input.max_acceleration = MAX_JOINT_ACCELERATION.tolist()
                ruckig_input.max_jerk = MAX_JOINT_JERK.tolist()

                target_joints = measured_joints.copy()
                last_target_version = target_version.value

                # Start from the measured pose to avoid a discontinuity at controller startup.
                control.writeOnce(JointPositions(measured_joints.tolist()))
                state_ready.set()
                print("Franka control ready.", flush=True)

                while not stop_event.is_set():
                    robot_state, _ = control.readOnce()
                    measured_joints = np.asarray(robot_state.q, dtype=np.float64)
                    write_shared_vector(shared_state, state_version, measured_joints)

                    new_target, new_version = read_shared_vector_if_updated(
                        shared_target,
                        target_version,
                        last_target_version,
                        target_joints,
                    )

                    if new_version != last_target_version:
                        target_joints = new_target.copy()
                        last_target_version = new_version

                        if np.any(target_joints < JOINT_LOWER_LIMIT) or np.any(
                            target_joints > JOINT_UPPER_LIMIT
                        ):
                            raise RuntimeError("π₀.₅ target exceeded Franka joint limits.")

                        # Only update the target. Ruckig keeps its current velocity and acceleration.
                        ruckig_input.target_position = target_joints.tolist()
                        ruckig_input.target_velocity = [0.0] * 7
                        ruckig_input.target_acceleration = [0.0] * 7

                    result = trajectory_generator.update(ruckig_input, ruckig_output)
                    if result < 0:
                        raise RuntimeError(
                            f"Ruckig trajectory generation failed: {result}"
                        )

                    command_joints = np.asarray(
                        ruckig_output.new_position, dtype=np.float64
                    )
                    if np.any(command_joints < JOINT_LOWER_LIMIT) or np.any(
                        command_joints > JOINT_UPPER_LIMIT
                    ):
                        raise RuntimeError(
                            "Ruckig generated a position outside Franka limits."
                        )

                    control.writeOnce(JointPositions(command_joints.tolist()))

                    # Carry the generated position, velocity, and acceleration forward.
                    ruckig_output.pass_to_input(ruckig_input)

                final_command = JointPositions(list(ruckig_input.current_position))
                final_command.motion_finished = True
                control.writeOnce(final_command)

            except Exception as exception:
                if stop_event.is_set():
                    break

                print(f"Franka control error: {exception}", flush=True)

                try:
                    robot.stop()
                except Exception:
                    pass

                if control is not None:
                    try:
                        del control
                    except Exception:
                        pass
                    control = None

                try:
                    error_queue.put_nowait(f"Franka motion fault: {exception!r}")
                except Exception:
                    pass
                stop_event.set()
                break
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass


class VLAController:
    """Coordinate cameras, π₀.₅ inference, Franka targets, and the gripper."""

    def __init__(
        self,
        shared_target,
        target_version,
        shared_state,
        state_version,
        stop_event,
        error_queue,
        experiment_dir,
    ):
        self.shared_target = shared_target
        self.target_version = target_version
        self.shared_state = shared_state
        self.state_version = state_version
        self.stop_event = stop_event
        self.error_queue = error_queue
        self.gripper = None

        self.experiment_dir = experiment_dir
        # self.image_dir = os.path.join(experiment_dir, "images")
        self.log_path = os.path.join(experiment_dir, "robot_run.txt")

        self.gripper = RobotiqGripper() if ENABLE_GRIPPER else None
        # Do not activate gripper: activation may move the fingers.
        self.cameras = DualZEDCameras()
        self.pi05 = Pi05Client(server_url=PI_SERVER)

        self.current_vla_state = None

    def write_log(self, text):
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(text + "\n")

    def get_joint_state(self):
        return read_shared_vector(
            self.shared_state, self.state_version
        ).astype(np.float32)

    def get_model_state(self):
        joint_state = self.get_joint_state()
        gripper_position = self.gripper.get_position_normalized() if self.gripper is not None else 0.0

        # binary gripper state
        if gripper_position <= 0.5:
            gripper_position = 0.0
        else:
            gripper_position = 1.0
        
        return np.concatenate(
            [joint_state, np.array([gripper_position], dtype=np.float32)]
        )

    def check_franka_error(self):
        if not self.stop_event.is_set():
            return

        try:
            error_text = self.error_queue.get_nowait()
        except Empty:
            error_text = "Franka control process stopped unexpectedly."

        raise RuntimeError(error_text)

    def get_fresh_joint_state(self, previous_version, timeout=None):
        """Wait for a newly published, consistent 7-joint measurement.

        A timeout is a fault, not permission to reuse stale feedback.
        """
        timeout = 1.0 if timeout is None else timeout
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self.check_franka_error()
            version_before = self.state_version.value
            if version_before % 2 or version_before == previous_version:
                time.sleep(0.001)
                continue
            measured = np.array([self.shared_state[i] for i in range(7)], dtype=np.float64)
            version_after = self.state_version.value
            if version_before == version_after and version_after % 2 == 0:
                if not np.all(np.isfinite(measured)):
                    raise ModelTrajectoryRejected("Non-finite Franka joint state")
                return measured, version_after
            time.sleep(0.001)
        raise ModelTrajectoryRejected(f"No fresh Franka state within {timeout * 1000:.0f} ms")

    def get_observation(self):
        side_rgb, wrist_rgb = self.cameras.get_frames_rgb()
        return side_rgb, wrist_rgb, self.get_model_state()

    def set_joint_target(self, target_joints):
        """Validate and publish a new seven-joint Franka target."""
        target_joints = np.asarray(target_joints, dtype=np.float64)

        if target_joints.shape != (7,):
            raise ValueError(
                f"Expected joint target shape (7,), got {target_joints.shape}"
            )
        if not np.all(np.isfinite(target_joints)):
            raise ValueError("Joint target contains NaN or Inf.")
        if np.any(target_joints < JOINT_LOWER_LIMIT) or np.any(
            target_joints > JOINT_UPPER_LIMIT
        ):
            raise ModelTrajectoryRejected(
                "π₀.₅ joint target is outside Franka joint limits."
            )

        write_shared_vector(
            self.shared_target, self.target_version, target_joints
        )

    def validate_action_trajectory(self, predicted_actions):
        """Validate raw π₀.₅ actions without interpreting them as joint positions."""
        predicted_actions = np.asarray(predicted_actions, dtype=np.float32)
        if predicted_actions.ndim != 2 or predicted_actions.shape[1] != 8 or len(predicted_actions) == 0:
            raise ModelTrajectoryRejected(f"Expected π₀.₅ action shape (N, 8), got {predicted_actions.shape}")
        if not np.all(np.isfinite(predicted_actions)):
            raise ModelTrajectoryRejected("π₀.₅ returned NaN or Inf.")
        return predicted_actions

    def infer(self, vla_step):
        """Capture the latest observation and request one action chunk."""
        self.check_franka_error()
        side_rgb, wrist_rgb, robot_state = self.get_observation()
        self.current_vla_state = robot_state.copy()

        predicted_actions, _ = self.pi05.predict(
            external_image=side_rgb,
            wrist_image=wrist_rgb,
            instruction=INSTRUCTION,
            robot_state=robot_state,
        )
        return self.validate_action_trajectory(predicted_actions)

    def execute_action_chunk(self, predicted_actions, vla_step):
        """DROID feedback-based action conversion, one measured pose per action."""
        actions = predicted_actions[:ACTIONS_PER_CHUNK]
        if len(actions) != ACTIONS_PER_CHUNK:
            raise ModelTrajectoryRejected("Incomplete DROID action chunk")
        self.write_log(f"VLA STEP {vla_step} | state={self.current_vla_state}")
        self.write_log("DROID actions:\n" + np.array2string(actions, threshold=np.inf))
        next_action_time = time.perf_counter()
        last_state_version = self.state_version.value
        for action_index, action in enumerate(actions, start=1):
            measured, last_state_version = self.get_fresh_joint_state(last_state_version)
            delta = convert_droid_action_to_joint_delta(action[:7])
            target = measured + delta
            validate_joint_target(target, measured)
            self.set_joint_target(target)
            # Gripper remains disabled until Robotiq/DROID open-close mapping is verified.
            self.write_log(f"action={action_index} measured={measured} delta={delta} target={target}")
            next_action_time += ACTION_DT
            remaining = next_action_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        print(f"VLA step {vla_step}: {len(actions)} DROID position targets published")

    def run(self):
        print(
            f"Running '{EXPERIMENT_NAME}' | instruction: {INSTRUCTION} | "
            f"{ACTIONS_PER_CHUNK} actions/chunk at {ACTION_RATE_HZ:.1f} Hz"
        )

        for vla_step in range(1, MAX_VLA_STEPS + 1):
            self.check_franka_error()

            predicted_actions = self.infer(vla_step)

            if vla_step == 1:
                print("FIRST CHUNK SAFETY CHECK: no actions executed.")
                self.write_log("FIRST CHUNK: raw actions\n" + np.array2string(predicted_actions, threshold=np.inf))
                continue
            self.execute_action_chunk(predicted_actions, vla_step)

    def close(self):
        """Release camera and gripper connections."""
        if self.cameras is not None:
            try:
                self.cameras.close()
            except Exception as exception:
                print(f"Camera shutdown warning: {exception}")

        if self.gripper is not None:
            try:
                self.gripper.close_connection()
            except Exception as exception:
                print(f"Gripper shutdown warning: {exception}")


def main():
    mp.set_start_method("spawn", force=True)
    experiment_dir = create_experiment_dir()

    # Main process publishes targets; Franka process publishes measured state.
    shared_target = mp.RawArray(ctypes.c_double, 7)
    target_version = mp.RawValue(ctypes.c_uint64, 0)
    shared_state = mp.RawArray(ctypes.c_double, 7)
    state_version = mp.RawValue(ctypes.c_uint64, 0)

    state_ready = mp.Event()
    stop_event = mp.Event()
    error_queue = mp.Queue(maxsize=1)

    franka_process = mp.Process(
        target=franka_control_process,
        args=(
            shared_target,
            target_version,
            shared_state,
            state_version,
            state_ready,
            stop_event,
            error_queue,
        ),
        daemon=False,
    )
    franka_process.start()

    if not state_ready.wait(timeout=10.0):
        stop_event.set()
        franka_process.join(timeout=2.0)

        if franka_process.is_alive():
            franka_process.terminate()
            franka_process.join(timeout=2.0)

        raise RuntimeError("Franka control process did not initialize.")

    controller = None

    try:
        controller = VLAController(
            shared_target,
            target_version,
            shared_state,
            state_version,
            stop_event,
            error_queue,
            experiment_dir,
        )

        print(
            f"Experiment output: {experiment_dir}\n"
            "WARNING: robot motion is enabled. Keep the Franka user-stop "
            "available and verify that the workspace is clear."
        )
        controller.run()

    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as exception:
        print(f"Controller error: {exception}")
        raise
    finally:
        stop_event.set()

        if controller is not None:
            controller.close()

        franka_process.join(timeout=5.0)
        if franka_process.is_alive():
            franka_process.terminate()
            franka_process.join(timeout=2.0)

        try:
            while True:
                print(f"Franka process error:\n{error_queue.get_nowait()}")
        except Empty:
            pass

        print("Shutdown complete.")


if __name__ == "__main__":
    main()