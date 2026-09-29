# MolmoAct2 Franka Controller

This repository uses the **MolmoAct2-DROID** vision-language-action model to control a Franka robot equipped with a Robotiq 2F-85 gripper and two ZED cameras.

The MolmoAct2 model runs behind an HTTP inference server. The robot controller captures camera observations, requests an action trajectory from the model, and executes the predicted trajectory in a closed loop.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd molmoact2
```

---

### 2. Install `uv`

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and dependency management.

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

Verify the installation:

```bash
uv --version
```

---

### 3. Create the robot-side environment

The robot-side environment uses Python 3.12.

From the repository root:

```bash
uv venv --python 3.12
```

Activate it:

```bash
source .venv/bin/activate
```

Verify the Python version:

```bash
python --version
```

It should report Python 3.12.

Install the robot-side dependencies:

```bash
uv pip install -r requirements.txt
```

---

## ZED SDK and PyZED setup

The ZED Python API is **not included in this repository**.

The correct PyZED wheel depends on:

- operating system
- CPU architecture
- Python version
- ZED SDK version
- CUDA compatibility

The ZED SDK must be installed before installing PyZED.

Official ZED SDK installation documentation:

https://www.stereolabs.com/docs/development/zed-sdk/linux

Official PyZED installation documentation:

https://www.stereolabs.com/docs/development/api-languages/python

### Check your system

Check the Linux architecture:

```bash
uname -m
```

For a standard x86-64 workstation, this should normally return:

```text
x86_64
```

Check the Python version:

```bash
python --version
```

This project expects:

```text
Python 3.12.x
```

Check whether the ZED SDK is installed:

```bash
ls /usr/local/zed
```

You can also check the installed ZED SDK version with:

```bash
cat /usr/local/zed/settings/Version
```

If that file is unavailable on your SDK version, inspect the SDK installation directory instead.

### Install the matching PyZED wheel

StereoLabs includes a helper script with the ZED SDK that automatically detects the platform, Python version, CUDA setup, and SDK version and downloads the corresponding PyZED package. :chatgpt-content-reference{index="1"}

With the robot virtual environment activated:

```bash
source .venv/bin/activate
cd /usr/local/zed
python3 get_python_api.py
```

The script downloads the appropriate wheel for the current system.

For example, a Python 3.12 x86-64 system may produce a wheel with a name similar to:

```text
pyzed-5.4-cp312-cp312-linux_x86_64.whl
```

If you only want to download the wheel and install it manually into the project environment, install it with:

```bash
uv pip install /path/to/pyzed-*.whl
```

Verify the installation:

```bash
python -c "import pyzed.sl as sl; print('PyZED import successful')"
```

The ZED SDK itself must still be installed on the machine; the Python wheel alone is not sufficient. :chatgpt-content-reference{index="2"}

---

## Franka setup

The workstation must be able to reach the Franka robot over the network and must have compatible Franka libraries installed.

Verify network connectivity to the robot:

```bash
ping <robot-ip>
```

For example:

```bash
ping 192.168.103.1
```

The project uses `pylibfranka` for robot communication.

If the Franka connection fails, verify that the installed Franka/libfranka version is compatible with the robot system version.

FCI must also be enabled on the robot.

---

## Franka Desk credentials

Create:

```text
robot/.env
```

Add the Franka Desk credentials:

```dotenv
DESK_USERNAME=your_username
DESK_PASSWORD=your_password
```

The `.env` file is ignored by Git and should not be committed.

---

## Model server setup

The MolmoAct2 model server has its own environment and dependency lockfile.

From the repository root:

```bash
cd model_server/official_molmoact2
uv sync
```

### Optional: download MolmoAct2-DROID beforehand

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download allenai/MolmoAct2-DROID
```

The checkpoint is approximately 22 GB.

Hugging Face authentication may be required depending on the model access settings.

If the model cache should live on another disk, set `HF_HOME` before downloading or starting the server:

```bash
export HF_HOME=/path/to/model/cache
```

---

# Running the system

The normal workflow uses two terminals.

- Terminal 1 runs the MolmoAct2 inference server.
- Terminal 2 initializes and runs the Franka controller.

---

## Terminal 1: start MolmoAct2-DROID

From the repository root:

```bash
cd model_server/official_molmoact2
```

Start the model server:

```bash
uv run python examples/droid/host_server_droid.py \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16
```

`0.0.0.0` allows the server to accept connections on all network interfaces.

Only expose port `8000` on trusted networks.

Wait until the model has fully loaded before starting the robot controller.

### Check that the server is reachable

For a model server running on the same computer:

```bash
curl http://127.0.0.1:8000/act
```

For a model server running on another machine:

```bash
curl http://<server-ip>:8000/act
```

Example:

```bash
curl http://192.168.1.20:8000/act
```

---

## Terminal 2: initialize the robot

Return to the repository root.

Activate the robot-side environment:

```bash
source .venv/bin/activate
```

Initialize the Franka:

```bash
uv run python robot/robot_init.py
```

Check the current joint pose:

```bash
uv run python robot/read_jointpose.py
```

Optionally move the robot to the predefined home pose:

```bash
uv run python robot/home_robot.py
```

---

## Start the controller

Currently, `main_controller.py` should be started from inside the `robot/` directory because it loads local runtime files relative to the current working directory.

```bash
cd robot
../.venv/bin/python main_controller.py
```

The controller prints the current run information before policy execution begins.

Review the output and robot workspace before continuing.

The controller waits for confirmation before beginning execution.

Stop the controller with:

```text
Ctrl+C
```

During shutdown, the controller requests a smooth robot stop, closes the camera and gripper connections, releases video writers, and joins the Franka control process.

---

# Typical startup sequence

After the environments have already been installed, the normal startup procedure is:

## Terminal 1

```bash
cd model_server/official_molmoact2

uv run python examples/droid/host_server_droid.py \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16
```

Wait until the model is ready.

## Terminal 2

From the repository root:

```bash
source .venv/bin/activate
```

Initialize the robot:

```bash
uv run python robot/robot_init.py
```

Check its current pose:

```bash
uv run python robot/read_jointpose.py
```

Optionally move to the home pose:

```bash
uv run python robot/home_robot.py
```

Then start the controller:

```bash
cd robot
../.venv/bin/python main_controller.py
```

---

# Useful diagnostics

Run these commands from the repository root unless otherwise specified.

## Test PyZED installation

```bash
python -c "import pyzed.sl as sl; print('PyZED OK')"
```

---

## Test both ZED cameras

```bash
uv run python -m cameras.zed_camera
```

A GUI window should open.

Press `q` to exit.

---

## Check connected USB devices

This can help when diagnosing the cameras or Robotiq adapter:

```bash
lsusb
```

---

## Check Robotiq serial device

```bash
ls -l /dev/ttyUSB*
```

The default setup expects:

```text
/dev/ttyUSB0
```

---

## Test Robotiq communication

```bash
uv run python gripper/robotiq_status_test.py
```

Press `Ctrl+C` to stop.

---

## Read the Franka joint pose

```bash
uv run python robot/read_jointpose.py
```

---

## Test the inference server

`inference/test_model_client.py` can be used as a standalone protocol test.

It currently expects:

```text
external.jpg
wrist.jpg
```

in the current working directory.

If the model server is running remotely, update the server address used by the test client.

---

# Experiment outputs

Each controller launch creates the next available directory under:

```text
robot/experiments/
```

For example:

```text
robot/experiments/iteration_3/
```

A typical experiment directory contains:

```text
iteration_N/
├── config.yaml
├── robot_run.txt
├── images/
│   ├── step_001_side.png
│   └── step_001_wrist.png
└── videos/
    ├── side_trajectory.mp4
    └── wrist_trajectory.mp4
```

`config.yaml` stores a snapshot of the configuration used for the run.

`robot_run.txt` stores recorded robot state and executed action chunks.

The `images/` and `videos/` directories contain observations and trajectory recordings from the cameras.

An experiment directory may be created even if initialization later fails.

---

# Hardware and software requirements

## Robot-side workstation

- Linux
- Python 3.12
- `uv`
- Franka robot with FCI enabled
- compatible Franka/libfranka installation
- Robotiq 2F-85 gripper
- two ZED cameras
- compatible ZED SDK
- matching PyZED Python API
- network connectivity to the Franka
- network connectivity to the MolmoAct2 server

## Model-server workstation

- NVIDIA GPU with sufficient VRAM for MolmoAct2-DROID
- compatible NVIDIA driver
- CUDA/PyTorch environment provided by the model-server project
- approximately 22 GB or more of disk space for the model checkpoint

The model server and robot controller can run on the same workstation or on separate machines.

---

# Troubleshooting

## `config.yaml` not found

Start the controller from the `robot/` directory:

```bash
cd robot
../.venv/bin/python main_controller.py
```

---

## `ModuleNotFoundError: No module named 'pyzed'`

Verify that the robot virtual environment is active:

```bash
source .venv/bin/activate
```

Then check:

```bash
python -c "import pyzed.sl"
```

If PyZED is not installed, use the StereoLabs installer:

```bash
cd /usr/local/zed
python3 get_python_api.py
```

Make sure the script is executed using the Python environment you intend to use.

---

## PyZED wheel is incompatible

Check:

```bash
uname -m
python --version
```

The wheel tags must match the system.

For example:

```text
cp312
```

means CPython 3.12, while:

```text
linux_x86_64
```

indicates 64-bit x86 Linux.

Do not use a wheel built for a different Python version or architecture.

Using StereoLabs' `get_python_api.py` is preferred because it automatically selects the matching package. :chatgpt-content-reference{index="3"}

---

## ZED SDK not found

Check:

```bash
ls /usr/local/zed
```

If the directory does not exist, install the ZED SDK first.

Official Linux installation guide:

https://www.stereolabs.com/docs/development/zed-sdk/linux

---

## ZED camera open failure

Verify:

```bash
lsusb
```

and check:

- ZED SDK installation
- USB connectivity
- camera permissions
- configured camera serial numbers
- compatibility between the ZED SDK and PyZED package

---

## Cannot reach `/act`

Check that:

- the model has finished loading
- the server process is running
- the server IP address is correct
- port `8000` is reachable
- the firewall allows the connection
- the robot workstation and model-server workstation can reach each other

Test manually:

```bash
curl http://<server-ip>:8000/act
```

---

## CUDA out of memory

Try:

- keeping `--dtype bfloat16`
- stopping other GPU workloads
- avoiding `--cuda-graph`, which requires additional VRAM

---

## Robotiq serial failure

Check whether the serial device exists:

```bash
ls -l /dev/ttyUSB*
```

Verify:

- `/dev/ttyUSB0`
- USB connectivity
- Linux serial permissions
- baud rate
- Modbus slave ID

---

## Franka connection failure

Verify:

- robot network connectivity
- Franka Desk credentials
- brakes are released
- FCI is enabled
- Franka/libfranka compatibility
- no other controller currently owns the robot connection

Test connectivity:

```bash
ping <robot-ip>
```

---

## Trajectory rejected

Review the error printed by the controller before changing any safety limits.

Do not increase safety thresholds simply to suppress a trajectory validation error.

---

## Permission denied for `/dev/ttyUSB0`

Check the device permissions:

```bash
ls -l /dev/ttyUSB0
```

On Ubuntu systems, serial devices are commonly associated with the `dialout` group.

Check your groups:

```bash
groups
```

If required, add the current user:

```bash
sudo usermod -aG dialout $USER
```

Log out and log back in before testing again.