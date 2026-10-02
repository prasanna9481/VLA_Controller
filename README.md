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
```

Verify the installation:

```bash
uv --version
```

---

### 3. Install ZED SDK and PyZED

The PyZED wheel is not included in this repository.

Install the ZED SDK first, then download the PyZED wheel that matches your Python version, system architecture, and installed ZED SDK version.

The correct PyZED wheel depends on:

- operating system
- CPU architecture
- Python version
- ZED SDK version
- CUDA compatibility

The ZED SDK must be installed before installing PyZED.

Official ZED SDK installation documentation:

https://www.stereolabs.com/docs/development/zed-sdk/linux

Official PyZED installation documentation. Follow the exact steps given in the documentation:

https://www.stereolabs.com/docs/development/api-languages/python

---
## Setup

### 1. Create and synchronize the robot-side environment

The robot-side environment uses Python 3.12 and is managed with `uv`.

From the repository root:

```bash
uv sync
```


### 2. Franka Desk credentials

Create in the exact path:

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

### 3. Model server setup

The MolmoAct2 model server has its own environment and dependency lockfile.

From the repository root:

```bash
mkdir model_server
git clone https://github.com/allenai/molmoact2.git
cd model_server/molmoact2
uv sync
```

### Download MolmoAct2-DROID beforehand

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
uv run hf download allenai/MolmoAct2-DROID
```

The checkpoint is approximately 22 GB.

Hugging Face authentication may be required depending on the model access settings.
Follow the official repositpory of molmoact2 instruction for fine details.

---

# Running the system

The normal workflow uses two terminals.

- Terminal 1 runs the MolmoAct2 inference server.
- Terminal 2 initializes and runs the Franka controller.

---

## Terminal 1: start MolmoAct2-DROID

From the repository root:

```bash
cd model_server/molmoact2
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
python main_controller.py
```

The controller prints the current run information before policy execution begins.

Review the output and robot workspace before continuing.

During shutdown, the controller requests a smooth robot stop, closes the camera and gripper connections, releases video writers, and ends the Franka control process.

