Project README — Nanonis_AutoSTM_C-Br_bond_selectivity
===============================================

Overview
--------
This repository is a research-grade Python project for simulating and/or controlling molecular-selective reactions and tip manipulation in atomic-scale Scanning Tunneling Microscopy (STM). The project contains:

- simulation environments,
- tip controllers based on reinforcement learning (SAC and DQN),
- image segmentation and keypoint detection modules,
- low-level interfaces to communicate with the Nanonis instrument.

This README documents the purpose of each folder/file to help you get started and develop the project further.

Top-level structure (high level)
--------------------------------
- `AI_Tip_Main.py` — Main entry script that coordinates scanning, molecule detection, SAC agent and Nanonis instrument interaction.
- `Auto_scan_class.py` — High-level wrapper for automated scanning and image acquisition (scheduling and thread management).
- `core.py` — Higher-level wrapper over the Nanonis TCP interface: convenience functions for moving, scanning, Z-control, PLL, etc.
- `env_Br.py` — Simulation environment customized for Br site reactions (Molecule, Surface, interaction and probability models).
- `env.py` (if present) — Generic simulation environment used by `SAC.py`.
- `interface.py` — Nanonis programming interface (`nanonis_programming_interface`): builds/parses binary packets, send/receive.
- `molecule_registry.py` — Molecule registry (Registry) for recording, matching and maintaining detected molecules and positions (drift correction, updates, nearest search).
- `SAC.py` — Soft Actor-Critic (SAC) training and agent implementation (general-purpose).
- `SAC_Br.py` — SAC implementation customized for the Br task (data augmentation, replay buffer, visualization, training loop).
- `SAC_Br_nanonis.py` — (referenced by `AI_Tip_Main.py`) likely a Nanonis-integrated variant — check repository for existence.
- `DQN/` — DQN agent, network definitions and training scripts used by the tip shaper.
- `EvaluationCNN/` — CNN code for image quality assessment and detection (predict/detect, training/inference).
- `keypoint/` — Keypoint detection module (yolov7-like structure: models, data, scripts, configs and requirements).
- `mol_segment/` — Molecule segmentation module (UNet etc.) producing masks.
- `square_fitting.py` — Tools to detect Br atom sites on the molecule, fit squares/rotated boxes and compute angles / corner coordinates.
- `utils.py` — A large collection of utility functions: image processing, math, coordinate transforms, statistics and clustering, and helpers for ML.
- `STM_img_simu/` — Simulated or example STM images used as test data.
- dataset / model folders within `mol_segment/` and `keypoint/` (inspect paths as needed).

Detailed file / folder description
---------------------------------

Top-level scripts and utilities

- `AI_Tip_Main.py`
  - The main program (runnable script).
  - Orchestrates Nanonis controller (`core.py` / `interface.py`), image acquisition, image quality assessment (`EvaluationCNN`), molecule detection (`keypoint`), segmentation (`mol_segment`), Br site detection (`square_fitting.py`), and SAC agent communication via queues.
  - Typical flow: scan -> quality check -> molecule detection -> zoom-in & segmentation -> compute Br positions -> send molecule state to SAC -> receive action -> perform tip induction (CH or CC mode) -> save logs/images.

- `Auto_scan_class.py`
  - Encapsulates scanning routines (batch scans, line-scan thread, producer/consumer buffers).
  - Manages image acquisition threads and integrates with Nanonis wrapper functions in `core.py`.

- `core.py`
  - Higher-level device control class `NanonisController` built on top of `nanonis_programming_interface`.
  - Implements common helper methods: MotorMove, AutoApproach, Z control, PLL, ScanFrameSet/ScanSpeedSet, BiasPulse, etc.
  - Contains logging and error handling relevant for instrument control.

- `interface.py`
  - Low-level Nanonis protocol implementation: packet header/body construction, binary encoding/decoding, socket send/receive and response parsing.
  - Provides `nanonis_programming_interface` with `send()`, `parse_response()`, `convert()` and other helpers.


Environments, molecules and registry

- `env_Br.py`
  - A custom environment tailored for Br atom site tasks (classes like `Molecule`, `Surface`).
  - `Molecule` models geometry (square, triangle, circle), interaction area and parameter-based reaction probabilities (suc/ind/non), and encodes molecule states (`_encode_molecular_state`).
  - `Surface` and related classes simulate multiple molecules / adsorption sites and system-level interactions.

- `molecule_registry.py`
  - Defines `Molecule` (position, keypoints, Br positions, orientation, state, etc.) and `Registry` to register, correct drift, update and search molecules.


Reinforcement learning and agents

- `SAC.py`
  - A general Soft Actor-Critic implementation: `PolicyNet`, `CriticNet`, `ReplayBuffer` and training logic.
  - Designed for use with `env.py`-style environments; includes action remapping, action selection, update steps and model saving/loading.

- `SAC_Br.py`
  - A Br-task-specific extension of SAC in this repo:
    - Integrates tightly with `env_Br.py`, provides a replay buffer with symmetry-based data augmentation (rotations/flips), state legalization, and prioritized sampling logic.
    - Visualization utilities (`plot_train`) to store action distributions and reward curves.
    - Network definitions similar to `SAC.py` but adapted for the task (extra layers and activations).

- `DQN/` folder
  - Contains tip shaper DQN agent implementations (`agent.py`), including DQN and Dueling DQN architectures, `ReplayMemory` and a tip_shaper agent class.


Image processing, detection and segmentation

- `EvaluationCNN/`
  - CNN code for image quality assessment and detection. `AI_Tip_Main.py` calls `EvaluationCNN.detect.predict_image_quality` to assess scan images.

- `keypoint/`
  - A keypoint detection submodule structured after yolov7; includes `detect.py`, dataset scripts, model definitions, config files and `requirements.txt`.

- `mol_segment/`
  - Molecule segmentation using UNet (training scripts, inference). `detect.py` exposes a segmentation inference API returning masks for downstream processing.

- `square_fitting.py`
  - Fits molecule squares (rotated boxes) from segmentation and keypoint outputs, detects candidate Br sites and computes orientation and corner positions.


Utilities

- `utils.py`
  - A collection of utility functions used across the project:
    - Math and distributions (2D normal, clustering helpers, FFT, fit routines)
    - Image utilities (CLAHE, coordinate transforms, motion vector estimation, feature matching)
    - Unit conversions and parsing helpers (`sci_to_unit`, `unit_to_sci`)
    - ML helpers (weights_init_ for model initialization)
  - Many behaviors and helpers referenced in the README come from this module; read specific docstrings in the file for details.


Quick start (minimal)
---------------------
Prerequisites:

- Python 3.8 / 3.9 / 3.10 recommended.
- Create a virtual environment and install dependencies (some submodules provide their own requirements).

| Category | Packages (tested version) |
|----------|---------------------------|
| Core runtime | `numpy 1.26`, `scipy 1.11`, `matplotlib 3.8`, `opencv-python 4.9`, `Pillow 10`, `tqdm 4.66`, `PyYAML 6.0` |
| Machine learning | `torch 2.2`, `torchvision 0.17`, `scikit-learn 1.3`, `tensorboard 2.16`, `onnxruntime 1.17`, `thop 0.1` |
| Detection extras | `pycocotools 2.0.7`, `seaborn 0.13`, `pandas 2.1` |
| Environment/sim | `gym 0.26`, `pygame 2.5` (only when running the tip game) |
| Imaging libraries | `scikit-image 0.22` (used by `AI_Tip_Main.py`) |

Nanonis API
Feedback control, bias/current pulses, and scanning operations are executed through the official Nanonis TCP programming interface.
The STM control layer is adapted from the open-source library:
https://github.com/dilwong/nanonis_control

Install example (global runtime deps):

```powershell
pip install numpy scipy matplotlib opencv-python torch torchvision scikit-learn Pillow tqdm
```

If you plan to use the `keypoint` module, install its requirements:

```powershell
pip install -r keypoint/requirements.txt
```

Run the main script:

- If you have a connected Nanonis instrument and want to drive it:

  - Edit `AI_Tip_Main.py` to set the Nanonis IP/PORT and choose `SAC`/`DQN` mode.
  - Run:

```powershell
python AI_Tip_Main.py
```

- If you only want to use the simulation or offline analysis (no device):

  - Use the simulation environment in `env_Br.py` or write a small test script to exercise `SAC.py` / `SAC_Br.py`.
  - Run training examples inside `SAC.py` / `SAC_Br.py` after adapting hyperparameters.


Dependencies and environment advice
----------------------------------
- Recommend running with a full PyTorch CPU/GPU environment when training RL agents or detection models.
- For detection/segmentation modules install `opencv-python`, `torch` and `torchvision`. The `keypoint` module (yolov7-style) may require additional packages listed in `keypoint/requirements.txt`.


Developer notes and improvement suggestions
-----------------------------------------
- This README is a high-level overview. After reading the code I suggest:
  - Decouple device communication (`core.py` / `interface.py`) from higher-level control logic (`AI_Tip_Main.py`) by introducing small interfaces for easier testing and simulation.
  - Add unit tests for key modules (e.g. `env_Br.py` state encoding, `molecule_registry.py` matching/drift correction, `square_fitting.py` site detection).
  - Add CLI/config support (yaml/json) to `SAC_Br.py` / `SAC.py` to make hyperparameter reuse easier across experiments.
  - Break large functions and add docstrings (especially inside `utils.py`) to improve maintainability.


Key file index (quick lookup)
-----------------------------
- `AI_Tip_Main.py` — main orchestration
- `core.py` / `interface.py` — Nanonis communication and higher-level controls
- `env_Br.py` — Br environment (Molecule/Surface)
- `molecule_registry.py` — molecule registry and management
- `SAC_Br.py` / `SAC.py` — RL agents
- `DQN/agent.py` — DQN agent implementation and training
- `mol_segment/` — molecule segmentation
- `keypoint/` — keypoint detection
- `square_fitting.py` — Br site detection / square fitting tools
- `utils.py` — utility functions


Done and next steps
-------------------
I have scanned the repository and generated this English README from the code and comments. I can also:

- Expand the "Quick start" section with step-by-step installation commands tailored to your environment (GPU/CPU, specific Python).
- Auto-generate a short API reference for core modules (extract classes/functions and docstrings).
- Add a minimal runnable example script that uses the simulation environment and trains a small SAC agent (fast smoke-test).

Would you like me to (choose one):

1) Add a step-by-step installation & run guide with PowerShell commands, or
2) Paste the full English README text here (already written to file), or
3) Create a short example script to run in simulation? 


Gameplay (interactive STM simulation)
------------------------------------
This repository includes an interactive STM-like simulation implemented in `env_Br.py`. Below is a concise guide to the gameplay, controls, displayed information and win/lose conditions.

Objective
- The final goal of the simulation is to successfully react (convert) all four reaction sites on a molecule to the "reacted" state (displayed as green). You accomplish this by manipulating the tip (mouse and keyboard) to apply position, bias voltage and current to induce reactions.

What you see on screen
- The molecule is drawn at the center (or grid position). For a square-shaped molecule:
  - The molecule body is drawn in blue.
  - Each of the 4 site circles is colored according to its state:
    - Blue / dark blue: unreacted
    - Green: reacted (success)
    - Red: broken / failed
  - The tip is a small green circle that follows the mouse pointer.
  - Top-left text: current tip bias, tip current, and the molecule's optimal bias/current.
  - Bottom-left: per-site success / non / fail probabilities computed in real time based on tip position and parameters.
  - Debug text also shows the molecule internal encoded state.

Controls (keyboard)
- R — Reset all molecules to their initial unreacted states (set each site to unreacted).
- LEFT / RIGHT arrows — Rotate the molecule (rotate the positions of keypoints / sites) by -1 / +1 degree.
- W / S — Increase / decrease tip bias voltage by 0.05 V per key press.
- D / A — Increase / decrease tip current by 0.05 nA per key press.

Note about holding keys
- The environment supports repeating actions when you hold down a key:
  - After holding a key for 0.5 s, the related action will repeat at ~0.1 s intervals until you release the key.

Controls (mouse)
- Move the mouse — Moves the tip position (the tip follows the mouse pointer). The simulation updates site reaction probabilities in real time based on distance and position.
- Left mouse button — Apply a single interaction at the current tip position: the environment runs the probabilistic interaction for each unreacted site and updates site states (success / fail / no-change).

Interaction model (brief)
- When you interact (left-click), the tip computes three per-site factors:
  - suc_factor: probability of successful reaction (site becomes reacted / green), depends on distance, tip bias and current, and site location.
  - fail_factor: probability of a failing/broken state (site becomes broken / red).
  - non_factor: probability of no change.
- These factors are calculated using the molecule's geometry (`interact_area`) and parameter response profiles (`interact_parameters`) and combined to probabilistically update each unreacted site.

Rewards and episode termination
- The environment computes a reward using `reward_culculate` that:
  - Penalizes broken/invalid reactions and non-changes.
  - Rewards valid reaction steps according to the reaction path.
- An episode ends (done=True) when any of the following occur:
  - Any site becomes broken (state value 2) — considered a failure for that site.
  - All four reaction sites are reacted (all the first four site flags are 1) — success.
  - The reaction result is categorized as "bad" or "wrong" by the reward logic.

How to run the interactive simulation
- Run `env_Br.py` directly (it contains a `__main__` entry that starts the Pygame window):

```powershell
python env_Br.py
```

Tips for play/testing
- Move the tip close to a particular site to raise the corresponding suc_factor. Use small bias/current adjustments with W/S and A/D to tune the reaction probabilities toward success while avoiding raising fail_factor.
- Rotate the molecule (LEFT/RIGHT) if you want to align the tip to a specific site geometry.
- Use R to reset the scene if you break a site.

Integration with agents
- The interactive environment is also used by RL agents (SAC / DQN) in the repo. When used in an algorithmic loop, actions are sampled from the agent and translated into tip x,y offset, bias and current, then the environment's `step()` applies the interaction and returns next state / reward / done / info.

Appendix: internal observation & action format
- Action space: 4D continuous — (dx, dy, bias, current). x,y are offsets relative to the molecule center (discretized to grid points in code), and bias/current are clamped to the defined ranges.
- Observation: the environment returns the molecule state vector (including path bit) as a float32 array.


