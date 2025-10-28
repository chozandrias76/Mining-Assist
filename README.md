# Mining Assist

Mining Assist is a Python-only, Poetry-managed project for training and running an agent that can play a 3D game across multiple control contexts (e.g., MENU and FLIGHT) using one keyboard/mouse. It uses a mode detector (CNN) plus context-specific policies (discrete for menus, continuous for flight) coordinated by a simple manager.

## Project Status

🚧 **This project is currently in early development.** The foundational components have been implemented and tested, but the full architecture is not yet complete. See the [Feature Progress](#feature-progress) section below for detailed status.

**Current State:**
- ✅ Core screen capture system with dxcam/mss fallback
- ✅ Basic teleoperation recording framework
- ✅ Utility functions for data handling
- ✅ Comprehensive test suite with 100% pass rate
- ✅ Development tooling (Ruff, pre-commit, Poetry)
- ❌ Mode detector CNN not implemented
- ❌ Policy training scripts not implemented
- ❌ Live agent inference not implemented

## Architecture

Mining Assist follows a modular architecture with clear separation of concerns:

**Core Components:**
- **Screen Capture** - Multi-backend system (dxcam/mss) for real-time frame acquisition
- **Mode Detection** - CNN classifier to identify game states (MENU/FLIGHT/LOADING)
- **Policy Manager** - Routes decisions to context-appropriate sub-policies
- **Reinforcement Learning** - Stable-Baselines3 integration for discrete/continuous control
- **Data Pipeline** - Recording, preprocessing, and training data management

**Key Design Principles:**
- **Modular**: Each component can be developed and tested independently
- **Extensible**: Easy to add new game modes or control schemes
- **Robust**: Comprehensive error handling and fallback mechanisms
- **Type-Safe**: Full type annotations and static analysis

## Feature Progress

### ✅ Completed Components

#### Core Infrastructure
- **Screen Capture (`grabber.py`)** - Complete with dxcam/mss fallback, window detection, FPS tracking
- **Teleoperation Recording (`teleop_record.py`)** - Basic framework for recording gameplay sessions
- **Utilities (`utils.py`)** - Image preprocessing, data validation, file I/O helpers
- **Test Suite** - Comprehensive tests with mocks for all core components
- **Development Tooling** - Ruff linting/formatting, pre-commit hooks, Poetry dependency management

#### Package Structure
- **Poetry Configuration** - Proper dependency management and virtual environment setup
- **Type Hints** - Full type annotation throughout codebase
- **Error Handling** - Robust exception handling with proper logging
- **Documentation** - Comprehensive docstrings and inline comments

### 🚧 In Progress Components

*None currently - ready for next development phase*

### ❌ Not Yet Implemented

#### Core ML Components
- **Mode Detector (`detector.py`)** - CNN for classifying game states (MENU/FLIGHT/LOADING)
- **Policy Manager (`manager.py`)** - Routes decisions to appropriate sub-policies
- **Policies (`policies.py`)** - SB3 wrapper policies for different game modes
- **Gymnasium Environment (`env.py`)** - RL environment wrapper

#### Training Scripts
- **Mode Labeling (`label_modes.py`)** - GUI tool for labeling captured frames
- **Detector Training (`train_detector.py`)** - CNN training pipeline
- **Policy Training (`train_policy.py`)** - RL policy training per mode
- **Evaluation (`eval_rollout.py`)** - Performance metrics and evaluation

#### Runtime Components
- **Live Agent (`run_agent.py`)** - Real-time inference and control
- **I/O Devices (`io_devices.py`)** - Keyboard/mouse/gamepad output mapping
- **Main CLI (`main.py`)** - Command-line interface

#### Configuration Files
- **Detector Config (`cfg/detector.yaml`)** - CNN training parameters
- **Policy Configs (`cfg/train_*.yaml`)** - RL training configurations

### 🎯 Next Development Priorities

1. **Mode Detector Implementation** - Build and train CNN for game state classification
2. **Basic Policy Framework** - Implement discrete/continuous policy wrappers
3. **Gymnasium Environment** - Create RL environment with proper observation/action spaces
4. **I/O Device Mapping** - Implement safe keyboard/mouse control with focus guards

## Install (Poetry)

**Navigate to project directory:**
```bash
cd "a:\Code Projects\Mining-Assist\mining_assist"
```

**Install dependencies:**
```bash
poetry install
```

**Activate shell:**
```bash
poetry shell
```

> **Note:** The project automatically falls back from dxcam to mss if dxcam is unavailable.

## Quick Start (Current Implementation)

**Test screen capture:**
```bash
poetry run python -c "from mining_assist.grabber import test_grabber; test_grabber()"
```

**Record a basic teleoperation session:**
```bash
poetry run python -m mining_assist.scripts.teleop_record --game-window "Notepad" --fps 10 --duration 30
```

**Run tests:**
```bash
poetry run pytest
```

**Check code quality:**
```bash
poetry run ruff check
poetry run ruff format
```

## Planned Quick Start (Future Implementation)

**Record a teleop session (frames + inputs + mode tags):**
```bash
poetry run python scripts/teleop_record.py --game-window "Star Citizen" --fps 20 --out data/demos/session1
```

**(Optional) Label/correct modes for detector training:**
```bash
poetry run python scripts/label_modes.py --in data/frames --out data/frames_labeled
```

**Train the mode detector:**
```bash
poetry run python scripts/train_detector.py --cfg cfg/detector.yaml --data data/frames_labeled --out runs/detector
```

**Train sub-policies (independently):**
```bash
poetry run python scripts/train_policy.py --mode menu --cfg cfg/train_menu.yaml --detector runs/detector/best.pt
poetry run python scripts/train_policy.py --mode flight --cfg cfg/train_flight.yaml --detector runs/detector/best.pt
```

**Run the live agent:**
```bash
poetry run python scripts/run_agent.py \
  --detector runs/detector/best.pt \
  --menu-policy runs/menu/best.zip \
  --flight-policy runs/flight/best.zip \
  --game-window "Star Citizen"
```

## Current Implementation Details

### Screen Capture (`grabber.py`)
- **Multi-backend support:** Automatic fallback from dxcam (high-performance) to mss (compatibility)
- **Window targeting:** Can focus on specific game windows with automatic region detection
- **Preprocessing:** Built-in frame resizing, format conversion, and FPS tracking
- **Focus detection:** Windows API integration to pause capture when game window loses focus

### Teleoperation Recording (`teleop_record.py`)
- **Input capture:** Records keyboard and mouse inputs with timestamps
- **Frame synchronization:** Captures screen frames synchronized with input events
- **Data structure:** Organized session data with metadata for training pipelines
- **Configurable recording:** Adjustable FPS, duration, and output formats

### Utilities (`utils.py`)
- **Image processing:** Efficient OpenCV-based preprocessing for ML pipelines
- **Data validation:** Type checking and format validation for recorded data
- **File I/O:** Robust handling of dataset loading and saving operations

### Development Infrastructure
- **Type safety:** Full type annotations with Pyright static analysis
- **Code quality:** Ruff linting and formatting with pre-commit hooks
- **Testing:** Comprehensive test suite with structured test organization
- **Documentation:** Detailed docstrings following Google style conventions

## 🧪 **Testing**

The project uses a structured testing approach with separate directories for different test types:

### **Test Structure**
```
tests/
├── unit/           # Unit tests - fast, isolated, no external dependencies
├── integration/    # Integration tests - test component interactions
├── e2e/           # End-to-end tests - full workflow validation
└── conftest.py    # Shared test fixtures
```

### **Running Tests**
```bash
# Run all tests
poetry run pytest

# Run only unit tests (fastest)
poetry run pytest tests/unit/

# Run integration tests
poetry run pytest tests/integration/

# Run end-to-end tests (may require additional setup)
poetry run pytest tests/e2e/

# Generate coverage report
poetry run pytest --cov=src/mining_assist --cov-report=html
```

### **Test Categories**
- **Unit Tests**: Test individual components in isolation with mocked dependencies
- **Integration Tests**: Test interactions between components, may use real system resources
- **E2E Tests**: Test complete workflows, require full system setup

## Planned Core Concepts (Future)

- **Mode detector:** small CNN (Torch) predicts MODE ∈ {MENU, FLIGHT, LOADING}.
- **Manager:** picks the sub-policy given the detected mode (no learning at first).
- **Policies:**
  - **MenuPolicy (discrete):** up/down/left/right, small/large cursor nudge, confirm, back, scroll up/down, tab left/right.
  - **FlightPolicy (continuous):** pitch, roll, yaw, throttle, fire (scaled in [-1, 1]).
- **IO mapping:** abstract actions → OS inputs via pywin32/pynput/vgamepad.
- **Env:** Gymnasium-compatible; observation = stacked frame(s) + optional detector logits/mode id.

## Minimal Configs

### cfg/detector.yaml
```yaml
model: resnet18
input_size: [128, 72] # width, height
modes: ["MENU","FLIGHT","LOADING"]
train:
  batch_size: 128
  epochs: 10
  lr: 3e-4
  augment:
    brightness: 0.1
    contrast: 0.1
```

### cfg/train_flight.yaml
```yaml
algo: PPO
frame_stack: 4
action_repeat: 3
reward:
  dense:
    stability: 0.05
    target_proximity: 0.2
  terminal:
    crash: -1.0
    mission_complete: 1.0
ppo:
  n_steps: 2048
  batch_size: 256
  gamma: 0.995
  gae_lambda: 0.95
  clip_range: 0.2
```

### cfg/train_menu.yaml
```yaml
algo: PPO
discrete_actions: 12
reward:
  step_penalty: -0.01
  correct_transition: 1.0
  timeout: -0.3
ppo:
  n_steps: 1024
  batch_size: 256
  gamma: 0.99
```

## Observations & Actions

### Observations (Dict):
- **frame:** uint8 [H, W, frame_stack] (preprocessed to grayscale/downsized).
- **mode (optional):** int or detector logits to aid policy stability.

### Actions:
- **MENU:** Discrete(12) (action masking ensures only menu actions are permitted).
- **FLIGHT:** Box([-1,1]^5) with smoothing and deadzones.
- **LOADING:** no-op until detector changes mode.

## Rewards (suggested)

### Menu:
- +1 on reaching target UI state (verified by simple template or OCR).
- −0.01 per step, −0.3 on timeout.

### Flight:
- +1 on mission completion; −1 on crash.
- Dense shaping: stability, target proximity, time alive.

## Safety & Robustness

- **Focus guard:** agent pauses if game window not focused.
- **Desync guard:** hash HUD regions; if mismatch, re-detect and reset.
- **Rate limiter:** clamp mouse delta and key cadence.
- **Failsafe hotkey:** Ctrl+Alt+S toggles agent enable/disable.

## Training Tips

- Start with behavior cloning from teleop data, then DAgger, then PPO fine-tune.
- Action repeat 2–4 for stability; frame_stack 4.
- Keep detector accuracy high (>98%) before policy training.
- Log to runs/ (TensorBoard, screenshots on reward events).

## Development Notes for AI Agents

### Current Technical Debt
- **Import structure:** Some modules have conditional imports that could be cleaner
- **Error handling:** Could use more specific exception types in some areas
- **Configuration:** Basic YAML config exists but needs expansion for training parameters

### Known Working Patterns
- **Screen capture:** The grabber module handles fallback between backends cleanly
- **Testing approach:** Mock-based testing works well for hardware-dependent components
- **Type annotations:** Using modern Python union syntax (`X | Y`) throughout

### Implementation Decisions Made
- **Poetry over pip:** Chosen for dependency management and virtual environment control
- **Ruff over Black/flake8:** Faster linting and formatting with single tool
- **src/ layout:** Package in src/ directory following modern Python practices
- **Comprehensive testing:** All core modules have full test coverage

### Next Implementation Strategy
1. **Start with mode detector:** Most critical component for the ML pipeline
2. **Build simple policies first:** Get discrete menu actions working before continuous flight
3. **Add safety features early:** Focus detection and rate limiting are crucial
4. **Use behavior cloning:** Start with imitation learning before full RL

### Dependencies (Current)

**Core ML/Vision:**
- torch, torchvision, numpy, opencv-python

**Screen Capture:**
- dxcam (preferred), mss (fallback)

**Input/Output:**
- pywin32 (Windows API), pynput (cross-platform input), vgamepad (gamepad simulation)

**RL Framework:**
- gymnasium, stable-baselines3

**Utilities:**
- tqdm (progress bars), omegaconf (configuration)

**Development:**
- pytest, ruff, mypy, pyright, pre-commit

## Extending

- **Add new modes** (e.g., INVENTORY, COMBAT) by: updating detector labels → adding policy config → mapping IO.
- **Plug in OCR** (easyocr) for text-driven menu targets.
- **Swap capture backend** (dxcam ↔ mss).
- **Add curriculum scripts** for staged missions.
