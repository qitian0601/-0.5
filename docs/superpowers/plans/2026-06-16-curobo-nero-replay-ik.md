# cuRobo Nero Replay IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `scripts/replay_bus_table_first5_ee_pose.sh` replay recorded Nero EE pose datasets through cuRobo IK and Nero joint commands.

**Architecture:** Keep the existing replay script and SDK connection flow. Add a small cuRobo IK adapter in `scripts/replay_nero_dual_ee_pose.py` that converts Nero SDK Euler poses to cuRobo `Pose`, solves joint targets, and sends them with `move_js`.

**Tech Stack:** Python, pytest, scipy Rotation, cuRobo `InverseKinematics`, Nero SDK `move_js`.

---

### Task 1: Test pose conversion and cuRobo command path

**Files:**
- Test: `tests/scripts/test_replay_nero_dual_ee_pose_curobo.py`
- Modify: `scripts/replay_nero_dual_ee_pose.py`

- [ ] Write tests for Euler-to-quaternion conversion and `move_dual_ee` dispatching to `move_js` when an IK backend is supplied.
- [ ] Run the focused tests and confirm they fail because the helpers/backend do not exist yet.
- [ ] Implement the helpers and backend dispatch.
- [ ] Run focused tests and confirm they pass.

### Task 2: Wire CLI and shell script

**Files:**
- Modify: `scripts/replay_nero_dual_ee_pose.py`
- Modify: `scripts/replay_bus_table_first5_ee_pose.sh`

- [ ] Add `--ik-backend`, `--curobo-robot`, `--curobo-num-seeds`, `--curobo-position-threshold`, `--curobo-rotation-threshold`, and `--euler-order`.
- [ ] Instantiate the cuRobo IK adapter only when `--ik-backend=curobo`.
- [ ] Pass `--ik-backend=curobo --curobo-robot=nero_custom.yml` from the shell script.
- [ ] Run CLI help and focused tests.

### Task 3: Fix and validate `nero_custom.yml`

**Files:**
- Modify: `/home/chenglong/workplace/nero_teleop_ws/curobo/curobo/content/configs/robot/nero_custom.yml`

- [ ] Wrap the existing `kinematics:` body under `robot_cfg:`.
- [ ] Replace stale `/home/zfc/...` asset paths with local paths.
- [ ] Copy the existing Nero URDF into cuRobo assets if missing.
- [ ] Validate `InverseKinematicsCfg.create(robot="nero_custom.yml")` in the `lerobot` environment.
