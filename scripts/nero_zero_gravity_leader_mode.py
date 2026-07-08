#!/usr/bin/env python3
"""Minimal right-arm Nero leader-mode test.

This script does exactly one experiment:
connect right Nero on nero_right, enable it, call robot.set_leader_mode(),
send drag-teach start, then wait.
It sends no move_j/move_js/move_p commands.
"""

from __future__ import annotations

import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
from pyAgxArm.protocols.can_protocol.msgs.nero.default import ArmMsgMotionCtrl


CHANNEL = "nero_right"
FIRMWARE = NeroFW.V120
ENABLE_TIMEOUT_S = 12.0


def start_drag_teach(robot) -> None:
    robot._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=0x01))


def stop_drag_teach(robot) -> None:
    robot._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=0x02))


def print_status(robot, label: str) -> None:
    print(f"\n== {label} ==")
    try:
        firmware = robot.get_firmware(timeout=0.8, min_interval=0.2)
    except TypeError:
        firmware = robot.get_firmware()
    print("firmware:", firmware)

    status = robot.get_arm_status()
    if status is None:
        print("arm_status: None")
    else:
        msg = status.msg
        print(
            "arm_status:",
            {
                "ctrl_mode": str(msg.ctrl_mode),
                "arm_status": str(msg.arm_status),
                "mode_feedback": str(msg.mode_feedback),
                "teach_status": str(msg.teach_status),
                "motion_status": str(msg.motion_status),
                "err_status": str(msg.err_status),
            },
        )

    try:
        print("joints_enable:", robot.get_joints_enable_status_list())
    except Exception as exc:
        print("joints_enable error:", type(exc).__name__, exc)

    joints = robot.get_joint_angles()
    print("joint_angles:", getattr(joints, "msg", joints))


def enable_or_timeout(robot) -> None:
    print("Clearing joint errors...")
    robot.clear_joint_error(255)
    time.sleep(0.3)

    print(f"Enabling right Nero on {CHANNEL}...")
    deadline = time.time() + ENABLE_TIMEOUT_S
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        ret = robot.enable()
        time.sleep(0.5)
        enable_list = robot.get_joints_enable_status_list()
        enabled = all(enable_list)
        print(f"enable attempt {attempt}: enable_ret={ret}, enable_list={enable_list}, enabled={enabled}")
        if enabled:
            return
    raise TimeoutError(f"Timed out enabling right Nero on {CHANNEL}.")


def main() -> None:
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=FIRMWARE,
        interface="socketcan",
        channel=CHANNEL,
    )
    robot = AgxArmFactory.create_arm(cfg)

    print(f"Connecting right Nero on {CHANNEL} with firmware={FIRMWARE}...")
    robot.connect()
    try:
        print_status(robot, "before enable")
        enable_or_timeout(robot)
        print_status(robot, "after enable")

        print("\nCalling robot.set_leader_mode() now.")
        robot.set_leader_mode()
        time.sleep(1.0)
        print_status(robot, "after set_leader_mode")

        print("\nSending drag teach start: ArmMsgMotionCtrl(grag_teach_ctrl=0x01).")
        start_drag_teach(robot)
        time.sleep(1.0)
        print_status(robot, "after drag teach start")

        print("\nRight arm should now be draggable if the controller accepted leader + drag-teach.")
        print("Try gently moving it while supporting the arm. Press ENTER to stop drag teach, restore follower mode, and disconnect.")
        input()

        print("Sending drag teach stop: ArmMsgMotionCtrl(grag_teach_ctrl=0x02).")
        stop_drag_teach(robot)
        time.sleep(0.5)
        print_status(robot, "after drag teach stop")

        print("Calling robot.set_follower_mode() before exit.")
        robot.set_follower_mode()
        time.sleep(0.5)
        print_status(robot, "after set_follower_mode")
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
