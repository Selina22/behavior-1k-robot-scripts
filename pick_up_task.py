"""
Pick Up Task - Robust Zero-Shot Script for Franka Manipulator
=============================================================
Learns from VLM code-writing agent experiments to create a reliable
pick-up policy that handles various objects and scenarios.

Experiment 1 Analysis: Pick up the green mug (2026-03-20)
---------------------------------------------------------
Key findings:
- get_position_from_labels works for locating objects but may fail for
  fine-grained parts (e.g., "handle of the green mug" had invalid depth)
- get_grasp_pose + plan_grasp attempted first but resulted in gripper closing
  on nothing (gripper_fraction_closed=0.99 = miss)
- Recovery: manual top-down grasp succeeded on retry
  - Moved to neutral [0.5, 0.0, 0.4] first
  - Re-located mug via get_position_from_labels
  - Used waypoint sequence: pre_grasp (+0.15z), approach (+0.05z), grasp (+0.02z)
  - Gripper closed at 0.96 = successful grasp, then lifted +0.20z
- Lesson: top-down manual grasp with proper waypoints is MORE reliable than
  get_grasp_pose for simple objects on tables. Always re-perceive after failure.

Initial script version based on Experiment 1:
"""

import numpy as np


# ============================================================
# Configuration
# ============================================================
NEUTRAL_POSITION = [0.5, 0.0, 0.4]
NEUTRAL_ORIENTATION = [np.pi, 0.0, 0.0]
TOP_DOWN_RPY = [np.pi, 0.0, 0.0]

PRE_GRASP_HEIGHT = 0.15   # Height above object for pre-grasp
APPROACH_HEIGHT = 0.05     # Height above object for approach
GRASP_HEIGHT = 0.02        # Height above object center for actual grasp
LIFT_HEIGHT = 0.20         # Height to lift after grasping


# ============================================================
# Core Functions
# ============================================================

def locate_object(robot, label):
    """Locate an object using get_position_from_labels with fallback to naive_pointing."""
    results = robot.get_position_from_labels([label])
    if results:
        pos = np.array(results[0]['3d_position'])
        print(f"Located '{label}' at {pos}")
        return pos

    # Fallback: rescan and retry
    print(f"Failed to locate '{label}', rescanning...")
    robot.rescan_wrist()
    results = robot.get_position_from_labels([label])
    if results:
        pos = np.array(results[0]['3d_position'])
        print(f"Located '{label}' at {pos} (after rescan)")
        return pos

    print(f"Could not locate '{label}' with get_position_from_labels")
    return None


def top_down_grasp(robot, target_pos):
    """
    Execute a top-down grasp sequence at the given position.
    Uses a 3-waypoint approach: pre-grasp -> approach -> grasp -> lift.

    Returns True if gripper is partially closed (indicating successful grasp).
    """
    pre_grasp = target_pos.copy()
    pre_grasp[2] += PRE_GRASP_HEIGHT

    approach = target_pos.copy()
    approach[2] += APPROACH_HEIGHT

    grasp = target_pos.copy()
    grasp[2] += GRASP_HEIGHT

    lift = target_pos.copy()
    lift[2] += LIFT_HEIGHT

    robot.open_gripper()
    robot.move_gripper_to(pre_grasp.tolist(), TOP_DOWN_RPY)
    robot.move_gripper_to(approach.tolist(), TOP_DOWN_RPY)
    robot.move_gripper_to(grasp.tolist(), TOP_DOWN_RPY)
    robot.close_gripper()
    robot.move_gripper_to(lift.tolist(), TOP_DOWN_RPY)

    return check_grasp_success(robot)


def check_grasp_success(robot):
    """Check if gripper is partially closed (0.1 < fraction < 0.9), indicating a grasp."""
    state = robot.get_state()
    gripper = state['robot_state']['gripper_fraction_closed']
    if 0.1 < gripper < 0.9:
        print(f"Grasp successful (gripper={gripper:.3f})")
        return True
    else:
        print(f"Grasp failed (gripper={gripper:.3f})")
        return False


def move_to_neutral(robot):
    """Move robot to a safe neutral position."""
    robot.move_gripper_to(NEUTRAL_POSITION, NEUTRAL_ORIENTATION)


def pick_up(robot, target_label):
    """
    Main pick-up routine. Locates the target object and picks it up
    using a top-down grasp strategy.

    Args:
        robot: The robot interface object
        target_label: String label for the object to pick up (e.g., "green mug")

    Returns:
        True if object was successfully picked up, False otherwise
    """
    print(f"=== Pick Up Task: '{target_label}' ===")

    # Step 1: Locate the object
    target_pos = locate_object(robot, target_label)
    if target_pos is None:
        print("FAILED: Could not locate target object.")
        return False

    # Step 2: Attempt top-down grasp
    if top_down_grasp(robot, target_pos):
        print(f"SUCCESS: Picked up '{target_label}'")
        return True

    # Step 3: Recovery - return to neutral, re-perceive, retry
    print("First attempt failed. Recovering...")
    move_to_neutral(robot)
    robot.open_gripper()

    target_pos = locate_object(robot, target_label)
    if target_pos is None:
        print("FAILED: Could not re-locate target after recovery.")
        return False

    if top_down_grasp(robot, target_pos):
        print(f"SUCCESS: Picked up '{target_label}' on retry")
        return True

    print(f"FAILED: Could not pick up '{target_label}' after retry.")
    return False


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # Usage: set target_label and optional obstacle_label before running
    # target_label = "green mug"
    # pick_up(robot, target_label)
    pass
