"""
Pick Up Task - Robust Zero-Shot Script for Franka Manipulator
=============================================================
Learns from VLM code-writing agent experiments to create a reliable
pick-up policy that handles various objects and scenarios.

Experiment 1 (2026-03-20): Pick up the green mug
- Top-down manual grasp more reliable than get_grasp_pose for simple table objects
- 3-waypoint approach (pre_grasp/approach/grasp) + lift works well
- Always re-perceive after a failed grasp attempt

Experiment 2 (2026-03-23_17-46-38): Pick up green cup with yellow block obstacle
- get_position_from_labels may fail for some objects; naive_pointing is a good fallback
- Obstacle removal: first grasp at detected z failed (too high), lowering by 1.5cm worked
- Top-down grasp on cups can hit 'dt' motion planning errors near table surface
- get_grasp_pose + plan_grasp succeeded for cup handle grasp as fallback
- Lesson: use naive_pointing fallback in locate_object; for cups/mugs, try
  get_grasp_pose as a fallback strategy when top-down fails with 'dt' errors

Experiment 3 (2026-03-23_18-38-26): Pick up green mug on the shelf
- Top-down grasp impossible due to shelf above; side grasp mandatory
- get_grasp_pose('the green mug', 'grasp from the side') + plan_grasp worked first try
- After grasping inside shelf: MUST retreat horizontally (x -= 0.15) BEFORE lifting
  vertically to avoid collision with shelf above
- Lesson: detect if object is in an enclosed space; use side grasp + horizontal retreat

Experiment 4 (2026-03-23_18-45-00): Pick up blue cube with orange cube obstacle
- Obstacle removal worked: top-down grasp on orange cube (gripper=0.436), placed aside
- Blue cube detected at z=0.035 - way too low for gripper to reach
- ALL grasp attempts (top-down, get_grasp_pose, neutral reset) failed with 'dt' error
- Root cause: z < ~0.05 means object is at/below table surface level, unreachable
- Lesson: clamp minimum grasp z to avoid motion planning failures. If z < MIN_GRASP_Z,
  raise it to MIN_GRASP_Z. Also re-perceive the target AFTER obstacle removal since
  the obstacle may have been occluding proper depth sensing.

Experiment 5 (2026-03-23_16-59-15): Pick up green cup with yellow block obstacle
- Same pattern: get_position_from_labels failed for yellow block, naive_pointing worked
- Obstacle grasp at approach height (z+0.02) succeeded (gripper=0.586)
- Cup detected at z=0.026 - too low, top-down and plan_grasp both hit 'dt' errors
- After robot.reset() + re-perceive + 4cm z-offset, top-down finally succeeded (gripper=0.41)
- Lesson: robot.reset() is a last-resort recovery for stuck states. Adding z_offset=+0.04
  to low-z objects is a proven fix. The approach height itself should ensure we don't
  go below the safe z threshold.
"""

import numpy as np


# ============================================================
# Configuration
# ============================================================
NEUTRAL_POSITION = [0.5, 0.0, 0.4]
NEUTRAL_ORIENTATION = [np.pi, 0.0, 0.0]
TOP_DOWN_RPY = [np.pi, 0.0, 0.0]

PRE_GRASP_HEIGHT = 0.15   # Height above object for pre-grasp
APPROACH_HEIGHT = 0.02     # Height above object for approach (reduced from 0.05)
GRASP_HEIGHT_OFFSET = 0.0  # Offset from detected position for grasp
LIFT_HEIGHT = 0.20         # Height to lift after grasping

PLACE_OFFSET_Y = -0.20    # Y-offset for placing obstacles to the side
MIN_GRASP_Z = 0.06        # Minimum z for grasp (below this causes 'dt' errors)


# ============================================================
# Core Functions
# ============================================================

def locate_object(robot, label):
    """
    Locate an object using get_position_from_labels, with fallback chain:
    1. get_position_from_labels
    2. rescan_wrist + get_position_from_labels
    3. naive_pointing (most robust fallback, learned from Exp 2)
    """
    # Primary method
    results = robot.get_position_from_labels([label])
    if results:
        pos = np.array(results[0]['3d_position'])
        print(f"Located '{label}' at {pos}")
        return pos

    # Fallback 1: rescan and retry
    print(f"Failed to locate '{label}', rescanning...")
    robot.rescan_wrist()
    results = robot.get_position_from_labels([label])
    if results:
        pos = np.array(results[0]['3d_position'])
        print(f"Located '{label}' at {pos} (after rescan)")
        return pos

    # Fallback 2: naive_pointing (learned from Exp 2 - worked when primary failed)
    print(f"Trying naive_pointing for '{label}'...")
    result = robot.naive_pointing(label)
    if result is not None:
        pos = np.array(result['3d_position'])
        print(f"Located '{label}' at {pos} (via naive_pointing)")
        return pos

    print(f"Could not locate '{label}' with any method")
    return None


def top_down_grasp(robot, target_pos, z_offset=0.0):
    """
    Execute a top-down grasp sequence at the given position.
    Uses a 3-waypoint approach: pre-grasp -> approach -> grasp -> lift.

    Args:
        target_pos: 3D position of the object
        z_offset: Additional z offset for grasp point (negative = lower)

    Returns True if gripper is partially closed (indicating successful grasp).
    """
    grasp = target_pos.copy()
    grasp[2] += GRASP_HEIGHT_OFFSET + z_offset

    # Clamp grasp z to minimum safe height (learned from Exp 4: z=0.035 caused 'dt' errors)
    if grasp[2] < MIN_GRASP_Z:
        print(f"Warning: grasp z={grasp[2]:.3f} too low, clamping to {MIN_GRASP_Z}")
        grasp[2] = MIN_GRASP_Z

    pre_grasp = target_pos.copy()
    pre_grasp[2] += PRE_GRASP_HEIGHT

    approach = target_pos.copy()
    approach[2] += APPROACH_HEIGHT
    if approach[2] < MIN_GRASP_Z:
        approach[2] = MIN_GRASP_Z + 0.01

    lift = target_pos.copy()
    lift[2] += LIFT_HEIGHT

    robot.open_gripper()
    robot.move_gripper_to(pre_grasp.tolist(), TOP_DOWN_RPY)
    robot.move_gripper_to(approach.tolist(), TOP_DOWN_RPY)
    robot.move_gripper_to(grasp.tolist(), TOP_DOWN_RPY)
    robot.close_gripper()
    robot.move_gripper_to(lift.tolist(), TOP_DOWN_RPY)

    return check_grasp_success(robot)


def grasp_via_plan(robot, label, description="grasp the object", hint_pos=None,
                   retreat_before_lift=False):
    """
    Use get_grasp_pose + plan_grasp for more complex grasps (e.g., handles, shelf objects).
    Learned from Exp 2: works for cup handles when top-down fails.
    Learned from Exp 3: for shelf objects, retreat horizontally before lifting.

    Args:
        retreat_before_lift: If True, move backward (x -= 0.15) before lifting.
            Use this for objects inside enclosed spaces like shelves.

    Returns True if gripper indicates successful grasp.
    """
    print(f"Attempting planned grasp for '{label}'...")
    robot.open_gripper()

    kwargs = {}
    if hint_pos is not None:
        kwargs['point_to_grasp'] = hint_pos.tolist() if isinstance(hint_pos, np.ndarray) else hint_pos

    grasp_candidates = robot.get_grasp_pose(label, description, **kwargs)
    if not grasp_candidates:
        print(f"get_grasp_pose found no candidates for '{label}'")
        return False

    robot.plan_grasp(grasp_candidates)

    if check_grasp_success(robot):
        state = robot.get_state()['robot_state']
        current_pos = np.array(state['position'])
        current_ori = state['orientation']

        if retreat_before_lift:
            # Retreat horizontally first (learned from Exp 3: shelf objects)
            retreat_pos = current_pos.copy()
            retreat_pos[0] -= 0.15  # Pull back toward robot
            robot.move_gripper_to(retreat_pos.tolist(), current_ori)
            current_pos = retreat_pos

        # Lift after grasp
        lift_pos = current_pos.copy()
        lift_pos[2] += LIFT_HEIGHT
        robot.move_gripper_to(lift_pos.tolist(), current_ori)
        return True

    return False


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


def safe_reset(robot):
    """
    Last-resort recovery: full robot reset to home position.
    Learned from Exp 5: when robot gets stuck in unrecoverable poses,
    robot.reset() returns it to a known safe configuration.
    """
    print("Performing full robot reset (last resort recovery)...")
    robot.reset()


def remove_obstacle(robot, obstacle_label, target_pos):
    """
    Remove an obstacle object by picking it up and placing it to the side.
    Learned from Exp 2: may need z-offset adjustment if first grasp misses.

    Args:
        robot: Robot interface
        obstacle_label: Label of the obstacle to remove
        target_pos: Position of the target object (used to compute placement)

    Returns True if obstacle was successfully removed.
    """
    print(f"--- Removing obstacle: '{obstacle_label}' ---")

    obs_pos = locate_object(robot, obstacle_label)
    if obs_pos is None:
        print(f"Could not find obstacle '{obstacle_label}', assuming already clear.")
        return True

    # Attempt 1: grasp at detected position
    if top_down_grasp(robot, obs_pos):
        place_obstacle(robot, target_pos)
        return True

    # Attempt 2: lower the grasp by 1.5cm (learned from Exp 2)
    print("Obstacle grasp failed, retrying with lower z-offset...")
    move_to_neutral(robot)
    robot.open_gripper()

    obs_pos = locate_object(robot, obstacle_label)
    if obs_pos is None:
        print(f"Could not re-locate obstacle '{obstacle_label}'")
        return False

    if top_down_grasp(robot, obs_pos, z_offset=-0.015):
        place_obstacle(robot, target_pos)
        return True

    print(f"FAILED to remove obstacle '{obstacle_label}'")
    return False


def place_obstacle(robot, target_pos):
    """Place the currently held obstacle to the side of the target object."""
    place_pos = target_pos.copy()
    place_pos[1] += PLACE_OFFSET_Y  # Move to the side
    place_pos[2] += 0.02  # Slightly above table

    pre_place = place_pos.copy()
    pre_place[2] += PRE_GRASP_HEIGHT

    current_ori = robot.get_state()['robot_state']['orientation']

    robot.move_gripper_to(pre_place.tolist(), TOP_DOWN_RPY)
    robot.move_gripper_to(place_pos.tolist(), TOP_DOWN_RPY)
    robot.open_gripper()
    robot.move_gripper_to(pre_place.tolist(), TOP_DOWN_RPY)
    print("Obstacle placed to the side.")


def pick_up(robot, target_label, obstacle_label=None, enclosed=False):
    """
    Main pick-up routine. Optionally removes an obstacle first, then
    locates and picks up the target object.

    Strategy (learned from experiments):
    - If obstacle exists, remove it first
    - For enclosed objects (e.g., on a shelf): use side grasp directly
    - For open objects (e.g., on a table): try top-down, then planned grasp
    - Always re-perceive after failures

    Args:
        robot: The robot interface object
        target_label: String label for the object to pick up
        obstacle_label: Optional label for an obstacle to remove first
        enclosed: If True, object is in an enclosed space (shelf, cabinet).
            Uses side grasp + horizontal retreat instead of top-down.

    Returns:
        True if object was successfully picked up, False otherwise
    """
    print(f"=== Pick Up Task: '{target_label}' ===")
    if obstacle_label:
        print(f"    Obstacle to remove: '{obstacle_label}'")
    if enclosed:
        print(f"    Object is in enclosed space - using side grasp strategy")

    # Step 0: Remove obstacle if specified
    if obstacle_label:
        target_pos = locate_object(robot, target_label)
        if target_pos is None:
            print("FAILED: Could not locate target for obstacle placement reference.")
            return False

        if not remove_obstacle(robot, obstacle_label, target_pos):
            print("FAILED: Could not remove obstacle.")
            return False

        move_to_neutral(robot)
        # Rescan after obstacle removal (Exp 4: obstacle may have blocked depth sensing)
        robot.rescan_wrist()

    # Step 1: Locate the target object (re-perceive after obstacle removal)
    target_pos = locate_object(robot, target_label)
    if target_pos is None:
        print("FAILED: Could not locate target object.")
        return False

    # For enclosed objects: go directly to planned side grasp (Exp 3)
    if enclosed:
        if grasp_via_plan(robot, target_label,
                          f"grasp the {target_label} from the side",
                          hint_pos=target_pos,
                          retreat_before_lift=True):
            print(f"SUCCESS: Picked up '{target_label}' from enclosed space")
            return True

        # Retry once after returning to neutral
        print("Side grasp failed, retrying...")
        move_to_neutral(robot)
        target_pos = locate_object(robot, target_label)
        if target_pos is not None:
            if grasp_via_plan(robot, target_label,
                              f"grasp the {target_label} from the side",
                              hint_pos=target_pos,
                              retreat_before_lift=True):
                print(f"SUCCESS: Picked up '{target_label}' on retry")
                return True

        print(f"FAILED: Could not pick up '{target_label}' from enclosed space.")
        return False

    # Step 2: Attempt top-down grasp (for open/table objects)
    if top_down_grasp(robot, target_pos):
        print(f"SUCCESS: Picked up '{target_label}'")
        return True

    # Step 3: Recovery - return to neutral, re-perceive, retry
    print("First grasp attempt failed. Recovering...")
    move_to_neutral(robot)
    robot.open_gripper()

    target_pos = locate_object(robot, target_label)
    if target_pos is None:
        print("FAILED: Could not re-locate target after recovery.")
        return False

    if top_down_grasp(robot, target_pos):
        print(f"SUCCESS: Picked up '{target_label}' on retry")
        return True

    # Step 4: Fallback - use get_grasp_pose + plan_grasp (for handles etc.)
    print("Top-down grasp failed. Trying planned grasp as fallback...")
    move_to_neutral(robot)
    if grasp_via_plan(robot, target_label,
                      f"grasp the {target_label} to pick it up",
                      hint_pos=target_pos):
        print(f"SUCCESS: Picked up '{target_label}' via planned grasp")
        return True

    # Step 5: Last resort - full reset + re-perceive + top-down with z-offset
    # Learned from Exp 5: robot.reset() + 4cm z-offset saved the task
    print("All methods failed. Attempting full reset recovery...")
    safe_reset(robot)
    robot.rescan_wrist()

    target_pos = locate_object(robot, target_label)
    if target_pos is None:
        print(f"FAILED: Could not locate '{target_label}' after reset.")
        return False

    # Try with explicit z-offset of +4cm (proven fix from Exp 5)
    if top_down_grasp(robot, target_pos, z_offset=0.04):
        print(f"SUCCESS: Picked up '{target_label}' after reset with z-offset")
        return True

    print(f"FAILED: Could not pick up '{target_label}' after all attempts.")
    return False


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # Example usage:
    # pick_up(robot, "green cup", obstacle_label="yellow block")
    # pick_up(robot, "green mug")
    # pick_up(robot, "green mug", enclosed=True)  # Object on a shelf
    pass
