
import numpy as np

# Assume 'robot' object is available in the execution environment with methods like:
# robot.get_object_position_from_mask(object_name) -> {'3d_position': [...]} or None
# robot.open_right_gripper()
# robot.close_right_gripper()
# robot.move_right_arm_to(position, orientation, steps=20)
# robot.move_base(vx, vy, vyaw, steps)

def ensure_reachability(target_object_name: str, robot, max_attempts: int = 3, move_distance: float = 0.7, move_steps: int = 40) -> bool:
    """
    Checks if the target object is reachable and moves the robot base if not.
    Returns True if reachable (or made reachable), False otherwise.
    """
    for attempt in range(max_attempts):
        print(f"[Reachability] Attempt {attempt + 1}/{max_attempts} to ensure reachability for {target_object_name}...")
        target_result = robot.get_object_position_from_mask(target_object_name)
        if target_result is not None:
            print(f"[Reachability] Found {target_object_name} at {target_result['3d_position']}. Assuming reachable.")
            return True
        else:
            print(f"[Reachability] Could not locate {target_object_name}. Moving robot base to improve reachability (vx={move_distance}, vy=0.0, vyaw=0.0, steps={move_steps}).")
            # Ensure vy and vyaw are always passed, defaulting to 0.0 for simple forward movement
            robot.move_base(vx=move_distance, vy=0.0, vyaw=0.0, steps=move_steps)

    print(f"[Reachability] Failed to ensure reachability for {target_object_name} after {max_attempts} attempts. Aborting.")
    return False

def remove_occluding_object(occluding_object_name: str, target_object_pos: np.ndarray, robot) -> bool:
    """
    Removes an occluding object from on top of the target object.
    Returns True if successful or no occluding object was found, False otherwise.
    """
    if not occluding_object_name: # Handle cases where there is no occluding object
        print("[Occlusion] No occluding object specified. Skipping removal.")
        return True

    print(f"[Occlusion] Attempting to remove occluding object: {occluding_object_name}")
    occluder_result = robot.get_object_position_from_mask(occluding_object_name)

    if occluder_result is None:
        print(f"[Occlusion] No {occluding_object_name} found to remove. Assuming it's already clear or not present.")
        return True # Nothing to remove, so consider it successful

    occluder_pos = np.array(occluder_result['3d_position'])
    print(f"[Occlusion] Found {occluding_object_name} at {occluder_pos}")

    # Define pre-grasp and grasp positions for the occluder.
    pre_grasp_occluder = occluder_pos + [0, 0, 0.15] # 15 cm above
    grasp_ori = [np.pi, 0.0, 0.0] # Top-down orientation

    # Move right arm to pre-grasp, grasp, lift, move aside, and release.
    robot.open_right_gripper()
    robot.move_right_arm_to(pre_grasp_occluder.tolist(), grasp_ori)
    robot.move_right_arm_to(occluder_pos.tolist(), grasp_ori)
    robot.close_right_gripper()
    robot.move_right_arm_to(pre_grasp_occluder.tolist(), grasp_ori)

    # Define a placement location for the occluder (e.g., to the side of original target).
    # This assumes a flat surface for placement, adjusting z to a safe counter height.
    place_pos_occluder = target_object_pos + [0.20, 0, 0] # Move 20cm to the side
    # Ensure placement height is reasonable, e.g., on a counter surface or slightly above the target.
    place_pos_occluder[2] = max(0.8, target_object_pos[2] + 0.1) # Minimum 0.8m or 10cm above target
    robot.move_right_arm_to(place_pos_occluder.tolist(), grasp_ori)
    robot.open_right_gripper()
    print(f"[Occlusion] Successfully removed {occluding_object_name} and placed it at {place_pos_occluder}.")
    return True

def pick_up_object(object_name: str, robot) -> bool:
    """
    Picks up the specified object.
    Returns True if successful, False otherwise.
    """
    print(f"[Pickup] Attempting to pick up: {object_name}")
    object_result = robot.get_object_position_from_mask(object_name)

    if object_result is None:
        print(f"[Pickup] Error: Could not locate {object_name} to pick up.")
        return False

    obj_pos = np.array(object_result['3d_position'])
    print(f"[Pickup] Found {object_name} at {obj_pos}")

    # Define pre-grasp and grasp positions.
    pre_grasp_obj = obj_pos + [0, 0, 0.15] # 15 cm above
    grasp_ori = [np.pi, 0.0, 0.0] # Top-down orientation

    # Move right arm to pre-grasp, grasp, and lift.
    robot.open_right_gripper()
    robot.move_right_arm_to(pre_grasp_obj.tolist(), grasp_ori)
    robot.move_right_arm_to(obj_pos.tolist(), grasp_ori)
    robot.close_right_gripper()

    lift_pos = obj_pos + [0, 0, 0.2] # Lift 20 cm above original position
    robot.move_right_arm_to(lift_pos.tolist(), grasp_ori)
    print(f"[Pickup] Successfully picked up {object_name}.")
    return True

def execute_pick_up_with_occlusion_handling(robot, target_object_name: str, occluding_object_name: str = None) -> bool:
    """
    Executes the task of picking up a target object,
    optionally handling an occluding object.
    The robot base will attempt to move closer if the target is not initially reachable.
    Returns True if the task is completed successfully, False otherwise.
    """
    print(f"[Main Task] Starting task: Pick up {target_object_name} (Occluding object: {occluding_object_name or 'None'})")

    # Step 1: Ensure robot can reach the target area
    if not ensure_reachability(target_object_name, robot):
        print(f"[Main Task] Failed to ensure reachability for {target_object_name}. Aborting.")
        return False

    # Get initial position of the target object (after potential base movement)
    target_result = robot.get_object_position_from_mask(target_object_name)
    if target_result is None:
        print(f"[Main Task] Error: {target_object_name} not found even after reachability check. Aborting.")
        return False
    initial_target_pos = np.array(target_result['3d_position'])

    # Step 2: Remove occluding object if present and specified
    if occluding_object_name:
        if not remove_occluding_object(occluding_object_name, initial_target_pos, robot):
            print(f"[Main Task] Failed to remove occluding object: {occluding_object_name}. Aborting.")
            return False

    # Step 3: Pick up the target object
    # Re-check reachability and object position in case of significant scene changes
    # after occluder removal, or if the robot moved.
    if not ensure_reachability(target_object_name, robot):
        print(f"[Main Task] Failed to re-ensure reachability for {target_object_name} after occlusion handling. Aborting.")
        return False

    if not pick_up_object(target_object_name, robot):
        print(f"[Main Task] Failed to pick up {target_object_name}. Aborting.")
        return False

    print(f"[Main Task] Task completed: Picked up {target_object_name} with occlusion handling.")
    return True
