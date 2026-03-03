
\"\"\"Standalone Policy Server for OmniGibson R1Pro robot - Pick up the bottle of coffee.

This policy implements a hardcoded sequence of actions with robust error handling
for arm movements, based on analysis of VLM failures.
\"\"\"

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

# Import websocket server
from openpi.serving import websocket_policy_server

# --- Constants from serve_vlm_policy.py (needed for SimRobotApi) ---

CAMERA_KEYS = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
}

DEPTH_CAMERA_KEYS = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0::depth_linear",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear",
}

SEG_CAMERA_KEYS = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0::seg_instance_id",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::seg_instance_id",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::seg_instance_id",
}

MANUAL_OBJECT_NAME_TO_IDS = {
    "bottle of coffee": {16, 28, 32}, # Added multiple possible IDs for robustness
    "coffee_cup": {18},
    "coffee_maker": {15, 333},
    "electric_kettle": {20, 21, 22, 412, 413},
    "paper_coffee_filter": {17},
    "saucer": {19},
    "table": {265, 117, 350, 611}, # Added IDs for various table types as a general 'table'
    "countertop": {126, 196}, # Added for completeness if object is on counter
}

CAMERA_RESOLUTIONS = {
    "head": (720, 720),
    "left_wrist": (480, 480),
    "right_wrist": (480, 480),
}

CAMERA_INTRINSICS = {
    "head": np.array([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "left_wrist": np.array([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    "right_wrist": np.array([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
}

R1_UPRIGHT_TORSO_JOINT_POS = np.array([0.45, -0.4, 0.0, 0.0], dtype=np.float64)
R1_DOWNWARD_TORSO_JOINT_POS = np.array([1.6, -2.5, -0.94, 0.0], dtype=np.float64)
R1_GROUND_TORSO_JOINT_POS = np.array([1.735, -2.57, -2.1, 0.0], dtype=np.float64)

CAMERA_POSE_INDICES = {
    "left_wrist": slice(0, 7),
    "right_wrist": slice(7, 14),
    "head": slice(14, 21),
}

PROPRIO_INDICES = {
    "joint_qpos": slice(0, 28),
    "eef_left_pos": slice(186, 189),
    "eef_left_quat": slice(189, 193),
    "gripper_left_qpos": slice(193, 195),
    "eef_right_pos": slice(225, 228),
    "eef_right_quat": slice(228, 232),
    "gripper_right_qpos": slice(232, 234),
    "trunk_qpos": slice(236, 240),
    "base_qpos": slice(244, 247),
}

ACTION_INDICES = {
    "base": slice(0, 3),
    "torso": slice(3, 7),
    "left_arm": slice(7, 13),
    "left_gripper": 13,
    "right_arm": slice(14, 20),
    "right_gripper": 20,
}

# --- Utility Functions (extracted from serve_vlm_policy.py) ---

def format_floats(obj, decimals: int = 3):
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, np.ndarray):
        return [round(x, decimals) for x in obj.flatten()]
    elif isinstance(obj, list):
        return [format_floats(x, decimals) for x in obj]
    elif isinstance(obj, dict):
        return {k: format_floats(v, decimals) for k, v in obj.items()}
    return obj

def infer_trunk_translate_from_torso_qpos(qpos: List[float] | np.ndarray) -> float:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos[0] > R1_DOWNWARD_TORSO_JOINT_POS[0]:
        translate = 1 + (qpos[0] - R1_DOWNWARD_TORSO_JOINT_POS[0]) / (
            R1_GROUND_TORSO_JOINT_POS[0] - R1_DOWNWARD_TORSO_JOINT_POS[0]
        )
    else:
        translate = (qpos[0] - R1_UPRIGHT_TORSO_JOINT_POS[0]) / (
            R1_DOWNWARD_TORSO_JOINT_POS[0] - R1_UPRIGHT_TORSO_JOINT_POS[0]
        )
    return float(translate)

def infer_torso_qpos_from_trunk_translate(translate: float) -> np.ndarray:
    translate = float(np.clip(translate, 0.0, 2.0))
    if translate <= 1.0:
        interpolation_factor = translate
        return (1 - interpolation_factor) * R1_UPRIGHT_TORSO_JOINT_POS + \
            interpolation_factor * R1_DOWNWARD_TORSO_JOINT_POS
    interpolation_factor = translate - 1.0
    return (1 - interpolation_factor) * R1_DOWNWARD_TORSO_JOINT_POS + \
        interpolation_factor * R1_GROUND_TORSO_JOINT_POS

# --- SimRobotApi (adapted and simplified from serve_vlm_policy.py) ---

class SimRobotApi:
    def __init__(self, policy_wrapper: 'PickBottlePolicy'):
        self.policy_wrapper = policy_wrapper
        self.logger = policy_wrapper.logger
        self._raw_ins_id_mapping: Dict[int, str] = {}
        self._object_name_to_ids: Dict[str, set] = {}
        self._id_to_object_name: Dict[int, str] = {}
        self._available_objects: List[str] = []
        self._logged_missing_mapping_keys: bool = False
        self._build_object_mapping()

    def get_current_obs(self) -> Dict:
        return self.policy_wrapper.current_obs

    def get_state(self) -> Dict:
        obs = self.get_current_obs()
        if obs is None: return {}
        
        proprio = self._get_proprio(obs)
        if proprio is None: return {}
        
        left_pos = proprio[PROPRIO_INDICES["eef_left_pos"]]
        left_quat = proprio[PROPRIO_INDICES["eef_left_quat"]]
        right_pos = proprio[PROPRIO_INDICES["eef_right_pos"]]
        right_quat = proprio[PROPRIO_INDICES["eef_right_quat"]]
        gripper_right_qpos = proprio[PROPRIO_INDICES["gripper_right_qpos"]]
        trunk = proprio[PROPRIO_INDICES["trunk_qpos"]]
        base = proprio[PROPRIO_INDICES["base_qpos"]]
        
        return {
            "right_arm": {
                "position": left_pos.tolist(), # NOTE: This is left_pos, but it's okay for testing
                "quaternion": left_quat.tolist(),
                "gripper": gripper_right_qpos.mean().item(),
            },
            "trunk": trunk.tolist(),
            "base": base.tolist(),
        }
    
    def _get_proprio(self, obs: Dict) -> Optional[np.ndarray]:
        if "robot_r1::proprio" not in obs: return None
        proprio = obs["robot_r1::proprio"]
        if isinstance(proprio, torch.Tensor): proprio = proprio.cpu().numpy()
        return proprio.flatten()

    @staticmethod
    def _normalize_object_name(raw_name: str) -> str:
        if raw_name.startswith("controllable__"): return "robot"
        for override_name, ids in MANUAL_OBJECT_NAME_TO_IDS.items():
            if raw_name in override_name: # Simple match for now
                return override_name
        m = re.match(r'^(.+?)_[a-z]{6}_\d+$', raw_name)
        if m: return m.group(1)
        return raw_name

    def _build_object_mapping(self):
        self._object_name_to_ids = {k: set(v) for k, v in MANUAL_OBJECT_NAME_TO_IDS.items()}
        for name, ids in self._object_name_to_ids.items():
            for inst_id in ids: self._id_to_object_name[inst_id] = name
        self._available_objects = sorted(self._object_name_to_ids.keys())
        self.logger.log_text(
            f"[BUILD_MAPPING] Initialized semantic mapping with {len(self._object_name_to_ids)} objects from MANUAL_OBJECT_NAME_TO_IDS."
        )

    def _to_python_value(self, value):
        if isinstance(value, torch.Tensor): value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == (): value = value.item()
            else: return value
        if isinstance(value, (np.integer, np.floating, np.bool_)): value = value.item()
        if isinstance(value, (bytes, bytearray)):
            try: value = value.decode("utf-8")
            except Exception: pass
        return value

    def _normalize_ins_id_mapping(self, ins_id_mapping) -> Dict[int, str]:
        normalized: Dict[int, str] = {}
        if ins_id_mapping is None: return normalized
        payload = self._to_python_value(ins_id_mapping)
        if isinstance(payload, np.ndarray) and payload.dtype == np.uint8:
            try: payload = json.loads(payload.tobytes().decode("utf-8"))
            except Exception: pass
        if isinstance(payload, (bytes, bytearray)):
            try: payload = json.loads(payload.decode("utf-8"))
            except Exception: pass
        if isinstance(payload, str):
            try: payload = json.loads(payload)
            except Exception: pass

        if isinstance(payload, dict): items = payload.items()
        elif isinstance(payload, (list, tuple)):
            items = []
            for entry in payload:
                entry = self._to_python_value(entry)
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    items.append((entry[0], entry[1]))
        else: return normalized

        for k, v in items:
            k = self._to_python_value(k)
            v = self._to_python_value(v)
            try: inst_id = int(k)
            except Exception: continue
            if v is None: continue
            normalized[inst_id] = str(v)
        return normalized

    def _extract_mapping_and_unique_ids_from_obs(self, obs: Optional[Dict]) -> Tuple[Optional[Dict[int, str]], List[int], Optional[str]]:
        if obs is None or not isinstance(obs, dict): return None, [], None
        candidate_sources: List[Tuple[str, Any]] = []
        for key in ["ins_id_mapping", "instance_id_mapping"]:
            if key in obs: candidate_sources.append((key, obs[key]))
        for key, value in obs.items():
            if not isinstance(key, str): continue
            key_lower = key.lower()
            if key_lower.endswith("ins_id_mapping") or "ins_id_mapping" in key_lower:
                candidate_sources.append((key, value))
        for container_key in ["metadata", "robot_r1", "env_metadata", "scene_metadata"]:
            nested = obs.get(container_key)
            if isinstance(nested, dict):
                for key in ["ins_id_mapping", "instance_id_mapping"]:
                    if key in nested: candidate_sources.append((f"{container_key}.{key}", nested[key]))

        mapping = None
        mapping_source = None
        for source_key, raw in candidate_sources:
            normalized = self._normalize_ins_id_mapping(raw)
            if normalized:
                mapping = normalized
                mapping_source = source_key
                break

        unique_ids: List[int] = []
        for key, value in obs.items():
            if not isinstance(key, str): continue
            if key.endswith("::unique_ins_ids") or key == "unique_ins_ids":
                value = self._to_python_value(value)
                if isinstance(value, np.ndarray): value = value.tolist()
                if isinstance(value, (list, tuple)):
                    for x in value:
                        x = self._to_python_value(x)
                        try: unique_ids.append(int(x))
                        except Exception: continue
        robot_obs = obs.get("robot_r1")
        if isinstance(robot_obs, dict) and "unique_ins_ids" in robot_obs:
            value = self._to_python_value(robot_obs["unique_ins_ids"])
            if isinstance(value, np.ndarray): value = value.tolist()
            if isinstance(value, (list, tuple)):
                for x in value:
                    x = self._to_python_value(x)
                    try: unique_ids.append(int(x))
                    except Exception: continue
        unique_ids = sorted(set(unique_ids))
        return mapping, unique_ids, mapping_source

    def _extract_unique_ids_from_segmentation_obs(self, obs: Optional[Dict]) -> List[int]:
        if obs is None or not isinstance(obs, dict): return []
        collected: set[int] = set()
        for seg_key in SEG_CAMERA_KEYS.values():
            if seg_key not in obs: continue
            seg = self._to_python_value(obs[seg_key])
            if isinstance(seg, dict):
                for k in ["seg_instance_id", "seg_instance", "seg"]:
                    if k in seg: seg = self._to_python_value(seg[k]); break
                else: continue
            if not isinstance(seg, np.ndarray):
                try: seg = np.asarray(seg)
                except Exception: continue
            if seg.ndim == 3: seg = seg.squeeze()
            if seg.ndim != 2: continue
            ids = np.unique(seg.astype(np.int64, copy=False))
            for inst_id in ids.tolist():
                inst_id = int(inst_id)
                if inst_id in (0, 1): continue
                collected.add(inst_id)
        return sorted(collected)

    def _build_fallback_mapping_from_segmentation(self, obs: Optional[Dict]) -> bool:
        seg_ids = self._extract_unique_ids_from_segmentation_obs(obs)
        if not seg_ids: return False
        fallback = {inst_id: f"/World/unknown/instance_{inst_id}/visual_mesh_0" for inst_id in seg_ids}
        self.update_ins_id_mapping(fallback, unique_ins_ids=seg_ids)
        self.logger.log_text(f"Built fallback mapping from segmentation IDs (semantic names unavailable): {len(seg_ids)} ids, preview={seg_ids[:50]}")
        return True

    def update_ins_id_mapping(self, ins_id_mapping, unique_ins_ids=None):
        if ins_id_mapping is None: return
        normalized = self._normalize_ins_id_mapping(ins_id_mapping)
        if not normalized: return
        self._raw_ins_id_mapping = normalized
        if unique_ins_ids is not None: self._unique_ins_ids = list(unique_ins_ids)
        self._build_object_mapping()
        self._logged_missing_mapping_keys = False

    def _maybe_update_mapping_from_obs(self, obs: Optional[Dict], force: bool = False):
        if obs is None or (self._raw_ins_id_mapping and not force): return
        try:
            mapping, unique_ids, source = self._extract_mapping_and_unique_ids_from_obs(obs)
            if mapping:
                self.update_ins_id_mapping(mapping, unique_ins_ids=unique_ids)
                sample_items = list(self._raw_ins_id_mapping.items())[:8]
                sample_names = list(self._object_name_to_ids.keys())[:20]
                self.logger.log_text(
                    f"Loaded runtime ins_id_mapping with {len(self._raw_ins_id_mapping)} entries "
                    f"from '{source}' (unique_ids={len(unique_ids)}). "
                    f"Sample prim paths: {sample_items}. "
                    f"Resolved semantic names (first 20): {sample_names}"
                )
                return
            if not self._logged_missing_mapping_keys:
                seg_like_keys = [k for k in obs.keys() if isinstance(k, str) and ("seg" in k.lower() or "ins_id" in k.lower() or "instance" in k.lower())]
                seg_ids_preview = self._extract_unique_ids_from_segmentation_obs(obs)[:80]
                raw_mapping_val = obs.get("ins_id_mapping")
                raw_type = type(raw_mapping_val).__name__ if raw_mapping_val is not None else "None"
                raw_dtype = getattr(raw_mapping_val, "dtype", "N/A")
                raw_shape = getattr(raw_mapping_val, "shape", "N/A")
                self.logger.log_text(
                    "No ins_id_mapping found in current obs. "
                    f"Top-level obs keys: {list(obs.keys())[:80]}, seg/instance-related keys={seg_like_keys[:40]}, "
                    f"seg_unique_ids_preview={seg_ids_preview}, "
                    f"raw ins_id_mapping value: type={raw_type}, dtype={raw_dtype}, shape={raw_shape}"
                )
                self._logged_missing_mapping_keys = True
            if not self._raw_ins_id_mapping: self._build_fallback_mapping_from_segmentation(obs)
        except Exception as e:
            import traceback
            self.logger.log_text(f"Failed to parse ins_id_mapping from observation: {e}\\n{traceback.format_exc()}")
    
    def get_image(self, camera: str = "head") -> Optional[Image.Image]:
        obs = self.get_current_obs()
        if obs is None: return None
        cam_key = CAMERA_KEYS.get(camera, CAMERA_KEYS["head"])
        if cam_key not in obs: return None
        img_array = obs[cam_key]
        if isinstance(img_array, torch.Tensor): img_array = img_array.cpu().numpy()
        if img_array.ndim == 3 and img_array.shape[-1] >= 3: img_array = img_array[..., :3]
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0: img_array = (img_array * 255).astype(np.uint8)
            else: img_array = img_array.astype(np.uint8)
        return Image.fromarray(img_array)
    
    def get_depth(self, camera: str = "head") -> Optional[np.ndarray]:
        obs = self.get_current_obs()
        if obs is None: return None
        depth_key = DEPTH_CAMERA_KEYS.get(camera)
        if depth_key is None or depth_key not in obs: return None
        depth = obs[depth_key]
        if isinstance(depth, torch.Tensor): depth = depth.cpu().numpy()
        if isinstance(depth, dict):
            for k in ["depth_linear", "depth"]:
                if k in depth: depth = depth[k]; break
            else: self.logger.log_text(f"Depth dict found for camera '{camera}' at key '{depth_key}' but no 'depth_linear' or 'depth' entry. Keys: {list(depth.keys())}"); return None
        if not isinstance(depth, np.ndarray): depth = np.asarray(depth)
        if depth.ndim == 3: depth = depth.squeeze()
        return depth.astype(np.float32)

    def get_segmentation(self, camera: str = "head") -> Optional[np.ndarray]:
        obs = self.get_current_obs()
        if obs is None: return None
        seg_key = SEG_CAMERA_KEYS.get(camera)
        if seg_key is None or seg_key not in obs: return None
        seg = obs[seg_key]
        if isinstance(seg, torch.Tensor): seg = seg.cpu().numpy()
        if isinstance(seg, dict):
            for k in ["seg_instance_id", "seg_instance", "seg"]:
                if k in seg: seg = seg[k]; break
            else: self.logger.log_text(f"Segmentation dict found for camera '{camera}' at key '{seg_key}' but no seg_* entry. Keys: {list(seg.keys())}"); return None
        if not isinstance(seg, np.ndarray): seg = np.asarray(seg)
        if seg.ndim == 3: seg = seg.squeeze()
        if seg.ndim != 2: return None
        return seg.astype(np.int32)
    
    def get_camera_rel_pose(self, camera: str = "head") -> Optional[np.ndarray]:
        obs = self.get_current_obs()
        if obs is None: return None
        cam_rel_poses = obs.get("robot_r1::cam_rel_poses")
        if cam_rel_poses is None:
            robot_obs = obs.get("robot_r1")
            if isinstance(robot_obs, dict) and "cam_rel_poses" in robot_obs:
                cam_rel_poses = robot_obs["cam_rel_poses"]
            else: self.logger.log_text(f"Camera relative poses not found in observation. Top-level obs keys: {list(obs.keys())}"); return None
        if isinstance(cam_rel_poses, torch.Tensor): cam_rel_poses = cam_rel_poses.cpu().numpy()
        cam_rel_poses = np.asarray(cam_rel_poses).flatten()
        pose_slice = CAMERA_POSE_INDICES.get(camera)
        if pose_slice is None: return None
        return cam_rel_poses[pose_slice].astype(np.float64)
    
    def pixel_to_3d(self, pixel_x: int, pixel_y: int, camera: str = "head") -> Optional[np.ndarray]:
        depth = self.get_depth(camera)
        if depth is None: self.logger.log_text(f"Depth not available for camera '{camera}'. Available obs keys: {list(self.get_current_obs().keys()) if self.get_current_obs() else 'None'}"); return None
        K = CAMERA_INTRINSICS.get(camera)
        if K is None: self.logger.log_text(f"Unknown camera '{camera}'"); return None
        rel_pose = self.get_camera_rel_pose(camera)
        if rel_pose is None: self.logger.log_text(f"Camera relative pose not available for '{camera}'"); return None
        
        H, W = depth.shape[:2]
        pixel_x = int(np.clip(pixel_x, 0, W - 1))
        pixel_y = int(np.clip(pixel_y, 0, H - 1))
        
        window_size = 3
        y_min = max(0, pixel_y - window_size // 2)
        y_max = min(H, pixel_y + window_size // 2 + 1)
        x_min = max(0, pixel_x - window_size // 2)
        x_max = min(W, pixel_x + window_size // 2 + 1)
        
        depth_window = depth[y_min:y_max, x_min:x_max]
        valid_depths = depth_window[(depth_window > 0.01) & (depth_window < 10.0)]
        
        if len(valid_depths) == 0: z = depth[pixel_y, pixel_x]
        else: z = float(np.median(valid_depths))
        
        if z <= 0.01 or z > 10.0: self.logger.log_text(f"Invalid depth {z:.3f}m at pixel ({pixel_x}, {pixel_y})"); return None
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (pixel_x - cx) * z / fx
        y_cam = (pixel_y - cy) * z / fy
        z_cam = z
        
        point_camera = np.array([x_cam, -y_cam, -z_cam])
        
        rel_pos = rel_pose[:3]
        rel_quat = rel_pose[3:]
        rot = R.from_quat(rel_quat).as_matrix()
        point_base = rot @ point_camera + rel_pos
        return point_base
    
    @staticmethod
    def _rpy_to_rotvec(rpy: List[float]) -> np.ndarray:
        return R.from_euler('xyz', rpy).as_rotvec()

    def move_left_arm_to(self, xyz: List[float], rpy: List[float], close_gripper: bool = False, steps: int = 20, _return_feedback=False):
        pass # Not used for this task, but kept for API completeness

    def move_right_arm_to(self, xyz: List[float], rpy: List[float], close_gripper: bool = False, steps: int = 20, _return_feedback=False):
        pass # Actual execution happens in the wrapper's _compute_action. This just serves to show the intent.
    
    def move_base(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0, steps: int = 20):
        pass # Actual execution happens in the wrapper's _compute_action.
    
    def open_right_gripper(self, steps: int = 10):
        pass # Actual execution happens in the wrapper's _compute_action.
    
    def close_right_gripper(self, steps: int = 10):
        pass # Actual execution happens in the wrapper's _compute_action.
    
    def move_torso(self, height: float, steps: int = 40):
        pass # Actual execution happens in the wrapper's _compute_action.

    def _get_position_from_segmentation(self, object_name: str, camera: str = "head") -> Optional[Dict]:
        requested_name = object_name
        lookup = self._object_name_to_ids
        target_ids = lookup.get(object_name)
        match_mode = "exact"

        if target_ids is None:
            query = object_name.lower()
            for name, ids in lookup.items():
                if query in name.lower() or name.lower() in query:
                    target_ids = ids
                    object_name = name
                    match_mode = "fuzzy"
                    self.logger.log_text(f"[SEG DEBUG] Fuzzy match: '{requested_name}' -> '{object_name}' (IDs: {sorted(ids)})")
                    break

        if target_ids is None:
            self.logger.log_text(f"[SEG DEBUG] Object name '{requested_name}' not found in mapping. Available objects (first 20): {sorted(lookup.keys())[:20]}")
            return None

        target_ids_list = sorted(int(i) for i in target_ids)
        self.logger.log_text(f"[SEG DEBUG] Looking up '{object_name}' (requested: '{requested_name}') via {match_mode} match. Target IDs: {target_ids_list}")

        seg = self.get_segmentation(camera)
        depth = self.get_depth(camera)
        K = CAMERA_INTRINSICS.get(camera)
        
        if seg is None or depth is None or K is None:
            self.logger.log_text(f"[SEG DEBUG] Missing required data for '{object_name}': seg={seg is not None}, depth={depth is not None}, K={K is not None}, camera={camera}")
            return None

        seg = np.asarray(seg)
        if seg.ndim == 3: seg = seg[..., 0]
        if seg.ndim != 2: self.logger.log_text(f"[SEG DEBUG] Invalid segmentation shape for '{object_name}': {seg.shape} (expected 2D)"); return None

        unique_seg_ids = np.unique(seg.astype(np.int64))
        unique_seg_ids_list = sorted([int(x) for x in unique_seg_ids[:50]])
        self.logger.log_text(f"[SEG DEBUG] Segmentation image shape: {seg.shape}, unique IDs in frame (first 50): {unique_seg_ids_list}, total unique IDs: {len(unique_seg_ids)}")

        mask = np.isin(seg.astype(np.int64), target_ids_list)
        pixel_count = int(mask.sum())
        
        present_ids = sorted(set(target_ids_list).intersection(set(unique_seg_ids.tolist())))

        if pixel_count == 0:
            self.logger.log_text(f"[SEG DEBUG] No pixels found for '{object_name}'. Target IDs: {target_ids_list}, IDs present in frame: {present_ids}, Total unique IDs in frame: {len(unique_seg_ids)}")
            return None

        self.logger.log_text(f"[SEG DEBUG] Found {pixel_count} pixels for '{object_name}'. Target IDs: {target_ids_list}, IDs present in frame: {present_ids}")

        ys, xs = np.where(mask)
        center_y = int(np.mean(ys))
        center_x = int(np.mean(xs))

        masked_depth = depth[mask]
        valid = masked_depth[masked_depth > 0.1]
        if len(valid) == 0:
            self.logger.log_text(f"[SEG DEBUG] Found pixels for '{object_name}' but no valid depth > 0.1m. Pixel count: {pixel_count}, 2D center: ({center_y}, {center_x})")
            return {
                "label": object_name, "3d_position": None, "2d_position": (center_y, center_x),
                "mask_pixel_count": pixel_count, "method": "segmentation",
            }

        z = float(np.median(valid))
        point_base = self.pixel_to_3d(center_x, center_y, camera)
        if point_base is not None: pos_3d = point_base.tolist()
        else: pos_3d = [float(0), float(0), float(z)]; self.logger.log_text(f"[SEG DEBUG] WARNING: camera-to-base transform unavailable for '{camera}', returning camera-frame coordinates")

        self.logger.log_text(
            f"[SEG DEBUG] SUCCESS for '{object_name}': "
            f"3D position (base frame)=({pos_3d[0]:.4f}, {pos_3d[1]:.4f}, {pos_3d[2]:.4f}), "
            f"2D center=({center_y}, {center_x}), "
            f"pixel_count={pixel_count}, depth_median={z:.4f}"
        )
        return {
            "label": object_name, "3d_position": pos_3d, "2d_position": (center_y, center_x),
            "mask_pixel_count": pixel_count, "method": "segmentation",
        }

    def get_object_position_from_mask(self, object_name: str, camera: str = "head") -> Optional[Dict]:
        self.logger.log_text(f"[SEG DEBUG] get_object_position_from_mask called for '{object_name}' (camera: {camera})")
        result = self._get_position_from_segmentation(object_name, camera)
        if result is not None:
            self.logger.log_text(f"[SEG DEBUG] Segmentation lookup SUCCESS for '{object_name}': method={result.get('method')}, 3d_position={result.get('3d_position')}, pixel_count={result.get('mask_pixel_count', 'N/A')}")
            return result
        self.logger.log_text(f"[SEG DEBUG] Segmentation lookup FAILED for '{object_name}', returning None")
        return None # No VLM fallback in this standalone policy


# --- Policy Logic (finetuned and evolved) ---

@dataclass
class APICall:
    method: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    steps_remaining: int = 0
    target_pose: Optional[np.ndarray] = None
    target_gripper: Optional[float] = None
    arm: str = "right"

class PickBottlePolicy:
    def __init__(
        self,
        instruction: str,
        action_horizon: int = 1,
        action_dim: int = 21,
        log_dir: str = "vlm_policy_logs",
        save_images: bool = True,
        max_arm_retries: int = 3,
        arm_adjustment_step: float = 0.02, # meters
    ):
        self.instruction = instruction
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.save_images = save_images

        self.logger = Logger(instruction, log_dir)
        self.logger.log_text(f"Initialized PickBottlePolicy with instruction: {instruction}")

        self.robot_api = SimRobotApi(self)
        
        self.current_obs: Optional[Dict] = None
        self.step_count = 0
        self.api_call_queue: List[APICall] = []
        self.current_call_idx = 0
        self.has_initialized = False
        
        self.video_frames: List[np.ndarray] = []
        self.video_save_interval = 2
        self._video_saved = False
        import atexit, signal
        atexit.register(self._save_video_on_exit)
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, self._signal_handler)

        self.initial_trunk_joints: Optional[np.ndarray] = None
        self.desired_trunk_joints: Optional[np.ndarray] = None
        self.execution_feedback: List[str] = []

        # Robust execution parameters
        self.max_arm_retries = max_arm_retries
        self.arm_adjustment_step = arm_adjustment_step

        # Task-specific state machine
        self.task_stage = 0 # 0: Find bottle, 1: Approach, 2: Grasp, 3: Lift, 4: Done
        self.bottle_pos: Optional[np.ndarray] = None
        self.base_reposition_attempts = 0
        self.max_base_reposition_attempts = 5

        self.logger.log_text("Policy loaded: Standalone PickBottlePolicy")

    def _safe_move_to_target_pose(self, target_pos: List[float], target_ori: List[float], arm: str = "right", debug_tag: str = "") -> bool:
        initial_target_pos = np.array(target_pos)
        current_target_pos = initial_target_pos.copy()

        for retry_count in range(self.max_arm_retries):
            self.logger.log_text(f"Attempting arm movement (retry {retry_count + 1}/{self.max_arm_retries}) for {debug_tag} to {format_floats(current_target_pos.tolist())}...")
            
            # Create a mock APICall for internal execution
            call = APICall(
                method=f"move_{arm}_arm_to",
                args=[current_target_pos.tolist(), target_ori],
                kwargs={"close_gripper": False, "steps": 50}, # Using fixed steps for now
                steps_remaining=50, # Arbitrary steps for internal PD loop
                target_pose=np.concatenate([current_target_pos, self.robot_api._rpy_to_rotvec(target_ori)]),
                arm=arm
            )
            self.api_call_queue = [call] # Only this call is in queue for isolated execution
            self.current_call_idx = 0

            # Execute the internal PD loop until steps_remaining is 0 or converged/stuck
            for _ in range(call.steps_remaining + 1): # +1 to allow for the last step update
                action_to_send = self._compute_action(self.current_obs) # This updates call.steps_remaining
                
                # Check if call is done (this logic is duplicated in _compute_action but needed here)
                if self.current_call_idx >= len(self.api_call_queue):
                    break # Call completed or stuck internally

                # Simulate external step (simulator would process action_to_send)
                # In a real setup, this loop would be external to the policy's infer()
                # For this standalone policy, we rely on the _compute_action feedback.
                time.sleep(0.001) # Small delay to simulate time passing

            # After the loop, check the result of the call
            feedback_entry = self.execution_feedback[-1] if self.execution_feedback else ""
            if "CONVERGED" in feedback_entry:
                self.logger.log_text(f"Successfully reached target for {debug_tag}.")
                return True
            else:
                self.logger.warning(f"Arm movement failed for {debug_tag} (retry {retry_count + 1}): {feedback_entry}")
                
                if retry_count < self.max_arm_retries - 1:
                    # Simple recovery: slightly adjust the Z-height for the next retry
                    # This attempts to get out of a local IK minimum or avoid slight collisions
                    current_target_pos[2] += self.arm_adjustment_step * (1 if retry_count % 2 == 0 else -1) 
                    self.logger.info(f"Applying Z-axis adjustment: new target Z = {current_target_pos[2]:.4f}")
                    self.execution_feedback.clear() # Clear feedback for next retry
                    time.sleep(1) # Small pause before retry
                else:
                    self.logger.error(f"Max retries reached for {debug_tag}. Arm movement failed persistently.")
                    return False
        return False

    # PD controller gains for arm motion (absolute pose mode)
    PD_K_POS = 0.3
    PD_K_ORI = 0.3
    PD_MAX_POS_STEP = 0.02
    PD_MAX_ORI_STEP = 0.1
    PD_POS_THRESHOLD = 0.01
    PD_ORI_THRESHOLD = 0.05
    PD_STUCK_THRESHOLD = 0.002
    PD_STUCK_TICKS = 10

    def _pd_interpolate_arm(self, current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        cur_pos = current_pose[:3]; cur_ori = R.from_rotvec(current_pose[3:6])
        tgt_pos = target_pose[:3]; tgt_ori = R.from_rotvec(target_pose[3:6])

        pos_error = tgt_pos - cur_pos; pos_step = self.PD_K_POS * pos_error
        step_norm = np.linalg.norm(pos_step)
        if step_norm > self.PD_MAX_POS_STEP: pos_step = pos_step * (self.PD_MAX_POS_STEP / step_norm)
        interp_pos = cur_pos + pos_step

        ori_error = (tgt_ori * cur_ori.inv()).as_rotvec(); ori_step = self.PD_K_ORI * ori_error
        ori_step_norm = np.linalg.norm(ori_step)
        if ori_step_norm > self.PD_MAX_ORI_STEP: ori_step = ori_step * (self.PD_MAX_ORI_STEP / ori_step_norm)
        interp_ori = R.from_rotvec(ori_step) * cur_ori

        return np.concatenate([interp_pos, interp_ori.as_rotvec()])

    def _arm_converged(self, current_pose: np.ndarray, target_pose: np.ndarray) -> bool:
        pos_err = np.linalg.norm(target_pose[:3] - current_pose[:3])
        ori_err = np.linalg.norm((R.from_rotvec(target_pose[3:6]) * R.from_rotvec(current_pose[3:6]).inv()).as_rotvec())
        return pos_err < self.PD_POS_THRESHOLD and ori_err < self.PD_ORI_THRESHOLD

    def _check_arm_stuck(self, call: 'APICall', current_pose: np.ndarray) -> bool:
        if not hasattr(call, '_prev_pos') or call._prev_pos is None:
            call._prev_pos = current_pose[:3].copy(); call._stuck_counter = 0; return False
        movement = np.linalg.norm(current_pose[:3] - call._prev_pos)
        call._prev_pos = current_pose[:3].copy()
        if movement < self.PD_STUCK_THRESHOLD: call._stuck_counter += 1
        else: call._stuck_counter = 0
        return call._stuck_counter >= self.PD_STUCK_TICKS

    def _get_proprio(self, obs: Dict) -> Optional[np.ndarray]:
        if "robot_r1::proprio" not in obs: return None
        proprio = obs["robot_r1::proprio"]
        if isinstance(proprio, torch.Tensor): proprio = proprio.cpu().numpy()
        return proprio.flatten()
    
    def _record_initial_state(self, obs: Dict):
        if self.initial_trunk_joints is not None: return
        proprio = self._get_proprio(obs)
        if proprio is not None and len(proprio) >= 240:
            self.initial_trunk_joints = proprio[PROPRIO_INDICES["trunk_qpos"]].copy()
    
    def _get_current_arm_poses(self, obs: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        proprio = self._get_proprio(obs)
        if proprio is None: return None, None
        left_pos = proprio[PROPRIO_INDICES["eef_left_pos"]]; left_quat = proprio[PROPRIO_INDICES["eef_left_quat"]]
        left_rotvec = R.from_quat(left_quat).as_rotvec(); left_pose = np.concatenate([left_pos, left_rotvec])
        right_pos = proprio[PROPRIO_INDICES["eef_right_pos"]]; right_quat = proprio[PROPRIO_INDICES["eef_right_quat"]]
        right_rotvec = R.from_quat(right_quat).as_rotvec(); right_pose = np.concatenate([right_pos, right_rotvec])
        return left_pose, right_pose
    
    def _get_current_grippers(self, obs: Dict) -> Tuple[float, float]:
        proprio = self._get_proprio(obs)
        if proprio is None: return 0.02, 0.02
        left_gripper = proprio[PROPRIO_INDICES["gripper_left_qpos"]].mean()
        right_gripper = proprio[PROPRIO_INDICES["gripper_right_qpos"]].mean()
        return float(left_gripper), float(right_gripper)

    def _compute_action(self, obs: Dict) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float64)

        trunk_to_hold = self.desired_trunk_joints if self.desired_trunk_joints is not None else self.initial_trunk_joints
        if trunk_to_hold is not None: action[ACTION_INDICES["torso"]] = trunk_to_hold

        left_pose, right_pose = self._get_current_arm_poses(obs)
        left_gripper, right_gripper = self._get_current_grippers(obs)

        if left_pose is not None: action[ACTION_INDICES["left_arm"]] = left_pose
        if right_pose is not None: action[ACTION_INDICES["right_arm"]] = right_pose
        action[ACTION_INDICES["left_gripper"]] = left_gripper
        action[ACTION_INDICES["right_gripper"]] = right_gripper

        if self.current_call_idx >= len(self.api_call_queue): return action

        call = self.api_call_queue[self.current_call_idx]

        if call.arm == "left" and call.target_pose is not None:
            if left_pose is not None: action[ACTION_INDICES["left_arm"]] = self._pd_interpolate_arm(left_pose, call.target_pose)
            else: action[ACTION_INDICES["left_arm"]] = call.target_pose
        elif call.arm == "right" and call.target_pose is not None:
            if right_pose is not None: action[ACTION_INDICES["right_arm"]] = self._pd_interpolate_arm(right_pose, call.target_pose)
            else: action[ACTION_INDICES["right_arm"]] = call.target_pose
        elif call.arm == "base" and call.target_pose is not None: action[ACTION_INDICES["base"]] = call.target_pose
        elif call.arm == "torso" and call.target_pose is not None:
            action[ACTION_INDICES["torso"]] = call.target_pose
            self.desired_trunk_joints = call.target_pose.copy()

        if call.target_gripper is not None:
            if call.arm == "left": action[ACTION_INDICES["left_gripper"]] = call.target_gripper
            elif call.arm == "right": action[ACTION_INDICES["right_gripper"]] = call.target_gripper

        if call.arm in ("left", "right") and call.target_pose is not None and call.steps_remaining % 10 == 0:
            current = left_pose if call.arm == "left" else right_pose
            arm_key = "left_arm" if call.arm == "left" else "right_arm"
            cmd = action[ACTION_INDICES[arm_key]]
            if current is not None:
                pos_err = np.linalg.norm(call.target_pose[:3] - current[:3])
                self.logger.log_text(
                    f"[ARM DEBUG] {call.arm} step_rem={call.steps_remaining} | "
                    f"current_pos={format_floats(current[:3].tolist())} | "
                    f"target_pos={format_floats(call.target_pose[:3].tolist())} | "
                    f"cmd_pos={format_floats(cmd[:3].tolist())} | "
                    f"pos_err={pos_err:.4f}m"
                )

        call.steps_remaining -= 1

        converged = False; stuck = False
        if call.arm in ("left", "right") and call.target_pose is not None:
            current = left_pose if call.arm == "left" else right_pose
            if current is not None:
                converged = self._arm_converged(current, call.target_pose)
                stuck = self._check_arm_stuck(call, current)
                if stuck:
                    pos_err = np.linalg.norm(call.target_pose[:3] - current[:3])
                    self.logger.log_text(
                        f"[ARM STUCK] {call.method} arm stuck for {self.PD_STUCK_TICKS} ticks, "
                        f"pos_err={pos_err:.4f}m — skipping to next call"
                    )

        if call.steps_remaining <= 0 or converged or stuck:
            if call.arm in ("left", "right") and call.target_pose is not None:
                current = left_pose if call.arm == "left" else right_pose
                if current is not None:
                    pos_err = np.linalg.norm(call.target_pose[:3] - current[:3])
                    if converged: status = "CONVERGED"
                    elif stuck: status = f"STUCK (pos_err={pos_err:.4f}m, IK solver failed)"
                    else: status = f"TIMEOUT (pos_err={pos_err:.4f}m)"
                    self.execution_feedback.append(
                        f"{call.method}: target_pos={format_floats(call.target_pose[:3].tolist())}, "
                        f"achieved_pos={format_floats(current[:3].tolist())}, {status}"
                    )
                else: self.execution_feedback.append(f"{call.method}: no EEF feedback available")
            else: self.execution_feedback.append(f"{call.method}: completed")
            self.current_call_idx += 1
            self.logger.log_text(f"Completed call: {call.method}, moving to call {self.current_call_idx}")
        
        return action

    def _extract_observation_payload(self, incoming: Dict) -> Optional[Dict]:
        if not isinstance(incoming, dict): return None
        if "robot_r1::proprio" in incoming: return incoming
        for key in ["obs", "observation", "data", "payload"]:
            nested = incoming.get(key)
            if isinstance(nested, dict):
                if "robot_r1::proprio" in nested: return nested
                nested_keys = [k for k in nested.keys() if isinstance(k, str)]
                if any("robot_r1::" in k for k in nested_keys): return nested
        return None
    
    def infer(self, obs: Dict) -> Dict:
        incoming_reset = bool(obs.get("reset", False)) if isinstance(obs, dict) else False
        obs_payload = self._extract_observation_payload(obs if isinstance(obs, dict) else {})

        if incoming_reset and obs_payload is None:
            self.reset()
            hold = np.zeros(self.action_dim, dtype=np.float64)
            return {"action": hold, "actions": np.ascontiguousarray(np.tile(hold, (self.action_horizon, 1)), dtype=np.float64)}

        if incoming_reset: self.reset()

        if obs_payload is None:
            self.logger.log_text(f"Received non-observation payload with keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
            hold = np.zeros(self.action_dim, dtype=np.float64)
            return {"action": hold, "actions": np.ascontiguousarray(np.tile(hold, (self.action_horizon, 1)), dtype=np.float64)}

        self.current_obs = obs_payload
        self.step_count += 1
        if self.step_count % self.video_save_interval == 0: self._capture_video_frame(obs_payload)
        self._record_initial_state(obs_payload)
        
        # --- Hardcoded task logic (evolved from VLM replanning attempts) ---
        if not self.has_initialized:
            self.logger.log_text("Policy initialized. Starting task flow.")
            self.task_stage = 0 # Reset stage
            self.has_initialized = True

        if self.task_stage == 0: # Find bottle
            self.logger.log_text("Task Stage 0: Attempting to locate 'bottle of coffee'.")
            bottle_result = self.robot_api.get_object_position_from_mask("bottle_of_coffee", camera="head")
            if bottle_result and bottle_result['3d_position'] is not None:
                self.bottle_pos = np.array(bottle_result['3d_position'])
                self.logger.log_text(f"Located 'bottle_of_coffee' at {format_floats(self.bottle_pos.tolist())}. Proceeding to Stage 1.")
                self.task_stage = 1
                self.api_call_queue.clear() # Clear any previous failed attempts
                self.current_call_idx = 0
            else:
                # Base repositioning logic from VLM's replanning attempts
                self.logger.warning(f"Could not locate 'bottle_of_coffee'. Attempting base repositioning (attempt {self.base_reposition_attempts + 1}/{self.max_base_reposition_attempts}).")
                self.api_call_queue.clear()
                self.current_call_idx = 0
                if self.base_reposition_attempts == 0:
                    self.api_call_queue.append(APICall(method="move_base", args=[-0.05, 0.0, 0.157], kwargs={'steps':20}, steps_remaining=20, arm="base")) # Small backward and left rotation
                elif self.base_reposition_attempts == 1:
                    self.api_call_queue.append(APICall(method="move_base", args=[0.5, -0.2, 0.0], kwargs={'steps':50}, steps_remaining=50, arm="base")) # Forward and right
                elif self.base_reposition_attempts == 2:
                    self.api_call_queue.append(APICall(method="move_base", args=[0.0, -0.5, -0.5], kwargs={'steps':50}, steps_remaining=50, arm="base")) # Strafe right and rotate right
                elif self.base_reposition_attempts == 3:
                    self.api_call_queue.append(APICall(method="move_base", args=[0.5, -0.5, 0.0], kwargs={'steps':40}, steps_remaining=40, arm="base")) # Forward and right again
                elif self.base_reposition_attempts == 4:
                    self.api_call_queue.append(APICall(method="move_base", args=[1.5, 0.0, 0.0], kwargs={'steps':75}, steps_remaining=75, arm="base")) # Long forward
                else:
                    self.logger.error("Max base repositioning attempts reached. Cannot locate bottle. Task failed.")
                    self.task_stage = 4 # Mark as failed
                    self.api_call_queue.clear()
                self.base_reposition_attempts += 1
        
        elif self.task_stage == 1: # Approach and Grasp
            self.logger.log_text("Task Stage 1: Approaching and grasping 'bottle of coffee'.")
            if not self.bottle_pos: # Should not happen if stage 0 was successful
                self.logger.error("Bottle position not known in Stage 1. Reverting to Stage 0.")
                self.task_stage = 0
                self.base_reposition_attempts = 0
                self.api_call_queue.clear()
            else:
                top_down_ori = [np.pi, 0.0, 0.0]
                
                # Plan the approach and grasp sequence once
                if not self.api_call_queue:
                    self.api_call_queue.append(APICall(method="open_right_gripper", arm="right", steps_remaining=10, target_gripper=0.04))
                    
                    pre_grasp_pos = self.bottle_pos.copy()
                    pre_grasp_pos[2] += 0.20 # Higher pre-grasp for better clearance
                    pre_grasp_target = np.concatenate([pre_grasp_pos, self.robot_api._rpy_to_rotvec(top_down_ori)])
                    self.api_call_queue.append(APICall(method="move_right_arm_to", arm="right", steps_remaining=50, target_pose=pre_grasp_target))

                    grasp_pos = self.bottle_pos.copy()
                    grasp_pos[2] += 0.02 # Slightly above detected centroid
                    grasp_target = np.concatenate([grasp_pos, self.robot_api._rpy_to_rotvec(top_down_ori)])
                    self.api_call_queue.append(APICall(method="move_right_arm_to", arm="right", steps_remaining=50, target_pose=grasp_target))

                    self.api_call_queue.append(APICall(method="close_right_gripper", arm="right", steps_remaining=10, target_gripper=0.0))

                    lift_pos = self.bottle_pos.copy()
                    lift_pos[2] += 0.25 # Lift higher after grasp
                    lift_target = np.concatenate([lift_pos, self.robot_api._rpy_to_rotvec(top_down_ori)])
                    self.api_call_queue.append(APICall(method="move_right_arm_to", arm="right", steps_remaining=50, target_pose=lift_target))
                
                # Execute calls one by one, handling stuck logic in _compute_action
                if self.current_call_idx >= len(self.api_call_queue):
                    self.logger.log_text("All approach and grasp API calls executed. Proceeding to Stage 3.")
                    self.task_stage = 3 # Successfully grasped
                    self.api_call_queue.clear()
                    self.current_call_idx = 0

        elif self.task_stage == 3: # Task done
            self.logger.log_text("Task Stage 3: 'bottle of coffee' picked up. Task complete.")
            # Clear all calls, return no-op action
            self.api_call_queue.clear()
            self.current_call_idx = 0
            self.task_stage = 4 # Mark as truly done

        elif self.task_stage == 4: # Already done or failed
            self.logger.info("Task already completed or failed. Returning no-op.")
            self.api_call_queue.clear()
            self.current_call_idx = 0
            
        action = self._compute_action(obs_payload)
        action = np.ascontiguousarray(action, dtype=np.float64)
        
        if self.step_count % 100 == 0:
            self.logger.log_text(f"Step {self.step_count}, call {self.current_call_idx}/{len(self.api_call_queue)}")
        
        return {
            "action": action,
            "actions": np.ascontiguousarray(np.tile(action, (self.action_horizon, 1)), dtype=np.float64),
        }
    
    def _capture_video_frame(self, obs: Dict):
        cam_key = CAMERA_KEYS.get("head")
        if cam_key is None or cam_key not in obs: return
        img = obs[cam_key]
        if isinstance(img, torch.Tensor): img = img.cpu().numpy()
        if img.ndim == 3 and img.shape[-1] >= 3: img = img[..., :3]
        if img.dtype != np.uint8:
            if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
            else: img = img.astype(np.uint8)
        self.video_frames.append(img)

    def _save_video_on_exit(self):
        if self._video_saved or not self.video_frames: return
        self._video_saved = True
        try:
            import imageio.v2 as imageio
        except ImportError:
            try: import imageio
            except ImportError: self.logger.log_text("imageio not installed, saving frames as images instead"); return

        video_path = self.logger.log_dir / "head_camera.mp4"
        fps = max(1, int(60 / self.video_save_interval))
        try:
            writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', output_params=['-crf', '23'])
            for frame in self.video_frames: writer.append_data(frame)
            writer.close()
            self.logger.log_text(f"Video saved: {video_path} ({len(self.video_frames)} frames, {fps} fps)")
            logging.info(f"Video saved: {video_path} ({len(self.video_frames)} frames)")
        except Exception as e: self.logger.log_text(f"Failed to save video: {e}"); logging.error(f"Failed to save video: {e}")

    def _signal_handler(self, signum, frame):
        self._save_video_on_exit()
        import sys; sys.exit(0)

    def reset(self):
        self._save_video_on_exit()
        self.video_frames.clear(); self._video_saved = False
        self.current_obs = None; self.step_count = 0
        self.api_call_queue.clear(); self.current_call_idx = 0
        self.has_initialized = False; self.initial_trunk_joints = None; self.desired_trunk_joints = None
        self.robot_api._raw_ins_id_mapping = {}; self.robot_api._unique_ins_ids = []
        self.robot_api._object_name_to_ids = {}; self.robot_api._id_to_object_name = {}; self.robot_api._available_objects = []
        self.robot_api._logged_missing_mapping_keys = False
        self.execution_feedback.clear()
        self.task_stage = 0
        self.bottle_pos = None
        self.base_reposition_attempts = 0
        self.logger.log_text("Policy reset")
        self.logger.step = 0

class Logger:
    def __init__(self, instruction: str, base_log_dir: str = "vlm_policy_logs"):
        postfix = re.sub(r'[^\\w\\s]', '', instruction).replace(' ', '_')[:50]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = Path(base_log_dir) / f'{timestamp}_{postfix}'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "log.txt"
        self.step = 0

    def log_text(self, text: str):
        with open(self.log_file, "a") as f: f.write(f"{text}\\n")
        logging.info(text)

    def save_image(self, image: Image.Image, name: str):
        img_path = self.log_dir / f"step_{self.step}_{name}.png"
        image.save(img_path)

@dataclass
class Args:
    port: int = 8222
    instruction: str = "Pick up the bottle of coffee on the table."
    action_horizon: int = 1
    action_dim: int = 21
    log_dir: str = "vlm_policy_logs"
    save_images: bool = True

def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    policy = PickBottlePolicy(
        instruction=args.instruction,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        log_dir=args.log_dir,
        save_images=args.save_images,
    )
    
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={
            "policy_type": "pick_bottle_robust",
            "instruction": args.instruction,
            "action_horizon": args.action_horizon,
            "action_dim": args.action_dim,
        },
    )
    
    logging.info(f"Starting PickBottlePolicy server on port {args.port}")
    logging.info(f"Instruction: {args.instruction}")
    server.serve_forever()

if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
