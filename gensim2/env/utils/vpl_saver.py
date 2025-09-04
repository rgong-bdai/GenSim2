# Copyright (c) 2024-2025 GenSim2 Contributors. All rights reserved.

import h5py
import json
import numpy as np
import os
from os.path import join
from typing import Any, Dict, Optional
import psutil


class GenSim2VPLSaver:
    """A VPL saver for GenSim2 that saves data in the same format as VPLSaver.
    
    This saver is designed to be compatible with the VPL data format used in rai.fm,
    but adapted for GenSim2's specific environment and data structure.
    """

    def __init__(self, base_dir: str, keep_terminated: bool = False, fps: int = 30, 
                 num_workers: int = 4, save_states_only: bool = False, 
                 save_critic_video: bool = False, log_memory: bool = False, 
                 sanity_check: bool = True):
        """Initialize the GenSim2 VPL saver.
        
        Args:
            base_dir: Base directory to save data.
            keep_terminated: Whether to keep terminated episodes.
            fps: Frames per second for video recording.
            num_workers: Number of worker processes for data writing.
            save_states_only: Whether to save only state data (no images/videos).
            save_critic_video: Whether to save critic videos.
            log_memory: Whether to log memory usage.
            sanity_check: Whether to perform sanity checks.
        """
        self.keep_terminated = keep_terminated
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.episode_data = {}
        self.metadata_info = {"num_timesteps": [], "num_episodes": 0}
        self.intrinsics = {}
        self.keys_to_save = ["joint_pos", "joint_vel", "ee_position", "ee_rotation", "eef_pose", "gripper_width", "robot_dof_targets"]
        self.fps = fps
        self.save_critic_video = save_critic_video
        self.num_workers = num_workers
        self.save_states_only = save_states_only
        self.log_memory = log_memory
        self.sanity_check = sanity_check
        
        # GenSim2 specific keys
        self.gensim2_keys_to_save = [
            "action", "observation", "reward", "done", "info",
            "robot_state", "object_positions", "object_rotations",
            "gripper_state", "keypoints", "task_description",
            "point_cloud", "state", "image"
        ]

    def _log_memory_usage(self, stage: str, episode_index: Optional[int] = None) -> dict:
        """Log memory usage if enabled."""
        if not self.log_memory:
            return {}
            
        memory = psutil.virtual_memory()
        memory_info = {
            'percent': memory.percent,
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3),
            'used_bytes': memory.used
        }
        
        episode_str = f" for episode {episode_index}" if episode_index is not None else ""
        print(f"RAM usage {stage}{episode_str}: {memory_info['percent']:.1f}% "
              f"({memory_info['used_gb']:.2f}GB used / {memory_info['total_gb']:.2f}GB total)")
        
        return memory_info

    def store(self, env, episode_data: Dict[str, Any], episode_index: int = 0) -> None:
        """Store episode data from GenSim2 environment.
        
        Args:
            env: GenSim2 environment instance.
            episode_data: Dictionary containing episode data.
            episode_index: Index of the current episode.
        """
        if self.log_memory:
            memory_before = self._log_memory_usage("before storing episode data")
        
        if episode_index not in self.episode_data:
            self.episode_data[episode_index] = {"action": []}

        # Store robot state data - extract from observation array
        if "observation" in episode_data and isinstance(episode_data["observation"], np.ndarray):
            obs_array = episode_data["observation"]
            if len(obs_array) >= 14:  # Expecting at least 7 joint pos + 7 joint vel
                # Extract joint positions (first 7 elements)
                robot_qpos = obs_array[:7].astype(np.float32)
                # Extract joint velocities (next 7 elements)
                robot_qvel = obs_array[7:14].astype(np.float32)
                
                if "joint_pos" not in self.episode_data[episode_index]:
                    self.episode_data[episode_index]["joint_pos"] = []
                if "joint_vel" not in self.episode_data[episode_index]:
                    self.episode_data[episode_index]["joint_vel"] = []
                    
                self.episode_data[episode_index]["joint_pos"].append(robot_qpos)
                self.episode_data[episode_index]["joint_vel"].append(robot_qvel)

        # Store end-effector pose - compute from joint positions using forward kinematics
        if "joint_pos" in self.episode_data[episode_index] and len(self.episode_data[episode_index]["joint_pos"]) > 0:
            # For now, we'll store the last joint position as a proxy for end-effector
            # In a real implementation, you'd compute forward kinematics here
            last_joint_pos = self.episode_data[episode_index]["joint_pos"][-1]
            
            if "ee_position" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["ee_position"] = []
            if "ee_rotation" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["ee_rotation"] = []
            if "eef_pose" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["eef_pose"] = []
            if "gripper_width" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["gripper_width"] = []
            
            # Use joint positions as proxy for end-effector position
            # This is a simplified approach - in practice you'd want proper FK
            ee_position = np.array([last_joint_pos[0], last_joint_pos[1], last_joint_pos[2]], dtype=np.float32)
            ee_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Default quaternion (w, x, y, z)
            
            # Create 4x4 transformation matrix and flatten it (16 elements)
            transform_matrix = np.eye(4, dtype=np.float32)
            
            # Convert quaternion to rotation matrix
            # SAPIEN uses [w, x, y, z] format (scalar-first)
            w, x, y, z = ee_rotation
            
            # Quaternion to rotation matrix conversion
            rotation_matrix = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)
            
            # Set rotation (top-left 3x3) and translation (top-right 3x1)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = ee_position
            
            eef_pose = transform_matrix.flatten()  # Flatten 4x4 to 16 elements
            
            # Get gripper width from gripper state (convert gripper_state to width)
            # gripper_state: 0.0 = closed, 1.0 = open
            # gripper_width: approximate width in meters
            if "observation" in episode_data and isinstance(episode_data["observation"], np.ndarray):
                obs_array = episode_data["observation"]
                if len(obs_array) >= 15:  # Expecting gripper state as last element
                    gripper_state = obs_array[-1]
                    # Convert gripper state to approximate width (0.0 to 0.08 meters)
                    gripper_width = np.array([gripper_state * 0.08], dtype=np.float32)
                else:
                    gripper_width = np.array([0.04], dtype=np.float32)  # Default middle position
            else:
                gripper_width = np.array([0.04], dtype=np.float32)  # Default middle position
            
            self.episode_data[episode_index]["ee_position"].append(ee_position)
            self.episode_data[episode_index]["ee_rotation"].append(ee_rotation)
            self.episode_data[episode_index]["eef_pose"].append(eef_pose)
            self.episode_data[episode_index]["gripper_width"].append(gripper_width)

        # Store gripper state - extract from observation array
        if "observation" in episode_data and isinstance(episode_data["observation"], np.ndarray):
            obs_array = episode_data["observation"]
            if len(obs_array) >= 15:  # Expecting gripper state as last element
                gripper_state = obs_array[-1]
                
                if "gripper_state" not in self.episode_data[episode_index]:
                    self.episode_data[episode_index]["gripper_state"] = []
                self.episode_data[episode_index]["gripper_state"].append(gripper_state)

        # Store object positions and rotations - try to get from environment
        obj_positions = {}
        obj_rotations = {}
        
        # Method 1: Try to get from rigid body objects
        if hasattr(env, 'rigid_body_id') and hasattr(env, '_rigid_objects'):
            for obj_name, obj_ids in env.rigid_body_id.items():
                if obj_ids and len(obj_ids) > 0:
                    obj_id = obj_ids[0]
                    if obj_id in env._rigid_objects:
                        obj = env._rigid_objects[obj_id]
                        if hasattr(obj, 'pose'):
                            obj_positions[obj_name] = obj.pose.p.astype(np.float32)
                            obj_rotations[obj_name] = obj.pose.q.astype(np.float32)
        
        # Store object data if we found it
        if obj_positions:
            if "obj_positions" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["obj_positions"] = {}
            for obj_name, pos in obj_positions.items():
                if obj_name not in self.episode_data[episode_index]["obj_positions"]:
                    self.episode_data[episode_index]["obj_positions"][obj_name] = []
                self.episode_data[episode_index]["obj_positions"][obj_name].append(pos)
                
        if obj_rotations:
            if "obj_rotations" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["obj_rotations"] = {}
            for obj_name, rot in obj_rotations.items():
                if obj_name not in self.episode_data[episode_index]["obj_rotations"]:
                    self.episode_data[episode_index]["obj_rotations"][obj_name] = []
                self.episode_data[episode_index]["obj_rotations"][obj_name].append(rot)

        # Store articulated object data
        if hasattr(env, 'articulator') and env.articulator is not None:
            if "articulated_joint_pos" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["articulated_joint_pos"] = {}
                self.episode_data[episode_index]["articulated_root_pos"] = {}
                self.episode_data[episode_index]["articulated_root_rot"] = {}
            
            articulator = env.articulator
            if hasattr(articulator, 'instance') and hasattr(articulator.instance, 'get_qpos'):
                joint_pos = articulator.instance.get_qpos().astype(np.float32)
                if "robot" not in self.episode_data[episode_index]["articulated_joint_pos"]:
                    self.episode_data[episode_index]["articulated_joint_pos"]["robot"] = []
                self.episode_data[episode_index]["articulated_joint_pos"]["robot"].append(joint_pos)
            
            if hasattr(articulator, 'instance') and hasattr(articulator.instance, 'get_root_pose'):
                root_pose = articulator.instance.get_root_pose()
                if "robot" not in self.episode_data[episode_index]["articulated_root_pos"]:
                    self.episode_data[episode_index]["articulated_root_pos"]["robot"] = []
                    self.episode_data[episode_index]["articulated_root_rot"]["robot"] = []
                self.episode_data[episode_index]["articulated_root_pos"]["robot"].append(root_pose.p.astype(np.float32))
                self.episode_data[episode_index]["articulated_root_rot"]["robot"].append(root_pose.q.astype(np.float32))

        # Store camera intrinsics if available
        if hasattr(env, 'cameras'):
            for cam_name, cam in env.cameras.items():
                if cam_name not in self.intrinsics:
                    if hasattr(cam, 'get_intrinsic_matrix'):
                        intrinsic_matrix = cam.get_intrinsic_matrix()
                        self.intrinsics[cam_name] = intrinsic_matrix.astype(np.float32)

        # Store observations if available
        if "observation" in episode_data:
            obs_list = episode_data["observation"]
            
            # Handle list of observation dictionaries
            if isinstance(obs_list, list) and len(obs_list) > 0:
                # Initialize data structures
                if "point_cloud" not in self.episode_data[episode_index]:
                    self.episode_data[episode_index]["point_cloud"] = {}
                if "state" not in self.episode_data[episode_index]:
                    self.episode_data[episode_index]["state"] = []
                if "image" not in self.episode_data[episode_index]:
                    self.episode_data[episode_index]["image"] = []
                
                # Process each observation
                for obs in obs_list:
                    if isinstance(obs, dict):
                        # Extract pointcloud data
                        if 'pointcloud' in obs:
                            pc_data = obs['pointcloud']
                            # Only save position and colors, ignore segmentation
                            for key in ['pos', 'colors']:
                                if key in pc_data and pc_data[key] is not None:
                                    if key not in self.episode_data[episode_index]["point_cloud"]:
                                        self.episode_data[episode_index]["point_cloud"][key] = []
                                    self.episode_data[episode_index]["point_cloud"][key].append(pc_data[key])
                        
                        # Extract state data
                        if 'state' in obs:
                            self.episode_data[episode_index]["state"].append(obs['state'])
                        
                        # Extract image data
                        if 'image' in obs:
                            self.episode_data[episode_index]["image"].append(obs['image'])
            
            # Handle single observation dictionary (fallback)
            elif isinstance(obs_list, dict):
                if "point_cloud" not in self.episode_data[episode_index]:
                    self.episode_data[episode_index]["point_cloud"] = {}
                
                if 'pointcloud' in obs_list:
                    pc_data = obs_list['pointcloud']
                    # Only save position and colors, ignore segmentation
                    for key in ['pos', 'colors']:
                        if key in pc_data and pc_data[key] is not None:
                            if key not in self.episode_data[episode_index]["point_cloud"]:
                                self.episode_data[episode_index]["point_cloud"][key] = []
                            self.episode_data[episode_index]["point_cloud"][key].append(pc_data[key])
                
                if 'state' in obs_list:
                    if "state" not in self.episode_data[episode_index]:
                        self.episode_data[episode_index]["state"] = []
                    self.episode_data[episode_index]["state"].append(obs_list['state'])
                
                if 'image' in obs_list:
                    if "image" not in self.episode_data[episode_index]:
                        self.episode_data[episode_index]["image"] = []
                    self.episode_data[episode_index]["image"].append(obs_list['image'])
        else:
            # Fallback: store observation as-is for backward compatibility
            if "observation" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["observation"] = []
            self.episode_data[episode_index]["observation"].append(episode_data["observation"])

        # Store actions if available
        if "action" in episode_data:
            if "action" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["action"] = []
            self.episode_data[episode_index]["action"].append(episode_data["action"])

        # Store rewards if available
        if "reward" in episode_data:
            if "reward" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["reward"] = []
            self.episode_data[episode_index]["reward"].append(episode_data["reward"])

        # Store done flags if available
        if "done" in episode_data:
            if "done" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["done"] = []
            self.episode_data[episode_index]["done"].append(episode_data["done"])

        # Store task description if available
        if hasattr(env, 'task') and hasattr(env.task, 'task_description'):
            if "task_description" not in self.episode_data[episode_index]:
                self.episode_data[episode_index]["task_description"] = env.task.task_description

        if self.log_memory:
            memory_after = self._log_memory_usage("after storing episode data")
            print(f"Memory difference: {(memory_after.get('used_bytes', 0) - memory_before.get('used_bytes', 0)) / (1024**2):.1f}MB")

    def _fuse_cameras_at_timestep(self, pointcloud_timestep_data):
        """Fuse point clouds from multiple cameras at a single timestep.
        
        Args:
            pointcloud_timestep_data: Dictionary with camera data for one timestep
                                    e.g., {'camera1': {'pos': ..., 'colors': ...}, 
                                          'camera2': {'pos': ..., 'colors': ...}}
            
        Returns:
            Dictionary with fused 'pos', 'colors' arrays from all cameras
        """
        if not pointcloud_timestep_data or not isinstance(pointcloud_timestep_data, dict):
            return None
        
        all_positions = []
        all_colors = []
        
        # Handle different data structures
        if 'pos' in pointcloud_timestep_data and 'colors' in pointcloud_timestep_data:
            # Single camera data - just return as is
            return {
                'pos': pointcloud_timestep_data['pos'],
                'colors': pointcloud_timestep_data.get('colors', None)
            }
        
        # Multiple cameras - fuse them spatially
        for camera_name, camera_data in pointcloud_timestep_data.items():
            if isinstance(camera_data, dict):
                if 'pos' in camera_data and camera_data['pos'] is not None:
                    all_positions.append(camera_data['pos'])
                if 'colors' in camera_data and camera_data['colors'] is not None:
                    all_colors.append(camera_data['colors'])
        
        # Concatenate data from all cameras
        fused_data = {}
        if all_positions:
            fused_data['pos'] = np.concatenate(all_positions, axis=0)
        if all_colors:
            fused_data['colors'] = np.concatenate(all_colors, axis=0)
            
        return fused_data if fused_data else None

    def save_episode(self, episode_index: int, save_to_video: bool = False, 
                    compression_filter: str = "gzip", compression_opts: int = 9,
                    point_cloud_only: bool = False) -> int:
        """Save an episode to disk.
        
        Args:
            episode_index: Index of the episode to save.
            save_to_video: Whether to save images as video.
            compression_filter: HDF5 compression filter.
            compression_opts: HDF5 compression options.
            point_cloud_only: Whether to save only point cloud data.
            
        Returns:
            Number of episodes saved.
        """
        if episode_index not in self.episode_data:
            print(f"Episode {episode_index} not found in episode_data")
            return 0

        # Create episode directory
        episode_dir = join(self.base_dir, f"episode_{episode_index}")
        os.makedirs(episode_dir, exist_ok=True)

        # Save episode data to HDF5
        with h5py.File(join(episode_dir, f"episode_{episode_index}.h5"), "w") as f_writer:
            for k, v in self.episode_data[episode_index].items():
                # Check if this key should be saved
                should_save = (k in self.keys_to_save or k in self.gensim2_keys_to_save)
                
                if should_save:
                    if len(v) == 0:
                        continue
                    
                    # Handle pointcloud data - save timesteps with camera fusion
                    if k == "point_cloud" and isinstance(v, dict):
                        pc_grp = f_writer.create_group(k)
                        
                        # Get the number of timesteps
                        pos_list = v.get('pos', [])
                        colors_list = v.get('colors', [])
                        num_timesteps = len(pos_list)
                        
                        print(f"Saving {num_timesteps} timesteps with camera-fused point clouds")
                        
                        # Save each timestep with camera fusion
                        for t in range(num_timesteps):
                            # Get pointcloud data for this timestep
                            pos_data = pos_list[t] if t < len(pos_list) else None
                            colors_data = colors_list[t] if t < len(colors_list) else None
                            
                            if pos_data is not None:
                                # Fuse cameras at this timestep (spatial fusion)
                                timestep_pc_data = {'pos': pos_data}
                                if colors_data is not None:
                                    timestep_pc_data['colors'] = colors_data
                                
                                fused_data = self._fuse_cameras_at_timestep(timestep_pc_data)
                                
                                if fused_data:
                                    # Create combined point cloud data for visuomotor compatibility
                                    if 'pos' in fused_data:
                                        if 'colors' in fused_data and fused_data['colors'] is not None:
                                            colors = fused_data['colors']
                                            # Normalize colors to float32
                                            if colors.dtype == np.uint8:
                                                colors_norm = colors.astype(np.float32) / 255.0
                                            elif colors.dtype != np.float32:
                                                colors_norm = colors.astype(np.float32)
                                            else:
                                                colors_norm = colors
                                            
                                            # Combine position and normalized colors (6 channels: xyz + rgb)
                                            combined = np.concatenate([fused_data['pos'], colors_norm], axis=1)
                                        else:
                                            # Only position available (3 channels: xyz)
                                            combined = fused_data['pos']
                                        
                                        # Store as direct dataset under point_cloud group (compatible with visuomotor)
                                        pc_grp.create_dataset(
                                            str(t), data=combined,
                                            compression=compression_filter,
                                            compression_opts=compression_opts
                                        )
                        
                        continue  # Skip general processing for pointcloud
                    
                    # Handle state data (convert list to array)
                    if k == "state" and isinstance(v, list):
                        state_array = np.array(v)
                        f_writer.create_dataset(
                            k, data=state_array,
                            compression=compression_filter,
                            compression_opts=compression_opts
                        )
                        continue  # Skip general processing for state
                    
                    # Handle image data (convert list to array)
                    if k == "image" and isinstance(v, list):
                        image_array = np.array(v)
                        f_writer.create_dataset(
                            k, data=image_array,
                            compression=compression_filter,
                            compression_opts=compression_opts
                        )
                        continue  # Skip general processing for images
                    
                    # Convert to numpy array if it's a list
                    if isinstance(v, list):
                        v = np.array(v)
                    
                    # Ensure data is numeric and not object dtype
                    if hasattr(v, 'dtype') and v.dtype == np.dtype('O'):
                        print(f"Warning: Skipping object dtype data for {k}")
                        continue
                    
                    # Check if data is scalar (no compression for scalars)
                    if np.isscalar(v) or (hasattr(v, 'size') and v.size == 1):
                        f_writer.create_dataset(k, data=v)
                    else:
                        f_writer.create_dataset(
                            k, data=v,
                            compression=compression_filter,
                            compression_opts=compression_opts
                        )

                # Handle task description (variable-length string)
                elif k == "task_description":
                    dt = h5py.special_dtype(vlen=str)
                    data = v
                    if compression_filter and data and len(data) >= 100:
                        ds_kwargs = {
                            "dtype": dt,
                            "compression": compression_filter,
                            "compression_opts": compression_opts
                        }
                    else:
                        ds_kwargs = {"dtype": dt}
                    f_writer.create_dataset(k, data=data, **ds_kwargs)

                # Handle object poses / articulated data
                elif k in ("obj_positions", "obj_rotations", "articulated_joint_pos", 
                          "articulated_root_pos", "articulated_root_rot"):
                    grp = f_writer.create_group(k)
                    for obj_id, arr in v.items():
                        if isinstance(arr, list):
                            arr = np.array(arr)
                        grp.create_dataset(
                            obj_id, data=arr,
                            compression=compression_filter,
                            compression_opts=compression_opts
                        )



        # Save intrinsics if available
        if self.intrinsics and episode_index == 0:
            with h5py.File(join(self.base_dir, "intrinsics.h5"), "w") as data:
                intrinsics_array = np.stack(list(self.intrinsics.values()))
                data["intrinsics"] = intrinsics_array

        # Update metadata - correctly handle nested data structures
        if "action" in self.episode_data[episode_index]:
            action_data = self.episode_data[episode_index]["action"]
            if hasattr(action_data, 'shape'):
                # Handle numpy arrays
                if len(action_data.shape) > 1:
                    num_timesteps = action_data.shape[1]  # (batch, timesteps, dims)
                else:
                    num_timesteps = action_data.shape[0]  # (timesteps,)
            else:
                # Handle lists
                if len(action_data) > 0 and hasattr(action_data[0], 'shape'):
                    # List contains arrays - use the shape of the first array
                    first_array = action_data[0]
                    if len(first_array.shape) > 1:
                        num_timesteps = first_array.shape[1]  # (batch, timesteps, dims)
                    else:
                        num_timesteps = first_array.shape[0]  # (timesteps,)
                elif len(action_data) > 0 and isinstance(action_data[0], list):
                    # Nested list - get the length of the inner list
                    num_timesteps = len(action_data[0])
                else:
                    # Regular list
                    num_timesteps = len(action_data)
        elif "observation" in self.episode_data[episode_index]:
            obs_data = self.episode_data[episode_index]["observation"]
            if hasattr(obs_data, 'shape'):
                # Handle numpy arrays
                if len(obs_data.shape) > 1:
                    num_timesteps = obs_data.shape[1]  # (batch, timesteps, dims)
                else:
                    num_timesteps = obs_data.shape[0]  # (timesteps,)
            else:
                # Handle list - for observations, typically a list of dictionaries
                num_timesteps = len(obs_data) if len(obs_data) > 0 else 0
        else:
            num_timesteps = 0
            
        self.metadata_info["num_timesteps"].append(num_timesteps)
        self.metadata_info["num_episodes"] += 1
        
        with open(join(self.base_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata_info, f, indent=4)

        print(f"Saved episode {episode_index} to {episode_dir}")
        
        # Clear episode data after saving
        self.episode_data[episode_index] = {"action": []}
        
        return self.metadata_info["num_episodes"]

    def save_dataset(self, save_to_video: bool = False, compression_filter: str = "gzip", 
                    compression_opts: int = 9, point_cloud_only: bool = False) -> int:
        """Save all episodes in the dataset.
        
        Args:
            save_to_video: Whether to save images as video.
            compression_filter: HDF5 compression filter.
            compression_opts: HDF5 compression options.
            point_cloud_only: Whether to save only point cloud data.
            
        Returns:
            Number of episodes saved.
        """
        total_episodes = 0
        
        for episode_index in list(self.episode_data.keys()):
            if episode_index != "action":  # Skip the dummy key
                episodes_saved = self.save_episode(
                    episode_index, save_to_video, compression_filter, 
                    compression_opts, point_cloud_only
                )
                total_episodes += episodes_saved
        
        print(f"Saved {total_episodes} episodes to {self.base_dir}")
        return total_episodes

    def reset_episode(self, episode_index: int):
        """Reset episode data for a specific episode.
        
        Args:
            episode_index: Index of the episode to reset.
        """
        if episode_index in self.episode_data:
            self.episode_data[episode_index] = {"action": []}

    def get_episode_data(self, episode_index: int) -> Dict[str, Any]:
        """Get episode data for a specific episode.
        
        Args:
            episode_index: Index of the episode.
            
        Returns:
            Episode data dictionary.
        """
        return self.episode_data.get(episode_index, {})

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata information.
        
        Returns:
            Metadata dictionary.
        """
        return self.metadata_info.copy()

    def close(self):
        """Close the saver and clean up resources."""
        # Clear episode data
        self.episode_data.clear()
        print("GenSim2 VPL saver closed")
