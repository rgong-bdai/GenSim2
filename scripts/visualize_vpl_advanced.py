#!/usr/bin/env python3
"""Advanced VPL Trajectory Visualizer for GenSim2.
"""

import argparse
import time
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import open3d as o3d
from einops import rearrange

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced VPL Trajectory Visualizer')
    parser.add_argument("--data-dir", type=str, default="./vpl_data", 
                       help="Directory containing VPL data")
    parser.add_argument("--episode", type=int, default=0, 
                       help="Episode index to visualize")
    parser.add_argument("--fps", type=int, default=30, 
                       help="Frames per second for visualization")
    parser.add_argument("--point-cloud", action="store_true", 
                       help="Visualize point cloud data")
    parser.add_argument("--trajectory", action="store_true", 
                       help="Visualize robot trajectory")
    parser.add_argument("--images", action="store_true", 
                       help="Visualize image data")
    parser.add_argument("--all", action="store_true", 
                       help="Visualize all available data")
    return parser.parse_args()


def load_vpl_data(data_dir: Path):
    """Load VPL data from directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load metadata
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata['num_episodes']} episodes")
    else:
        print("Warning: metadata.json not found")
        metadata = {"num_episodes": 0, "num_timesteps": []}
    
    # Load intrinsics if available
    intrinsics_path = data_dir / "intrinsics.h5"
    intrinsics = None
    if intrinsics_path.exists():
        with h5py.File(intrinsics_path, 'r') as f:
            intrinsics = f['intrinsics'][:]
        print(f"Loaded camera intrinsics: {intrinsics.shape}")
    
    # Load episode data
    episodes = []
    episode_dirs = sorted([d for d in data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('episode_')])
    
    for episode_dir in episode_dirs:
        episode_idx = int(episode_dir.name.split('_')[1])
        h5_path = episode_dir / f"episode_{episode_idx}.h5"
        
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                episode_data = {}
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        # Handle groups (like pointcloud)
                        if key == 'pointcloud':
                            # Load timestep data from pointcloud group
                            pc_group = f[key]
                            timestep_keys = [k for k in pc_group.keys() if k.startswith('timestep_')]
                            timestep_keys.sort()
                            
                            # Extract positions and colors from first few timesteps for visualization
                            positions = []
                            colors = []
                            for ts_key in timestep_keys[:10]:  # Limit to first 10 for performance
                                ts_group = pc_group[ts_key]
                                if 'pos' in ts_group:
                                    positions.append(ts_group['pos'][:])
                                if 'colors' in ts_group:
                                    colors.append(ts_group['colors'][:])
                            
                            episode_data['pointcloud_pos'] = positions
                            episode_data['pointcloud_colors'] = colors
                        else:
                            # For other groups, try to extract what we can
                            episode_data[key] = {}
                            for subkey in f[key].keys():
                                try:
                                    episode_data[key][subkey] = f[key][subkey][:]
                                except:
                                    pass
                    else:
                        # Handle datasets
                        try:
                            if f[key].shape == ():  # Scalar
                                episode_data[key] = f[key][()]
                            else:
                                episode_data[key] = f[key][:]
                        except Exception as e:
                            print(f"Warning: Could not load {key}: {e}")
                            continue
                
                episodes.append((episode_idx, episode_data))
                action_len = len(episode_data.get('action', [[]])[0]) if 'action' in episode_data else 0
                print(f"Loaded episode {episode_idx}: {action_len} timesteps")
    
    return metadata, intrinsics, episodes


def visualize_images(episodes, episode_idx=0, fps=30):
    """Visualize image data from VPL dataset."""
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found")
        return
    
    episode_idx_actual, episode_data = episodes[episode_idx]
    print(f"Visualizing images for episode {episode_idx_actual}")
    
    # Check for image data
    if 'observation' not in episode_data:
        print("No observation data found")
        return
    
    observations = episode_data['observation']
    
    # Handle different observation formats
    if isinstance(observations, dict) and 'image' in observations:
        images = observations['image']
        if isinstance(images, dict):
            # Multiple camera images
            for cam_name, cam_images in images.items():
                print(f"Visualizing camera {cam_name}: {cam_images.shape}")
                
                cv2_window_name = f"Camera {cam_name} - Episode {episode_idx_actual}"
                cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
                
                for t in range(len(cam_images)):
                    # Convert to BGR for OpenCV
                    if cam_images[t].shape[-1] == 3:  # RGB
                        img = cv2.cvtColor(cam_images[t], cv2.COLOR_RGB2BGR)
                    else:
                        img = cam_images[t]
                    
                    cv2.imshow(cv2_window_name, img)
                    if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
                        break
                
                cv2.destroyWindow(cv2_window_name)
        else:
            print(f"Unexpected image data format: {type(images)}")
    else:
        print("No image data found in observations")


def visualize_point_cloud(episodes, episode_idx=0, fps=30):
    """Visualize point cloud data with trajectory."""
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found")
        return
    
    episode_idx_actual, episode_data = episodes[episode_idx]
    print(f"Visualizing point cloud for episode {episode_idx_actual}")
    
    # Create world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"VPL Point Cloud - Episode {episode_idx_actual}")
    vis.add_geometry(world_frame)
    
    # Create point cloud object
    point_cloud_o3d = o3d.geometry.PointCloud()
    
    # Get trajectory color
    trajectory_color = np.random.uniform(0, 1, 3)
    
    # Check for point cloud data
    if 'pointcloud_pos' not in episode_data:
        print("No point cloud data found")
        return
    
    positions = episode_data['pointcloud_pos']
    colors = episode_data.get('pointcloud_colors', None)
    
    print(f"Found {len(positions)} timesteps of point cloud data")
    
    # Animate through timesteps
    vis.add_geometry(point_cloud_o3d)
    
    for t in range(len(positions)):
        print(f"\rTimestep {t+1}/{len(positions)}", end='', flush=True)
        
        # Update point cloud
        current_points = positions[t]
        point_cloud_o3d.points = o3d.utility.Vector3dVector(current_points)
        
        # Add colors if available
        if colors and t < len(colors):
            current_colors = colors[t]
            if current_colors.dtype == np.uint8:
                current_colors = current_colors.astype(np.float32) / 255.0
            point_cloud_o3d.colors = o3d.utility.Vector3dVector(current_colors)
        
        # Update visualization
        vis.update_geometry(point_cloud_o3d)
        vis.poll_events()
        vis.update_renderer()
        
        # Check if window closed
        if not vis.poll_events():
            break
            
        time.sleep(1 / fps)
    
    print(f"\nAnimation complete!")
    
    # Keep window open for inspection
    print("Press 'q' or close window to exit")
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()
        time.sleep(0.1)
    
    vis.destroy_window()
    
    # Old code below - remove this section
    if False and isinstance(observations, dict) and 'pointcloud' in observations:
        pointcloud_data = observations['pointcloud']
        
        if isinstance(pointcloud_data, dict):
            # Multiple camera point clouds
            for cam_name, cam_data in pointcloud_data.items():
                if 'pos' in cam_data:
                    points = cam_data['pos']
                    if points.ndim == 2 and points.shape[1] == 3:
                        print(f"Processing camera {cam_name} point cloud: {points.shape}")
                        
                        # Set point cloud data
                        point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
                        
                        # Add colors if available
                        if 'rgb' in cam_data:
                            colors = cam_data['rgb'] / 255.0  # Normalize to [0,1]
                            point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
                        
                        # Add to visualizer
                        vis.add_geometry(point_cloud_o3d)
                        
                        # Add end-effector trajectory if available
                        if 'ee_position' in episode_data:
                            ee_positions = episode_data['ee_position']
                            for t, pos in enumerate(ee_positions):
                                # Create sphere for end-effector position
                                eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                                eef_sphere.paint_uniform_color(trajectory_color)
                                
                                # Transform to position
                                transform = np.eye(4)
                                transform[:3, 3] = pos
                                eef_sphere.transform(transform)
                                
                                vis.add_geometry(eef_sphere, reset_bounding_box=False)
                        
                        # Update visualization
                        vis.poll_events()
                        vis.update_renderer()
                        time.sleep(1 / fps)
                        
                        break  # For now, just show first camera
        else:
            print(f"Unexpected pointcloud data format: {type(pointcloud_data)}")
    else:
        print("No pointcloud data found in observations")
    
    # Keep window open
    print("Press 'q' to close visualization")
    while True:
        vis.poll_events()
        vis.update_renderer()
        if not vis.poll_events():
            break
        time.sleep(0.1)
    
    vis.destroy_window()


def visualize_trajectory_3d(episodes, episode_idx=0, fps=30):
    """Visualize 3D trajectory with point clouds."""
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} not found")
        return
    
    episode_idx_actual, episode_data = episodes[episode_idx]
    print(f"Visualizing 3D trajectory for episode {episode_idx_actual}")
    
    # Create world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    
    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"VPL Trajectory - Episode {episode_idx_actual}")
    vis.add_geometry(world_frame)
    
    # Check for trajectory data
    if 'ee_position' not in episode_data:
        print("No end-effector position data found")
        return
    
    ee_positions = episode_data['ee_position']
    print(f"End-effector trajectory: {ee_positions.shape}")
    
    # Create trajectory visualization
    trajectory_color = np.random.uniform(0, 1, 3)
    
    # Add trajectory line segments
    for i in range(len(ee_positions) - 1):
        start_pos = ee_positions[i]
        end_pos = ee_positions[i + 1]
        
        # Create line segment
        points = np.array([start_pos, end_pos])
        lines = np.array([[0, 1]])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(trajectory_color)
        
        vis.add_geometry(line_set, reset_bounding_box=False)
    
    # Add end-effector positions as spheres
    for t, pos in enumerate(ee_positions):
        # Create sphere for position
        eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        
        # Color by time (start = green, end = red)
        if t == 0:
            color = [0, 1, 0]  # Green for start
        elif t == len(ee_positions) - 1:
            color = [1, 0, 0]  # Red for end
        else:
            color = trajectory_color
        
        eef_sphere.paint_uniform_color(color)
        
        # Transform to position
        transform = np.eye(4)
        transform[:3, 3] = pos
        eef_sphere.transform(transform)
        
        vis.add_geometry(eef_sphere, reset_bounding_box=False)
    
    # Add point cloud if available
    if 'observation' in episode_data:
        observations = episode_data['observation']
        if isinstance(observations, dict) and 'pointcloud' in observations:
            pointcloud_data = observations['pointcloud']
            
            if isinstance(pointcloud_data, dict):
                for cam_name, cam_data in pointcloud_data.items():
                    if 'pos' in cam_data:
                        points = cam_data['pos']
                        if points.ndim == 2 and points.shape[1] == 3:
                            print(f"Adding point cloud from camera {cam_name}: {points.shape}")
                            
                            # Create point cloud
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(points)
                            
                            # Add colors if available
                            if 'rgb' in cam_data:
                                colors = cam_data['rgb'] / 255.0
                                pcd.colors = o3d.utility.Vector3dVector(colors)
                            
                            vis.add_geometry(pcd, reset_bounding_box=False)
                            break
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Keep window open
    print("Press 'q' to close visualization")
    while True:
        vis.poll_events()
        vis.update_renderer()
        if not vis.poll_events():
            break
        time.sleep(0.1)
    
    vis.destroy_window()


def main() -> None:
    """Main function."""
    args = get_args()
    
    try:
        # Load VPL data
        data_dir = Path(args.data_dir)
        print(f"Loading VPL data from: {data_dir}")
        metadata, intrinsics, episodes = load_vpl_data(data_dir)
        
        if len(episodes) == 0:
            print("No episodes found!")
            return
        
        # Determine visualization mode
        if args.all:
            # Visualize all available data
            print("Visualizing all available data...")
            
            if args.episode < len(episodes):
                # Visualize images
                try:
                    visualize_images(episodes, args.episode, args.fps)
                except Exception as e:
                    print(f"Image visualization failed: {e}")
                
                # Visualize point cloud
                try:
                    visualize_point_cloud(episodes, args.episode, args.fps)
                except Exception as e:
                    print(f"Point cloud visualization failed: {e}")
                
                # Visualize trajectory
                try:
                    visualize_trajectory_3d(episodes, args.episode, args.fps)
                except Exception as e:
                    print(f"Trajectory visualization failed: {e}")
            else:
                print(f"Episode {args.episode} not found. Available: 0-{len(episodes)-1}")
        
        elif args.point_cloud:
            visualize_point_cloud(episodes, args.episode, args.fps)
        
        elif args.trajectory:
            visualize_trajectory_3d(episodes, args.episode, args.fps)
        
        elif args.images:
            visualize_images(episodes, args.episode, args.fps)
        
        else:
            # Default: show trajectory
            visualize_trajectory_3d(episodes, args.episode, args.fps)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
