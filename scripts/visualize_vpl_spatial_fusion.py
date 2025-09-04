#!/usr/bin/env python3
"""VPL Visualizer for Spatial Camera Fusion - Frame by Frame."""

import argparse
import time
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VPL Spatial Fusion Visualizer')
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing VPL data")
    parser.add_argument("--episode", type=int, default=0,
                       help="Episode index to visualize")
    parser.add_argument("--fps", type=int, default=8,
                       help="Frames per second for animation")
    parser.add_argument("--use-colors", action="store_true",
                       help="Use color information if available")
    parser.add_argument("--subsample", type=int, default=1,
                       help="Subsample points by this factor (1=no subsampling)")
    parser.add_argument("--loop", action="store_true",
                       help="Loop the animation continuously")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="Start from this frame")
    parser.add_argument("--end-frame", type=int, default=-1,
                       help="End at this frame (-1 for last frame)")
    parser.add_argument("--static", action="store_true",
                       help="Show static view of single timestep instead of animation")
    return parser.parse_args()


def load_episode_data(data_dir, episode_idx):
    """Load episode data from VPL directory."""
    data_path = Path(data_dir)
    episode_dir = data_path / f"episode_{episode_idx}"
    h5_file = episode_dir / f"episode_{episode_idx}.h5"
    
    if not h5_file.exists():
        raise FileNotFoundError(f"Episode file not found: {h5_file}")
    
    print(f"Loading episode {episode_idx} from {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
        episode_data = {}
        
        # Load pointcloud timesteps
        if 'point_cloud' in f and isinstance(f['point_cloud'], h5py.Group):
            pc_grp = f['point_cloud']
            episode_data['point_cloud'] = {}
            
            # Get all timestep keys
            timestep_keys = [k for k in pc_grp.keys() if k.startswith('timestep_')]
            timestep_keys.sort()
            
            print(f"Found {len(timestep_keys)} timesteps")
            
            for ts_key in timestep_keys:
                ts_grp = pc_grp[ts_key]
                episode_data['point_cloud'][ts_key] = {}
                
                for data_key in ts_grp.keys():
                    episode_data['point_cloud'][ts_key][data_key] = ts_grp[data_key][:]
        
        # Load other data
        for key in f.keys():
            if key != 'point_cloud':
                try:
                    if f[key].shape == ():  # Scalar
                        episode_data[key] = f[key][()]
                    else:
                        episode_data[key] = f[key][:]
                except Exception as e:
                    print(f"Warning: Could not load {key}: {e}")
                    continue
    
    return episode_data


def visualize_spatial_fusion(episode_data, fps=8, use_colors=True, subsample=1, 
                           loop=False, start_frame=0, end_frame=-1, static=False):
    """Visualize spatially-fused point cloud data frame by frame."""
    
    if 'point_cloud' not in episode_data:
        print("No point cloud data found")
        return
    
    pc_data = episode_data['point_cloud']
    
    # Get timestep keys
    timestep_keys = [k for k in pc_data.keys() if k.startswith('timestep_')]
    timestep_keys.sort()
    
    if not timestep_keys:
        print("No timestep data found")
        return
    
    num_timesteps = len(timestep_keys)
    print(f"Found {num_timesteps} timesteps with spatially-fused point clouds")
    
    if end_frame == -1:
        end_frame = num_timesteps - 1
    
    start_frame = max(0, min(start_frame, num_timesteps - 1))
    end_frame = max(start_frame, min(end_frame, num_timesteps - 1))
    
    frame_range = list(range(start_frame, end_frame + 1))
    
    # Get state data for additional info
    state_data = episode_data.get('state', None)
    
    if static:
        # Show single frame
        print(f"Displaying static view of timestep {start_frame}")
        visualize_single_timestep(pc_data, timestep_keys[start_frame], use_colors, subsample, state_data, start_frame)
    else:
        # Animate through frames
        print(f"Animating frames {start_frame} to {end_frame} ({len(frame_range)} frames) at {fps} FPS")
        animate_timesteps(pc_data, timestep_keys, frame_range, fps, use_colors, subsample, loop, state_data)


def visualize_single_timestep(pc_data, timestep_key, use_colors, subsample, state_data, frame_idx):
    """Visualize a single timestep."""
    
    timestep_data = pc_data[timestep_key]
    
    # Load data
    if 'combined' in timestep_data:
        combined = timestep_data['combined']
        positions = combined[:, :3]
        colors = combined[:, 3:6] if combined.shape[1] == 6 and use_colors else None
    else:
        positions = timestep_data.get('pos', None)
        colors = timestep_data.get('colors', None) if use_colors else None
        
        if colors is not None and colors.dtype == np.uint8:
            colors = colors.astype(np.float32) / 255.0
    
    if positions is None:
        print("No position data found")
        return
    
    # Subsample
    if subsample > 1:
        indices = np.arange(0, len(positions), subsample)
        positions = positions[indices]
        if colors is not None:
            colors = colors[indices]
    
    print(f"Timestep {timestep_key}: {len(positions)} points")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    # Show statistics
    print(f"Position range:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    
    if colors is not None:
        print(f"Color range:")
        print(f"  R: [{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
        print(f"  G: [{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}]")
        print(f"  B: [{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")
    
    # Create custom visualization with better view control
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Timestep {timestep_key}", width=1200, height=800)
    
    # Add geometries
    vis.add_geometry(coord_frame)
    vis.add_geometry(pcd)
    
    # Set view to fit and center the point cloud
    vis.reset_view_point(True)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Run visualization
    vis.run()
    vis.destroy_window()


def animate_timesteps(pc_data, timestep_keys, frame_range, fps, use_colors, subsample, loop, state_data):
    """Animate through timesteps."""
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("Spatial Fusion Animation", width=1200, height=800)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coord_frame)
    
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    print(f"\n=== Animation Controls ===")
    print(f"  - Mouse: Rotate view")
    print(f"  - Mouse wheel: Zoom")
    print(f"  - Ctrl+Mouse: Pan")
    print(f"  - 'q' or close window: Exit")
    print(f"  - Space: Pause/Resume (not implemented)")
    
    frame_time = 1.0 / fps
    current_idx = 0
    view_initialized = False
    
    try:
        while True:
            # Get current frame
            frame_idx = frame_range[current_idx]
            timestep_key = timestep_keys[frame_idx]
            timestep_data = pc_data[timestep_key]
            
            # Load data for this timestep
            if 'combined' in timestep_data:
                combined = timestep_data['combined']
                positions = combined[:, :3]
                colors = combined[:, 3:6] if combined.shape[1] == 6 and use_colors else None
            else:
                positions = timestep_data.get('pos', None)
                colors = timestep_data.get('colors', None) if use_colors else None
                
                if colors is not None and colors.dtype == np.uint8:
                    colors = colors.astype(np.float32) / 255.0
            
            if positions is None:
                print(f"No position data for {timestep_key}")
                continue
            
            # Subsample
            if subsample > 1:
                indices = np.arange(0, len(positions), subsample)
                positions = positions[indices]
                if colors is not None:
                    colors = colors[indices]
            
            # Update point cloud
            pcd.points = o3d.utility.Vector3dVector(positions)
            
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Default color based on height
                if len(positions) > 0:
                    z_values = positions[:, 2]
                    z_min, z_max = z_values.min(), z_values.max()
                    z_normalized = (z_values - z_min) / (z_max - z_min + 1e-8)
                    
                    default_colors = np.zeros((len(positions), 3))
                    default_colors[:, 0] = z_normalized
                    default_colors[:, 1] = 0.5
                    default_colors[:, 2] = 1.0 - z_normalized
                    pcd.colors = o3d.utility.Vector3dVector(default_colors)
            
            # Update geometry
            vis.update_geometry(pcd)
            
            # Reset view to center on point cloud (only for first frame)
            if not view_initialized and len(positions) > 0:
                vis.reset_view_point(True)
                view_initialized = True
            
            # Print progress
            robot_info = ""
            if state_data is not None and frame_idx < len(state_data):
                joint_pos = state_data[frame_idx][:7]
                robot_info = f" | Joint0: {joint_pos[0]:.3f}"
            
            print(f"\r{timestep_key} | Points: {len(positions):5d} | "
                  f"Frame: {current_idx + 1}/{len(frame_range)}{robot_info}", 
                  end='', flush=True)
            
            # Update visualization
            vis.poll_events()
            vis.update_renderer()
            
            # Check if window was closed
            if not vis.poll_events():
                break
            
            # Advance frame
            current_idx += 1
            
            # Handle looping
            if current_idx >= len(frame_range):
                if loop:
                    current_idx = 0
                    print(f"\n  Looping...")
                else:
                    print(f"\n  Animation complete!")
                    break
            
            # Frame timing
            time.sleep(frame_time)
    
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user")
    
    finally:
        print(f"\n  Closing visualizer...")
        vis.destroy_window()


def main():
    """Main function."""
    args = get_args()
    
    try:
        # Load episode data
        episode_data = load_episode_data(args.data_dir, args.episode)
        
        print(f"\nLoaded episode {args.episode}")
        
        # Start visualization
        print(f"\n=== Spatial Fusion Visualization ===")
        visualize_spatial_fusion(
            episode_data,
            fps=args.fps,
            use_colors=args.use_colors,
            subsample=args.subsample,
            loop=args.loop,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            static=args.static
        )
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
