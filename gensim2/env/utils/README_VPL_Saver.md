# GenSim2 VPL Saver

This module provides a VPL (Visual Programming Language) saver for GenSim2 that saves data in the same format as the VPLSaver used in rai.fm, making it compatible with existing VPL data processing pipelines.

## Overview

The `GenSim2VPLSaver` class allows you to save GenSim2 environment data in a standardized VPL format that includes:

- **Robot State**: Joint positions, velocities, end-effector poses
- **Object Data**: Positions and rotations of rigid bodies and articulated objects
- **Actions**: Robot actions and gripper states
- **Observations**: Environment observations including point clouds
- **Metadata**: Task descriptions, episode information, camera intrinsics

## Installation

The saver requires the following dependencies:
```bash
pip install h5py numpy psutil
```

## Basic Usage

### 1. Initialize the Saver

```python
from gensim2.env.utils.vpl_saver import GenSim2VPLSaver

saver = GenSim2VPLSaver(
    base_dir="./vpl_data",           # Directory to save data
    keep_terminated=False,           # Whether to keep terminated episodes
    fps=30,                         # Frames per second for video
    num_workers=2,                  # Number of worker processes
    save_states_only=False,         # Save only state data (no images/videos)
    log_memory=True                 # Log memory usage
)
```

### 2. Store Data During Episode Execution

```python
# In your environment loop
for timestep in range(max_timesteps):
    # Execute action and get observation
    obs, reward, done, info = env.step(action)
    
    # Prepare episode data
    episode_data = {
        "action": action,
        "observation": obs,
        "reward": reward,
        "done": done
    }
    
    # Store data in the saver
    saver.store(env, episode_data, episode_index=episode_idx)
    
    if done:
        break
```

### 3. Save Episode Data

```python
# When episode ends, save the data
saver.save_episode(episode_index=episode_idx)

# Or save all episodes at once
saver.save_dataset()
```

### 4. Clean Up

```python
# When done with all episodes
saver.close()
```

## Data Structure

The saved data follows the VPL format structure:

```
vpl_data/
├── metadata.json                    # Episode metadata
├── intrinsics.h5                    # Camera intrinsics
├── episode_0/                       # Episode 0 data
│   └── episode_0.h5                # Episode 0 HDF5 file
├── episode_1/                       # Episode 1 data
│   └── episode_1.h5                # Episode 1 HDF5 file
└── ...
```

### HDF5 File Contents

Each episode HDF5 file contains:

- **`action`**: Robot actions (N, 7) - position, rotation, gripper
- **`joint_pos`**: Joint positions (N, 9) - robot joint states
- **`joint_vel`**: Joint velocities (N, 9) - robot joint velocities  
- **`ee_position`**: End-effector positions (N, 3) - TCP positions
- **`ee_rotation`**: End-effector rotations (N, 4) - TCP quaternions
- **`gripper_state`**: Gripper states (N,) - open/close states
- **`obj_positions`**: Object positions - rigid body positions
- **`obj_rotations`**: Object rotations - rigid body rotations
- **`observation`**: Environment observations - state, point clouds, etc.
- **`reward`**: Rewards (N,) - per-timestep rewards
- **`done`**: Done flags (N,) - episode termination flags
- **`task_description`**: Task description string

## Integration with GenSim2

### Environment Setup

The saver automatically detects and stores:

- **Robot State**: From `env.agent.get_qpos()`, `env.agent.get_qvel()`
- **End-Effector**: From `env.tcp.pose`
- **Gripper State**: From `env.gripper_state`
- **Objects**: From `env.rigid_body_id`, `env._rigid_objects`
- **Articulated Objects**: From `env.articulator`
- **Cameras**: From `env.cameras`

### Custom Data

You can add custom data by including it in the `episode_data` dictionary:

```python
episode_data = {
    "action": action,
    "observation": obs,
    "reward": reward,
    "done": done,
    "custom_key": custom_value,  # Will be saved automatically
    "keypoints": detected_keypoints,
    "task_info": additional_task_info
}
```

## Advanced Features

### Memory Management

Enable memory logging to monitor RAM usage:

```python
saver = GenSim2VPLSaver(log_memory=True)
```

### Compression Options

Customize HDF5 compression:

```python
saver.save_episode(
    episode_index=0,
    compression_filter="gzip",
    compression_opts=9
)
```

### Point Cloud Only Mode

Save only point cloud data (no images/videos):

```python
saver.save_dataset(point_cloud_only=True)
```

## Compatibility

The saved data is compatible with:

- **rai.fm VPLSaver**: Same HDF5 structure and metadata format
- **VPL Data Processing**: Standard VPL data loading pipelines
- **HDF5 Tools**: Standard HDF5 viewing and analysis tools

## Example Script

See `vpl_saver_example.py` for a complete working example.

## Error Handling

The saver includes robust error handling:

- **Missing Attributes**: Gracefully handles missing environment attributes
- **Data Conversion**: Automatically converts data types to float32
- **Directory Creation**: Creates output directories automatically
- **Memory Management**: Logs memory usage and provides cleanup

## Performance Considerations

- **Memory**: Data is stored in memory until saved, monitor usage with `log_memory=True`
- **Workers**: Use `num_workers` to parallelize data writing for large datasets
- **Compression**: Use compression to reduce file sizes (default: gzip level 9)
- **Batch Saving**: Save episodes individually or batch with `save_dataset()`

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `num_workers` or enable `save_states_only`
2. **Permission Errors**: Check write permissions for `base_dir`
3. **Import Errors**: Ensure all dependencies are installed
4. **Data Type Errors**: Data is automatically converted to appropriate types

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To extend the saver:

1. Add new data keys to `gensim2_keys_to_save`
2. Implement storage logic in the `store()` method
3. Add saving logic in `save_episode()`
4. Update documentation and examples

## License

This module is part of GenSim2 and follows the same license terms.
