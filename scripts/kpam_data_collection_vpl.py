#!/usr/bin/env python3
"""KPAM Data Collection Script with VPL Saver Integration.

This script collects data using KPAM planner and saves it in VPL format
compatible with rai.fm data processing pipelines.
"""

import icecream
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gensim2.env.task.origin_tasks import *
from gensim2.env.task.primitive_tasks import *
from gensim2.env.solver.planner import KPAMPlanner
from gensim2.env.create_task import create_gensim
from gensim2.env.utils.vpl_saver import GenSim2VPLSaver

from common_parser import parser

# Add VPL-specific arguments
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--load_from_cache", action="store_true")
parser.add_argument("--dataset", type=str, default="gensim2")
parser.add_argument("--vpl_dir", type=str, default="./vpl_data", 
                   help="Directory to save VPL format data")
parser.add_argument("--log_memory", action="store_true", 
                   help="Log memory usage during data collection")

# Define both articulated and rigid object tasks
articulated_envs = [
    "OpenBox",
    "CloseBox",
    "OpenLaptop",
    "CloseLaptop",
    "OpenDrawer",
    "PushDrawerClose",
    "SwingBucketHandle",
    "LiftBucketUpright",
    "PressToasterLever",
    "PushToasterForward",
    "MoveBagForward",
    "OpenSafe",
    "CloseSafe",
    "RotateMicrowaveDoor",
    "CloseMicrowave",
    "CloseSuitcaseLid",
    "SwingSuitcaseLidOpen",
    "RelocateSuitcase",
    "TurnOnFaucet",
    "TurnOffFaucet",
    "SwingDoorOpen",
    "ToggleDoorClose",
    "CloseRefrigeratorDoor",
    "OpenRefrigeratorDoor",
]

rigid_body_envs = [
    "ReachRigidBody",
    "ReachCuboid", 
    "ReachThinObject",
    "LiftBanana",
    "PushBox",
]

# Combine all environments
all_envs = articulated_envs + rigid_body_envs

def collect_episode_data(env, max_steps, render=False):
    """Collect data for a single episode using KPAM planner."""
    obs = env.reset()
    episode_data = []
    
    for task in env.sub_tasks:
        if task == "Grasp":
            env.grasp()
            continue
        elif task == "UnGrasp":
            env.ungrasp()
            continue
            
        # Create KPAM planner for this task
        config_path = f"gensim2/env/solver/kpam/config/{task}.yaml"
        
        # Handle different task types
        has_articulator = (hasattr(env.task, 'articulator') and
                         env.task.articulator is not None)
        if has_articulator:
            # Articulated object task
            expert_planner = KPAMPlanner(
                env, config_path,
                env.task.articulator.random_pose["rot"]
            )
        else:
            # Rigid body task - no articulator rotation needed
            expert_planner = KPAMPlanner(env, config_path)
        
        expert_planner.reset_expert()
        
        if env.viewer and env.viewer.closed:
            break
            
        for step in range(max_steps):
            action = expert_planner.get_action()
            
            # Store step data
            step_data = {
                "obs": obs,
                "action": action,
                "reward": 0.0,  # Default reward
                "done": False,
                "step": step,
                "task": task
            }
            episode_data.append(step_data)
            
            # Execute action
            obs, reward, done, info = env.step(action)
            
            # Update step data with results
            step_data["reward"] = reward
            step_data["done"] = done
            step_data["info"] = info
            
            if render:
                env.render()
                
            if done:
                break
                
        if not info.get("sub_task_success", False):
            break
            
    return episode_data

def main():
    args = parser.parse_args()
    
    # Initialize VPL saver if saving is enabled
    vpl_saver = None
    if args.save:
        vpl_saver = GenSim2VPLSaver(
            base_dir=args.vpl_dir,
            keep_terminated=False,
            fps=30,
            num_workers=2,
            save_states_only=False,
            log_memory=args.log_memory
        )
        print(f"VPL saver initialized. Data will be saved to: {args.vpl_dir}")
    
    success_rate = {}
    if args.env != "":
        envs = [args.env]
    else:
        envs = all_envs

    for env_name in envs:
        print(f"\n{'='*50}")
        print(f"Starting data collection for: {env_name}")
        print(f"{'='*50}")
        
        env = create_gensim(
            task_name=env_name,
            asset_id=args.asset_id,
            use_ray_tracing=args.rt,
            use_gui=args.render,
            eval=False,
            obs_mode=args.obs_mode,
            headless=not args.render,
            cam=args.cam,
        )
        
        icecream.ic(env.horizon, env_name, env.task_description)

        eps = 0
        all_eps = 0
        
        while eps < args.num_episodes:
            all_eps += 1
            print(f"\nCollecting Episode {eps + 1}/{args.num_episodes} for {env_name}")
            
            # Collect episode data
            episode_data = collect_episode_data(
                env, args.max_steps, args.render
            )
            
            # Check if episode was successful
            has_articulator = (hasattr(env.task, 'articulator') and
                             env.task.articulator is not None)
            
            if has_articulator:
                success = env.task.get_progress_state()
            else:
                success = env.task.get_progress_state()
            
            print(f"Task progress: {success:.3f}")
            
            if success >= 1.0:
                eps += 1
                print(f"✓ Episode {eps} succeeded!")
                
                # Save episode data using VPL saver
                if args.save and vpl_saver:
                    try:
                        # Collect episode data in the new format
                        episode_actions = []
                        episode_observations = []
                        episode_rewards = []
                        episode_dones = []
                        
                        for step_data in episode_data:
                            episode_actions.append(step_data["action"])
                            episode_observations.append(step_data["obs"])
                            episode_rewards.append(step_data["reward"])
                            episode_dones.append(step_data["done"])
                        
                        # Store complete episode data
                        vpl_episode_data = {
                            "action": episode_actions,
                            "observation": episode_observations,
                            "reward": episode_rewards,
                            "done": episode_dones
                        }
                        
                        print(f"  Debug: VPL episode data keys: {list(vpl_episode_data.keys())}")
                        print(f"  Debug: Observation type: {type(vpl_episode_data['observation'])}")
                        print(f"  Debug: Observation length: {len(vpl_episode_data['observation'])}")
                        if len(vpl_episode_data['observation']) > 0:
                            print(f"  Debug: First observation keys: {list(vpl_episode_data['observation'][0].keys())}")
                        
                        # Store in VPL saver
                        vpl_saver.store(env, vpl_episode_data, episode_index=eps-1)
                        
                        print(f"  Debug: Episode data keys after store: {list(vpl_saver.episode_data[eps-1].keys())}")
                        
                        print(f"  ✓ Episode {eps} data stored in VPL saver!")
                        
                    except Exception as e:
                        print(f"  ✗ Error saving episode {eps}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("  Episode data not saved (--save flag not set)")
            else:
                print(f"✗ Episode failed (progress: {success:.3f})")
                
            # Check if viewer was closed
            if env.viewer and env.viewer.closed:
                print("Viewer closed, stopping collection")
                break
        
        # Calculate and display success rate
        success_rate[env_name] = eps / all_eps if all_eps > 0 else 0.0
        print(f"\n{env_name} Results:")
        print(f"  Episodes attempted: {all_eps}")
        print(f"  Episodes succeeded: {eps}")
        print(f"  Success rate: {success_rate[env_name]:.2%}")
        
        # Close viewer
        if env.viewer:
            env.viewer.close()
    
    # Final summary
    print(f"\n{'='*50}")
    print("DATA COLLECTION COMPLETED")
    print(f"{'='*50}")
    
    for env_name, rate in success_rate.items():
        print(f"{env_name}: {rate:.2%} ({rate*100:.1f}%)")
    
    # Save all data and close VPL saver
    if args.save and vpl_saver:
        print(f"\nSaving all episodes to VPL format...")
        total_episodes = vpl_saver.save_dataset()
        print(f"✓ Total episodes saved: {total_episodes}")
        vpl_saver.close()
        print(f"✓ VPL data saved to: {args.vpl_dir}")
        print(f"  Data format: Compatible with rai.fm VPLSaver")

if __name__ == "__main__":
    main()
