import sapien.core as sapien
import transforms3d
import numpy as np
import json
import os
from typing import Dict

ROBOT_TABLE_MARGIN_X = 0.06
ROBOT_TABLE_MARGIN_Y = 0.04

def load_camera_config_from_json(json_path: str = "hand_eye_calibration.json") -> Dict:
    """
    Load camera configuration from hand-eye calibration JSON file.
    
    Args:
        json_path: Path to the hand-eye calibration JSON file
        
    Returns:
        Dictionary with camera configurations in the expected format
    """
    # Get the absolute path to the JSON file
    if not os.path.isabs(json_path):
        # If relative path, assume it's in the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        json_path = os.path.join(project_root, json_path)
    
    with open(json_path, 'r') as f:
        calibration_data = json.load(f)
    
    camera_config = {}
    
    for cam_name, cam_data in calibration_data.items():
        # Convert quaternion from [x, y, z, w] to [w, x, y, z] for SAPIEN
        pos = cam_data["pos"]
        quat_xyzw = cam_data["quat"]  # [x, y, z, w]
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]
        
        # Create pose array: [x, y, z, w, x, y, z] (position + quaternion)
        pose = np.array(pos + quat_wxyz)
        
        camera_config[cam_name] = {
            cam_name: dict(
                pose=pose,
                base_frame_id=cam_data["base_frame_id"],
                fov=np.deg2rad(68.8),  # From your provided intrinsics
                resolution=(320, 240),
                serial_number=cam_data["serial_number"]
            )
        }
    
    return camera_config

BOUND_CONFIG = {
    "faucet": [0.1, 2.0, -1.0, 1, -0.1352233 + 0.14, 0.4],
    "bucket": [0.1, 2.0, -2.0, 2.0, -0.29, 0.4],
    "laptop": [0.1, 1.0, -1.0, 2, -0.1352233 + 0.14, 0.6],
    "toilet": [0.1, 2.0, -2.0, 2, -0.3, 0.8],
}

ROBUSTNESS_INIT_CAMERA_CONFIG = {
    "laptop": {
        "r": 1,
        "phi": np.pi / 2,
        "theta": np.pi / 2,
        "center": np.array([0, 0, 0.5]),
    },
}

TRAIN_CONFIG = {
    "faucet": {
        "seen": [148, 693, 822, 857, 991, 1011, 1053, 1288, 1343, 1370, 1466],
        "unseen": [1556, 1633, 1646, 1667, 1741, 1832, 1925],
    },
    "faucet_half": {
        "seen": [148, 693, 822, 857, 991],
        "unseen": [1556, 1633, 1646, 1667, 1741, 1832, 1925],
    },
    "bucket": {
        "seen": [
            100431,
            100435,
            100438,
            100439,
            100441,
            100444,
            100446,
            100448,
            100454,
            100461,
            100462,
        ],
        "unseen": [100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358],
    },
    "bucket_half": {
        "seen": [100431, 100435, 100438, 100439, 100441],
        "unseen": [100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358],
    },
    "laptop": {
        "seen": [
            11395,
            11405,
            11406,
            11477,
            11581,
            11586,
            9996,
            10090,
            10098,
            10101,
            10125,
        ],
        "unseen": [9748, 9912, 9918, 9960, 9968, 9992],
    },
    "laptop_half": {
        "seen": [11395, 11405, 11406, 11477, 11581],
        "unseen": [9748, 9912, 9918, 9960, 9968, 9992],
    },
    "toilet": {
        "seen": [
            102677,
            102687,
            102689,
            102692,
            102697,
            102699,
            102701,
            102703,
            102707,
            102708,
            103234,
            102663,
            102666,
            102667,
            102669,
            102670,
            102675,
        ],
        "unseen": [
            101320,
            102621,
            102622,
            102630,
            102634,
            102645,
            102648,
            102651,
            102652,
            102654,
            102658,
        ],
    },
    "toilet_half": {
        "seen": [102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703],
        "unseen": [
            101320,
            102621,
            102622,
            102630,
            102634,
            102645,
            102648,
            102651,
            102652,
            102654,
            102658,
        ],
    },
}

TASK_CONFIG = {
    "faucet": [
        148,
        693,
        822,
        857,
        991,
        1011,
        1053,
        1288,
        1343,
        1370,
        1466,
        1556,
        1633,
        1646,
        1667,
        1741,
        1832,
        1925,
    ],
    "bucket": [
        100431,
        100435,
        100438,
        100439,
        100441,
        100444,
        100446,
        100448,
        100454,
        100461,
        100462,
        100468,
        100470,
        100473,
        100482,
        100484,
        100486,
        102352,
        102358,
    ],
    "laptop": [
        9748,
        9912,
        9918,
        9960,
        9968,
        9992,
        11395,
        11405,
        11406,
        11477,
        11581,
        11586,
        9996,
        10090,
        10098,
        10101,
        10125,
    ],
    "toilet": [
        101320,
        102621,
        102622,
        102630,
        102634,
        102645,
        102648,
        102651,
        102652,
        102654,
        102658,
        102677,
        102687,
        102689,
        102692,
        102697,
        102699,
        102701,
        102703,
        102707,
        102708,
        103234,
        102663,
        102666,
        102667,
        102669,
        102670,
        102675,
    ],
}

# Camera config - Load from hand-eye calibration JSON file
CAMERA_CONFIG = load_camera_config_from_json()
    # "faucet": {
    #     "instance_1": dict(
    #         pose=np.concatenate([np.array([0, 0.8, 0.5]), transforms3d.euler.euler2quat(np.pi / 3, np.pi, 0)]),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     )
    #     # "instance_1": dict(
    #     #     pose=np.concatenate([np.array([0, 1, 0.5]), transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)]),
    #     #     fov=np.deg2rad(69.4),
    #     #     resolution=(320, 240),
    #     # ),
    # },
    # "bucket": {
    #     "instance_1": dict(
    #         pose=np.concatenate([np.array([0, 1, 0.5]), transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)]),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     ),
    # },
    # "laptop": {
    #     "instance_1": dict(
    #         pose=np.concatenate([np.array([0, 1, 0.5]), transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)]),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     ),
    # },
    # "toilet": {
    #     "instance_1": dict(
    #         pose=np.concatenate([np.array([0, 1, 0.5]), transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)]),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     ),
    # },
    # "default": {
    #     "instance_1": dict(
    #         pose=np.concatenate([np.array([0, 1, 0.5]), transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)]),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     ),
    # },
    # "default": {
    #     "default_cam": dict(
    #         pose=np.concatenate(
    #             [
    #                 np.array([0, -0.8, 0.5]),
    #                 transforms3d.euler.euler2quat(-2 * np.pi / 3, 0, 0),
    #             ]
    #         ),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     )
    # },
    # "viz": {
    #     "viz_cam": dict(
    #         pose=np.array(
    #             [0.375, 0.696, 0.6, 0.412333, 0.212321, 0.0994459, -0.880348]
    #         ),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     )
    # },
    # "wrist": {
    #     "wrist_cam": dict(
    #         pose=np.array(
    #             [
    #                 0.07552634,
    #                 -0.02599394,
    #                 0.02404529,
    #                 0.73417488,
    #                 0.02648626,
    #                 -0.00778071,
    #                 0.67839898,
    #             ]
    #         ),
    #         fov=[np.deg2rad(87), np.deg2rad(58)],
    #         resolution=(320, 240),
    #         mount_actor_name="panda_hand",  # "panda_hand",
    #         noise_scale=0.5,
    #     )
    # },
    # "top": {
    #     "top_cam": dict(
    #         pose=np.concatenate(
    #             [
    #                 np.array([0, 0, 1.5]),
    #                 transforms3d.euler.euler2quat(np.pi, 0, -np.pi / 2),
    #             ]
    #         ),
    #         fov=np.deg2rad(69.4),
    #         resolution=(320, 240),
    #     )
    # },
    # "left": {
    #     "left_cam": dict(
    #         pose=np.array(
    #             [
    #                 -0.19207954,
    #                 0.59551661,
    #                 0.53830318,
    #                 -0.01238816,
    #                 -0.01772414,
    #                 0.92530522,
    #                 -0.37860617,
    #             ]
    #         ),
    #         fov=[np.deg2rad(87), np.deg2rad(58)],
    #         resolution=(320, 240),
    #         noise_scale=0.5,
    #     )
    # },
    # "right": {
    #     "right_cam": dict(
    #         pose=np.array(
    #             [
    #                 -0.09048795,
    #                 -0.61669572,
    #                 0.43609362,
    #                 -0.47339763,
    #                 0.88004426,
    #                 0.03741855,
    #                 -0.00407857,
    #             ]
    #         ),
    #         fov=[np.deg2rad(87), np.deg2rad(58)],
    #         resolution=(320, 240),
    #         noise_scale=0.5,
    #     )
    # },
    # Multi-camera setup with provided extrinsics
    # "camera0": {
    #     "camera0": dict(
    #         pose=np.array([
    #             -0.022565587794617703,      # x
    #             -0.662003675079514,         # y  
    #             0.593798049759222,          # z
    #             0.8335989637460393,         # w (moved from end)
    #             -0.17889949739264788,       # x (quaternion)
    #             0.3914096246504936,         # y (quaternion)
    #             0.34627480879374745         # z (quaternion)
    #         ]),
    #         base_frame_id="fr3_link0",
    #         fov=np.deg2rad(120),  # From horizontal_aperture=1.658, focal_length=1.0
    #         resolution=(320, 240),
    #         serial_number="215122251334"
    #     )
    # },
    # "camera1": {
    #     "camera1": dict(
    #         pose=np.array([
    #             0.6899072165113053,         # x
    #             -0.6858108131755757,        # y
    #             0.2398100897303653,         # z
    #             0.3667691712090961,         # w (moved from end)
    #             -0.25121055631838207,       # x (quaternion)
    #             0.09807786562229344,        # y (quaternion)
    #             0.8903675441747507          # z (quaternion)
    #         ]),
    #         base_frame_id="fr3_link0",
    #         fov=np.deg2rad(120),  # From horizontal_aperture=1.658, focal_length=1.0
    #         resolution=(320, 240),
    #         serial_number="213622252200"
    #     )
    # },
    # "camera2": {
    #     "camera2": dict(
    #         pose=np.array([
    #             0.693868725700083,          # x
    #             -0.6856771396231571,        # y
    #             0.599392445222885,          # z
    #             0.3429591117809985,         # w (moved from end)
    #             -0.4023461474814715,        # x (quaternion)
    #             0.1651437453325628,         # y (quaternion)
    #             0.8326008459224264          # z (quaternion)
    #         ]),
    #         base_frame_id="fr3_link0",
    #         fov=np.deg2rad(120),  # From horizontal_aperture=1.658, focal_length=1.0
    #         resolution=(320, 240),
    #         serial_number="231622302195"
    #     )
    # },
    # "camera3": {
    #     "camera3": dict(
    #         pose=np.array([
    #             0.8756826746675879,         # x
    #             0.052999383161436285,       # y
    #             0.5963114855382364,         # z
    #             -0.36568702407213793,       # w (moved from end)
    #             -0.3930176278604368,        # x (quaternion)
    #             -0.1521161465105955,        # y (quaternion)
    #             0.8298619298335029          # z (quaternion)
    #         ]),
    #         base_frame_id="fr3_link0",
    #         fov=np.deg2rad(120),  # From horizontal_aperture=1.658, focal_length=1.0
    #         resolution=(320, 240),
    #         serial_number="242422302772"
    #     )
    # },
    # TODO: Do we need this? We already defined the viewer in sim.py
#     "viz_only": {  # only for visualization (human), not for visual observation
#         "faucet_viz": dict(
#             position=np.array([-0.3, 0.6, 0.4]),
#             look_at_dir=np.array([0.16, -0.7, -0.35]),
#             # from left side.
#             right_dir=np.array([-1.5, -2, 0]),
#             fov=np.deg2rad(69.4),
#             resolution=(320, 240),
#         ),
#         "faucet_viz2": dict(
#             pose=np.concatenate(
#                 [
#                     np.array([0, 0.8, 0.5]),
#                     transforms3d.euler.euler2quat(np.pi / 3, np.pi, 0),
#                 ]
#             ),
#             fov=np.deg2rad(69.4),
#             resolution=(320, 240),
#         ),
#         "bucket_viz": dict(
#             pose=np.concatenate(
#                 [
#                     np.array([0, 1, 0.5]),
#                     transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0),
#                 ]
#             ),
#             fov=np.deg2rad(69.4),
#             resolution=(320, 240),
#         ),
#         "laptop_viz": dict(
#             pose=np.concatenate(
#                 [
#                     np.array([0, 1, 0.5]),
#                     transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0),
#                 ]
#             ),
#             fov=np.deg2rad(69.4),
#             resolution=(320, 240),
#         ),
#         "toilet_viz": dict(
#             pose=np.concatenate(
#                 [
#                     np.array([0, 1, 0.5]),
#                     transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0),
#                 ]
#             ),
#             fov=np.deg2rad(69.4),
#             resolution=(320, 240),
#         ),
#         "default_viz": dict(
#             pose=np.concatenate(
#                 [
#                     np.array([0, 1, 0.5]),
#                     transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0),
#                 ]
#             ),
#             fov=np.deg2rad(69.4),
#             resolution=(320, 240),
#         ),
#     },
# }

EVAL_CAM_NAMES_CONFIG = {
    "faucet": ["faucet_viz"],
    "bucket": ["bucket_viz"],
    "laptop": ["laptop_viz"],
    "toilet": ["toilet_viz"],
    "default": ["default_viz"],
}

# Observation config type
OBS_CONFIG = {
    "instance_rgb": {
        "instance_1": {"rgb": True},
    },
    "instance": {
        "instance_1": {"point_cloud": {"num_points": 512}},
        # "hand": {"point_cloud": {"process_fn": 1, "num_points": 512}},
    },
    "instance_noise": {
        "instance_1": {
            "point_cloud": {
                "num_points": 512,
                "pose_perturb_level": 0.5,
                "process_fn_kwargs": {"noise_level": 0.5},
            },
        },
    },
    "instance_pc_seg": {
        "instance_1": {
            "point_cloud": {
                "use_seg": True,
                "use_2frame": True,
                "num_points": 512,
                "pose_perturb_level": 0.5,
                "process_fn_kwargs": {"noise_level": 0.5},
            },
        },
    },
}

# Imagination config type
IMG_CONFIG = {
    "robot": {
        "robot": {
            # "link_base": 8, "link1": 8, "link2": 8, "link3": 8, "link4": 8, "link5": 8, "link6": 8,
            "link_15.0_tip": 8,
            "link_3.0_tip": 8,
            "link_7.0_tip": 8,
            "link_11.0_tip": 8,
            "link_15.0": 8,
            "link_3.0": 8,
            "link_7.0": 8,
            "link_11.0": 8,
            "link_14.0": 8,
            "link_2.0": 8,
            "link_6.0": 8,
            "link_10.0": 8,  # "base_link": 8
        },
    }
}

RANDOM_CONFIG = {
    "bucket": {"rand_pos": 0.05, "rand_degree": 0},
    "laptop": {"rand_pos": 0.1, "rand_degree": 60},
    "faucet": {"rand_pos": 0.1, "rand_degree": 90},
    "toilet": {"rand_pos": 0.2, "rand_degree": 45},
}
