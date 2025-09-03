#!/usr/bin/env python3
"""Script to update camera resolutions for VPL data collection."""

import re

def update_camera_resolution(file_path, new_width=1280, new_height=720):
    """Update camera resolutions in task_setting.py"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match resolution=(width, height)
    pattern = r'resolution=\(\d+,\s*\d+\)'
    replacement = f'resolution=({new_width}, {new_height})'
    
    # Replace all resolution settings
    updated_content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated all camera resolutions to ({new_width}, {new_height})")
    print(f"File: {file_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Update camera resolutions')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--square', action='store_true', help='Use square resolution (1024x1024)')
    
    args = parser.parse_args()
    
    if args.square:
        width, height = 1024, 1024
    else:
        width, height = args.width, args.height
    
    file_path = "gensim2/env/base/task_setting.py"
    update_camera_resolution(file_path, width, height)
