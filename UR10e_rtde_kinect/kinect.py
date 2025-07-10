#!/usr/bin/env python3
"""
Simple Kinect 360 Viewer - Separate Windows
Shows RGB and depth in separate windows
"""

import freenect
import cv2
import numpy as np

def display_kinect():
    """Display Kinect RGB and depth streams in separate windows"""
    
    print("Starting Kinect viewer...")
    print("Press 'q' in any window to quit")
    
    while True:
        # Get RGB frame
        try:
            rgb, _ = freenect.sync_get_video()
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except:
            print("Failed to get RGB frame")
            continue
        
        # Get depth frame
        try:
            depth, _ = freenect.sync_get_depth()
            
            depth_normalized = np.clip(depth / 2048.0, 0, 1)
            depth_8bit = (depth_normalized * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
            
        except:
            print("Failed to get depth frame")
            continue
        
        # Display frames
        cv2.imshow('Kinect RGB', rgb_bgr)
        #cv2.imshow('Kinect Depth', depth_8bit)
        cv2.imshow('Kinect Depth Colored', depth_colored)  # If using colorized version
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_kinect()