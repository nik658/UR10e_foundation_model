#!/usr/bin/env python3

import numpy as np
import cv2
import time
import threading
import freenect
from pathlib import Path
import torch

# UR10e specific imports
from UR10e.rtde import UR10eJoystickControl

# LeRobot imports
from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.utils import get_safe_torch_device

class CameraFeedManager:
    def __init__(self, fps=15):
        self.fps = fps
        self.rgb_frame = None
        self.depth_frame = None
        self.running = False
        self.camera_thread = None
        self.frame_lock = threading.Lock()
        self.initialized = False
        
    def start_camera_thread(self):
        """Start camera capture in separate thread"""
        print("Starting camera thread...")
        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Wait for first frames to be captured
        timeout = 5.0  # Wait up to 5 seconds
        start_time = time.time()
        while not self.initialized and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.initialized:
            print("✓ Camera feed thread started and initialized")
        else:
            print("⚠ Camera thread started but no frames received within timeout")
        
    def _camera_loop(self):
        """Camera capture loop running in separate thread"""
        target_dt = 1.0 / self.fps
        
        while self.running:
            loop_start = time.time()
            
            try:
                rgb, _ = freenect.sync_get_video()
                depth, _ = freenect.sync_get_depth()
                
                if rgb is None:
                    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                if depth is None:
                    depth = np.zeros((480, 640), dtype=np.uint16)
                
                with self.frame_lock:
                    self.rgb_frame = rgb.copy()
                    self.depth_frame = depth.copy()
                    if not self.initialized:
                        self.initialized = True
                        print("✓ First camera frames captured")
                    
            except Exception as e:
                print(f"Camera capture error: {e}")
                with self.frame_lock:
                    self.rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    self.depth_frame = np.zeros((480, 640), dtype=np.uint16)
                    if not self.initialized:
                        self.initialized = True  # Initialize with dummy frames
            
            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_latest_frames(self):
        """Get the latest camera frames (thread-safe)"""
        with self.frame_lock:
            if self.rgb_frame is not None and self.depth_frame is not None:
                return self.rgb_frame.copy(), self.depth_frame.copy()
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640), dtype=np.uint16)
    
    def display_frames(self):
        """Display camera frames in OpenCV windows (non-blocking)"""
        rgb_frame, depth_frame = self.get_latest_frames()
        
        # Display RGB
        cv2.imshow('UR10e Kinect RGB', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        
        # Display depth as colored image
        depth_normalized = np.clip(depth_frame / 2048.0, 0, 1)
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        cv2.imshow('UR10e Kinect Depth', depth_colored)
        
        # Non-blocking waitKey
        key = cv2.waitKey(1) & 0xFF
        return key
    
    def stop(self):
        """Stop camera thread"""
        print("Stopping camera thread...")
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        print("✓ Camera feed thread stopped")

class UR10ePolicyRunner:
    def __init__(self, policy_path="pretrained_model", fps=15):
        print("Initializing UR10e Policy Runner...")
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Initialize camera FIRST
        print("Initializing camera...")
        self.camera_manager = CameraFeedManager(fps=fps)
        self.camera_manager.start_camera_thread()
        
        # Initialize robot connection
        print("Connecting to UR10e robot...")
        self.robot = UR10eJoystickControl("192.168.1.102")
        robot_connected = self.robot.connect_robot()
        
        if robot_connected:
            print("✓ Robot connected successfully")
            # Connect to gripper
            self.robot.gripkit_connect()
            
            # Store home pose
            self.home_pose = self.robot.get_robot_state()
            print(f"✓ Home pose stored: {self.home_pose}")
        else:
            raise ConnectionError("Failed to connect to robot. Cannot run policy.")
        
        # Load policy
        print(f"Loading policy from: {policy_path}")
        self.device = torch.device('cpu')
        self.policy = PI0FASTPolicy.from_pretrained(policy_path)
        self.policy = self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        print("✓ Policy loaded successfully")
        
        # Define observation features to match training data
        self.obs_features = {
            "observation.kinect_rgb": {
                "dtype": "video",
                "shape": (120, 160, 3),
                "names": ["height", "width", "channel"]
            },
            "observation.kinect_depth": {
                "dtype": "video",
                "shape": (120, 160, 3),
                "names": ["height", "width", "channel"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["x", "y", "z",]
            },
            "task": {
                "dtype": "string",
                "shape": (1,),
                "names": None
            },
        }
        
        self.running = True
        self.prev_exit_button = False
        
        print("✓ UR10e Policy Runner initialized!")
        print("--- Controls ---")
        print("- 'q' key: Exit")
        print("- ESC key: Exit")
        print("----------------")

    def get_observation(self):
        """Get current observation from robot and cameras"""
        state = self.robot.get_robot_state()
        rgb, depth = self.camera_manager.get_latest_frames()
        
        # Normalize depth for policy input (same as during training)
        depth_normalized = self.normalize_depth_for_policy(depth)
        
        return {
            "observation.kinect_rgb": rgb,
            "observation.kinect_depth": depth_normalized,
            "observation.state": state,
            "x": state[0],
            "y": state[1],
            "z": state[2],
            "rx": state[3],
            "ry": state[4],
            "rz": state[5],
        }

    def normalize_depth_for_policy(self, depth_image):
        """Normalize depth image for policy input (same as during training)"""
        # Convert to float32 and normalize to meters (assuming mm input)
        depth_meters = depth_image.astype(np.float32) / 1000.0
        
        # Clip to reasonable range (0 to 5 meters) and normalize to [0, 1]
        depth_clipped = np.clip(depth_meters, 0.5, 3.0)
        depth_normalized = depth_clipped / 2.5
        
        # Convert to 3-channel by repeating the depth values
        depth_3d = np.repeat(depth_normalized[:, :, np.newaxis], 3, axis=2)
        
        return depth_3d.astype(np.float32)

    def build_dataset_frame(self, values):
        """Build observation frame in LeRobot format"""
        frame = {}
        for key, ft in self.obs_features.items():
            if ft["dtype"] == "video":
                img = values[key]
                # Convert HWC -> CHW
                if img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1))
                frame[key] = img.astype(np.float32)
            elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
                # Build state from named components
                frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        
        return frame

    def run(self):
        """Main execution loop to run policy on robot"""
        print("Starting policy execution loop...")
        cycle_count = 0
        self.previous_action = np.zeros(4, dtype=np.float32)  # [dx, dy, dz, gripper]

        try:
            while self.running:
                loop_start_time = time.time()

                # Get observation & policy action
                obs = self.get_observation()
                observation_frame = self.build_dataset_frame(obs)
                action_values = self.policy.select_action(
                    observation_frame,
                    task="Pick and place the block"
                )

                # Smooth the action
                action = np.array([
                    action_values[0].item(),
                    action_values[1].item(),
                    action_values[2].item(),
                    action_values[3].item()
                ])

                # Display camera feeds and check for exit
                #key = self.camera_manager.display_frames()
                #cv2.setWindowTitle('UR10e Kinect RGB', f'Cycle {cycle_count}')
                
                # Check for exit keys
                #if key == ord('q') or key == 27:  # 'q' or ESC
                    #print("Exit key pressed")
                   # self.running = False
                   # break
                
                # Send actions to robot
                action_xyz = action[:3]
                gripper_cmd = action[3]
                print("sent actionto robot and elapsed time is", cycle_count , time.time()-loop_start_time)
                self.robot.send_tcp_action(action_xyz)
                self.robot.control_gripper(gripper_cmd)
                
                if cycle_count % 50 == 0:  # Minimal debug every 50 cycles
                    print(f"C{cycle_count}: TCP{action_xyz[:3]}, G{gripper_cmd:.2f}")

                # Maintain FPS
                elapsed_time = time.time() - loop_start_time
                sleep_duration = self.dt - elapsed_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                cycle_count += 1

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("Cleaning up...")
        self.camera_manager.stop()
        self.robot.disconnect_robot()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    POLICY_PATH = "Pi0_train/pretrained_model"  # Path to your trained policy
    FPS = 15
    
    print("=== UR10e Policy Runner ===")
    print(f"Policy path: {POLICY_PATH}")
    print(f"Target FPS: {FPS}")
    
    try:
        runner = UR10ePolicyRunner(policy_path=POLICY_PATH, fps=FPS)
        runner.run()
    except Exception as e:
        print(f"[CRITICAL ERROR] Policy runner failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Script execution finished.")



#(base) nik@nik-Legion-5-15ACH6H:~$ ssh -R 8080:192.168.1.101:50007 nikhilesh@pleiades.ieeta.pt


