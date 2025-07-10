#!/usr/bin/env python3

import numpy as np
import cv2
import time
import socket
import pickle
import threading
from multiprocessing import Process, Queue
import freenect
from pathlib import Path
import torch
import einops

from transformers import AutoTokenizer
# UR10e specific imports
from UR10e.rtde import UR10eJoystickControl

class CameraManager:
    def __init__(self, fps=15):
        self.fps = fps
        self.rgb_frame = None
        self.depth_frame = None
        self.running = False
        self.initialized = False
        self.frame_lock = threading.Lock()
        
    def start(self):
        """Start camera capture thread"""
        print("[CAMERA] Starting camera capture thread...", flush=True)
        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Wait for initialization
        timeout = 10.0
        start_time = time.time()
        while not self.initialized and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.initialized:
            print("[CAMERA] Camera initialized successfully", flush=True)
        else:
            print("[CAMERA]  Camera initialization timeout", flush=True)
        
    def stop(self):
        """Stop camera capture"""
        print("[CAMERA] Stopping camera capture...", flush=True)
        self.running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=2.0)
        print("[CAMERA]  Camera stopped", flush=True)
        
    def _camera_loop(self):
        """Camera capture loop"""
        target_dt = 1.0 / self.fps
        
        while self.running:
            loop_start = time.time()
            
            try:
                rgb_result = freenect.sync_get_video()
                depth_result = freenect.sync_get_depth()
                
                rgb = rgb_result[0] if isinstance(rgb_result, tuple) else rgb_result
                depth = depth_result[0] if isinstance(depth_result, tuple) else depth_result
                
                if rgb is None:
                    print("No valid frames")
                    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                if depth is None:
                    print("no valid frames")
                    depth = np.zeros((480, 640), dtype=np.uint16)
                
                with self.frame_lock:
                    self.rgb_frame = rgb.copy()
                    self.depth_frame = depth.copy()
                    if not self.initialized:
                        self.initialized = True
                        print("[CAMERA] ✓ First frames captured", flush=True)
                    
            except Exception as e:
                print(f"[CAMERA] Capture error: {e}", flush=True)
                with self.frame_lock:
                    self.rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    self.depth_frame = np.zeros((480, 640), dtype=np.uint16)
                    if not self.initialized:
                        self.initialized = True
            
            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_latest_frames(self):
        """Get latest camera frames"""
        with self.frame_lock:
            if self.rgb_frame is not None and self.depth_frame is not None:
                return self.rgb_frame.copy(), self.depth_frame.copy()
            else:
                print("NO valid frames in get_latest_frames")
                return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640), dtype=np.uint16)

class RobotPolicyServer:
    def __init__(self, host='0.0.0.0', port=5000, fps=15):
        print(f"[SERVER] Initializing Robot Policy Server", flush=True)
        print(f"[SERVER] Server address: {host}:{port}", flush=True)
        print(f"[SERVER] Target FPS: {fps}", flush=True)
        
        self.host = host
        self.port = port
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Initialize camera
        self.camera_manager = CameraManager(fps=fps)
        
        # Initialize robot
        self.robot = None
        self._initialize_robot()
        
        # Define observation features
        self.obs_features = {
            "observation.kinect_rgb": {
                "dtype": "video",
                "shape": (160, 120, 3),
                "names": ["height", "width", "channel"]
            },
            "observation.kinect_depth": {
                "dtype": "video",
                "shape": (160, 120, 3),
                "names": ["height", "width", "channel"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["x", "y", "z"]
            }#,
            # "task": {
            #     "dtype": "string",
            #     "shape": (1,),
            #     "names": None
            # },
        }
        
        # Server state
        self.running = False
        self.client_socket = None
        self.server_socket = None
        
    def _initialize_robot(self):
        """Initialize robot connection"""
        try:
            print("[SERVER] Connecting to UR10e robot...", flush=True)
            self.robot = UR10eJoystickControl("192.168.1.102")
            robot_connected = self.robot.connect_robot()
            
            if robot_connected:
                print("[SERVER]  Robot connected successfully", flush=True)
                self.robot.gripkit_connect()
                
                self.home_pose = self.robot.get_robot_state()
            else:
                raise ConnectionError("Failed to connect to robot")
                
        except Exception as e:
            print(f"[SERVER]  Robot initialization error: {e}", flush=True)
            self.robot = None
            raise
    
    def _process_depth_image(self, depth_image):
        """Process depth image for policy input"""
        # Convert to float32 and normalize to meters
        depth_meters = depth_image.astype(np.float32) / 1000.0
        
        # Clip to reasonable range and normalize
        depth_clipped = np.clip(depth_meters, 0.0, 5.0)
        depth_normalized = depth_clipped / 5.0
        
        # Convert to 3-channel and scale to 0-255
        depth_3d = np.repeat(depth_normalized[:, :, np.newaxis], 3, axis=2)
        depth_uint8 = (depth_3d * 255).astype(np.uint8)
        
        return depth_uint8
    def _get_observation(self):
        """Get current observation from robot and cameras"""
        state = self.robot.get_robot_state()[:3]
        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        else:
            state = state.astype(np.float32)
            
        rgb, depth = self.camera_manager.get_latest_frames()
        # Ensure RGB is a numpy array
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb, dtype=np.uint8)
        
        rgb_resized = cv2.resize(rgb, (160, 120))
        depth_processed = self._process_depth_image(depth)
        depth_resized = cv2.resize(depth_processed, (160, 120))

        obs = {
            "observation.kinect_rgb": rgb_resized,
            "observation.kinect_depth": depth_resized,
            "observation.state": state,
        }

        
        return obs


    
    def _build_observation_frame(self, observations):
        """Convert environment observation to LeRobot format observation."""
        return_observations = {}
        
        # Handle kinect RGB observation
        if "observation.kinect_rgb" in observations:
            img = observations["observation.kinect_rgb"]
            # Ensure image is uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = torch.from_numpy(img)
            # Add batch dimension if needed
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # Convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255
            return_observations["observation.image"] = img

        # Handle kinect depth observation
        if "observation.kinect_depth" in observations:
            depth = observations["observation.kinect_depth"]
            # Ensure depth is uint8 (after normalization)
            if depth.dtype != np.uint8:
                depth = (depth * 255).astype(np.uint8)
            depth = torch.from_numpy(depth)
            # Add batch dimension if needed
            if depth.ndim == 3:
                depth = depth.unsqueeze(0)
            # Convert to channel first of type float32 in range [0,1]
            depth = einops.rearrange(depth, "b h w c -> b c h w").contiguous()
            depth = depth.type(torch.float32)
            depth /= 255
            return_observations["observation.kinect_depth"] = depth

        # Handle state observation
        if "observation.state" in observations:
            state = torch.from_numpy(observations["observation.state"]).float()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return_observations["observation.state"] = state


        #return_observations["task"] = ["pick up the block"]
        
        return return_observations
    
    def _send_observation(self, observation_frame):
        """Send observation data to client"""
        try:
            print("[SERVER] Sending observation to client...", flush=True)
            
            # Serialize observation
            obs_data = pickle.dumps(observation_frame)
            
            # Send size first
            size_bytes = len(obs_data).to_bytes(4, byteorder='big')
            self.client_socket.sendall(size_bytes)
            
            # Send observation data
            self.client_socket.sendall(obs_data)
            
            print(f"[SERVER] Observation sent ({len(obs_data)} bytes)", flush=True)
            
        except Exception as e:
            print(f"[SERVER]  Error sending observation: {e}", flush=True)
            raise
    
    def _receive_action(self):
        """Receive action from client"""
        try:
            print("[SERVER] Waiting for action from client...", flush=True)
            
            # Receive size first
            size_data = b""
            while len(size_data) < 4:
                chunk = self.client_socket.recv(4 - len(size_data))
                if not chunk:
                    raise ConnectionError("Client closed connection")
                size_data += chunk
            
            data_size = int.from_bytes(size_data, byteorder='big')
            print(f"[SERVER] Expecting {data_size} bytes of action data", flush=True)
            
            # Receive action data
            action_data = b""
            while len(action_data) < data_size:
                chunk = self.client_socket.recv(data_size - len(action_data))
                if not chunk:
                    raise ConnectionError("Client closed connection")
                action_data += chunk
            
            # Deserialize action
            action = pickle.loads(action_data)
            print(f"[SERVER] ✓ Received action: {action}", flush=True)
            
            return action
            
        except Exception as e:
            print(f"[SERVER] Error receiving action: {e}", flush=True)
            raise
    
    def _execute_action(self, action):
        """Execute action on robot"""
        try:
            if self.robot is None:
                print("[SERVER]  No robot connection, skipping action execution", flush=True)
                return
            
            if action is None:
                print("None action- not executing")
                return 
            if isinstance(action, (list, np.ndarray)) and len(action) >= 4:
                action_xyz = action[:3]
                gripper_cmd = action[3]
                
                print(f"[SERVER] Executing - XYZ: {action_xyz}, Gripper: {gripper_cmd:.3f}", flush=True)
                
                # Send actions to robot
                self.robot.send_tcp_action(action_xyz)
                self.robot.control_gripper(gripper_cmd)
                
                print("[SERVER]  Action executed successfully", flush=True)
            else:
                print(f"[SERVER]  Invalid action format: {action}", flush=True)
                
        except Exception as e:
            print(f"[SERVER]  Error executing action: {e}", flush=True)

    
    def _handle_client_connection(self):
        print("[SERVER] ✓ Client connected, starting control loop...", flush=True)
        cycle_count = 0
        
        try:
            while self.running:
                loop_start_time = time.time()
                
                # Get observation
                print("getting obs")
                obs = self._get_observation()

                try:
                    observation_frame = self._build_observation_frame(obs)
                except Exception as e:
                    print(f"Error during frame building: {e}", flush=True)
                    raise
                
        
                # Send observation to client
                self._send_observation(observation_frame)
                
                # Receive action from client
                action = self._receive_action()
                
                # Execute action on robot
                self._execute_action(action)
                
                cycle_count += 1
                if cycle_count % 50 == 0:
                    print(f"[SERVER] Completed {cycle_count} control cycles", flush=True)
                
                # Maintain FPS
                elapsed_time = time.time() - loop_start_time
                sleep_duration = self.dt - elapsed_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                
        except KeyboardInterrupt:
            print("\n[SERVER] KeyboardInterrupt received", flush=True)
        except Exception as e:
            print(f"[SERVER] Error in client connection: {e}", flush=True)
        finally:
            print("[SERVER] Client connection ended", flush=True)
    
    def start_server(self):
        print(f"[SERVER] Starting server on {self.host}:{self.port}...", flush=True)
        
        try:
            # Start camera
            self.camera_manager.start()
            
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            print(f"[SERVER] Server listening on {self.host}:{self.port}", flush=True)
            print("[SERVER] Waiting for client connections...", flush=True)
            
            self.running = True
            
            while self.running:
                try:
                    # Accept client connection
                    print("[SERVER] Waiting for client...", flush=True)
                    self.client_socket, client_address = self.server_socket.accept()
                    print(f"[SERVER]  Client connected from {client_address}", flush=True)
                    
                    # Set socket timeout
                    self.client_socket.settimeout(30.0)
                    
                    # Handle client
                    self._handle_client_connection()
                    
                except socket.timeout:
                    print("[SERVER] Socket timeout, continuing...", flush=True)
                    continue
                except Exception as e:
                    print(f"[SERVER]  Error accepting connection: {e}", flush=True)
                    time.sleep(1)
                finally:
                    if self.client_socket:
                        try:
                            self.client_socket.close()
                        except:
                            pass
                        self.client_socket = None
                    print("[SERVER] Client disconnected", flush=True)
            
        except KeyboardInterrupt:
            print("\n[SERVER] KeyboardInterrupt received, shutting down...", flush=True)
        except Exception as e:
            print(f"[SERVER] Server error: {e}", flush=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("[SERVER] Cleaning up...", flush=True)
        self.running = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        self.camera_manager.stop()
        
        # Disconnect robot
        if self.robot:
            try:
                self.robot.disconnect_robot()
            except:
                pass
        
        # Close any OpenCV windows
        cv2.destroyAllWindows()
        
        print("[SERVER] ✓ Cleanup complete", flush=True)



if __name__ == "__main__":
    try:
        server = RobotPolicyServer(host='0.0.0.0', port=5000, fps=15)
        server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Server error: {e}")
