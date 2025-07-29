#!/usr/bin/env python3

import numpy as np
import cv2
import time
import multiprocessing as mp
import threading
import freenect
import torch
import einops
from copy import deepcopy
from UR10e.rtde_withoutjoy import UR10eJoystickControl
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device

# -------------------- CAMERA -------------------- #

def find_video_devices():
    """Find available video devices"""
    available_devices = []
    for i in range(10):  # Check /dev/video0 through /dev/video9
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            available_devices.append(f"/dev/video{i}")
            cap.release()
    return available_devices

def _camera_worker(fps, out_q, stop_evt):
    dt = 1.0 / fps
    
    # Initialize C922 webcam - try both /dev/video2 and /dev/video3
    c922_cap = None
    for video_device in ["/dev/video2", "/dev/video3"]:
        c922_cap = cv2.VideoCapture(video_device)
        c922_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c922_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if c922_cap.isOpened():
            print(f"C922 webcam initialized successfully on {video_device}")
            break
        else:
            c922_cap.release()
            c922_cap = None
    
    if c922_cap is None:
        print("[ERROR] Cannot open C922 webcam on any device")
    else:
        print("C922 webcam ready")
    
    while not stop_evt.is_set():
        t0 = time.time()
        try:
            # Get Kinect RGB
            rgb, _ = freenect.sync_get_video()
            rgb = cv2.resize(rgb, (640, 480))  # Note: cv2.resize uses (width, height)
            
            # Get C922 frame
            c922_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Default fallback
            if c922_cap is not None:
                ret, frame = c922_cap.read()
                if ret:
                    c922_frame = cv2.resize(frame, (640, 480))
                    # Convert BGR to RGB for consistency
                    c922_frame = cv2.cvtColor(c922_frame, cv2.COLOR_BGR2RGB)
                    
        except Exception as e:
            print(f"[CAMERA] error: {e}")
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            c922_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            out_q.put_nowait((rgb, c922_frame))
        except mp.queues.Full:
            pass

        t_sleep = dt - (time.time() - t0)
        if t_sleep > 0:
            time.sleep(t_sleep)
    
    # Cleanup
    if c922_cap is not None:
        c922_cap.release()

class CameraFeedManager:
    def __init__(self, fps=15, queue_size=2):
        self.fps = fps
        self._queue = mp.Queue(maxsize=queue_size)
        self._stop_evt = mp.Event()
        self._proc = None
        self._latest_rgb = None
        self._latest_c922 = None

    def start_camera_thread(self):
        print("[CAMERA] starting capture process...")
        print(f"[CAMERA] Available video devices: {find_video_devices()}")
        self._proc = mp.Process(
            target=_camera_worker,
            args=(self.fps, self._queue, self._stop_evt),
            daemon=True,
        )
        self._proc.start()

        for _ in range(50):
            if not self._queue.empty():
                self._latest_rgb, self._latest_c922 = self._queue.get()
                print("✓ camera feed initialized")
                return
            time.sleep(0.1)
        print("⚠ camera initialization timeout")

    def stop(self):
        self._stop_evt.set()
        if self._proc is not None:
            self._proc.join(timeout=2)

    def get_latest_frames(self):
        while not self._queue.empty():
            self._latest_rgb, self._latest_c922 = self._queue.get_nowait()
        if self._latest_rgb is None or self._latest_c922 is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)
        return self._latest_rgb.copy(), self._latest_c922.copy()

# --------------- PREPROCESSING FUNCTIONS ---------------- #

def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert environment observation to LeRobot format observation.
    
    Args:
        observations: Dictionary of observation batches from environment.
        
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    return_observations = {}
    
    # Handle image observations
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}
            
        for imgkey, img in imgs.items():
            img = torch.from_numpy(img)
            # Add batch dimension if needed (for non-vectorized environments)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # Sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
            # Sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
            # Convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255
            return_observations[imgkey] = img
    
    # Handle your specific kinect RGB observation
    if "observation.image.kinect_rgb" in observations:
        img = torch.from_numpy(observations["observation.image.kinect_rgb"])
        # Add batch dimension if needed
        if img.ndim == 3:
            img = img.unsqueeze(0)
        # Sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
        # Sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
        # Convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        img /= 255
        return_observations["observation.image.kinect_rgb"] = img
    
    # Handle your specific C922 webcam observation
    if "observation.image.c922_webcam" in observations:
        img = torch.from_numpy(observations["observation.image.c922_webcam"])
        # Add batch dimension if needed
        if img.ndim == 3:
            img = img.unsqueeze(0)
        # Sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
        # Sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"
        # Convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        img /= 255
        return_observations["observation.image.c922_webcam"] = img
    
    # Handle environment state
    if "environment_state" in observations:
        env_state = torch.from_numpy(observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)
        return_observations["observation.environment_state"] = env_state
    
    # Handle agent position / robot state
    if "agent_pos" in observations:
        agent_pos = torch.from_numpy(observations["agent_pos"]).float()
        if agent_pos.dim() == 1:
            agent_pos = agent_pos.unsqueeze(0)
        return_observations["observation.state"] = agent_pos
    elif "observation.state" in observations:
        state = torch.from_numpy(observations["observation.state"]).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return_observations["observation.state"] = state
    
    # Handle task
    if "task" in observations:
        task = torch.from_numpy(observations["task"]).float()
        if task.dim() == 1:
            task = task.unsqueeze(0)
        return_observations["task"] = task
    
    return return_observations

def add_envs_task(env, observation):
    """Add task information to observation if available from environment."""
    # This is a placeholder - implement based on your environment setup
    # For now, just return the observation as-is
    return observation

# --------------- POLICY RUNNER ---------------- #

class UR10ePolicyRunner:
    def __init__(self, policy_path, fps=15):
        print("Initializing UR10e Policy Runner...")
        self.fps = fps
        self.dt = 1.0 / fps

        print("Starting camera feeds...")
        self.camera = CameraFeedManager(fps)
        self.camera.start_camera_thread()

        print("Connecting to robot...")
        self.robot = UR10eJoystickControl("192.168.1.102")
        if not self.robot.connect_robot():
            raise ConnectionError("Robot connection failed")
        self.robot.gripkit_connect()

        print("Loading policy model (this may take a while)...")
        self.device = get_safe_torch_device('cuda')
        try:
            self.policy = SmolVLAPolicy.from_pretrained(policy_path).to(self.device).eval()
            self.policy.reset()
            print("✓ Policy model loaded successfully")
        except KeyboardInterrupt:
            print("\n[WARNING] Policy loading interrupted. Cleaning up...")
            self.camera.stop()
            self.robot.disconnect_robot()
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load policy: {e}")
            self.camera.stop()
            self.robot.disconnect_robot()
            raise

        self.running = True
        self.return_observations = False  # Set to True if you want to collect observations
        self.all_observations = []  # Store observations if needed

    def save_debug_frame(self, obs, prefix="policy"):
        """Save a single frame for debugging"""
        import os
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save RGB image
        rgb = obs["observation.image.kinect_rgb"]
        cv2.imwrite(f"{debug_dir}/{prefix}_kinect_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # Save C922 webcam image
        if "observation.image.c922_webcam" in obs:
            c922 = obs["observation.image.c922_webcam"]
            cv2.imwrite(f"{debug_dir}/{prefix}_c922_webcam.png", cv2.cvtColor(c922, cv2.COLOR_RGB2BGR))
        
        # Save state data
        with open(f"{debug_dir}/{prefix}_state.txt", "w") as f:
            f.write(f"State: {obs['observation.state']}\n")
            f.write(f"Task: {obs['task']}\n")
            f.write(f"Kinect RGB shape: {rgb.shape} dtype: {rgb.dtype}\n")
            f.write(f"Kinect RGB Min/Max/Mean: {rgb.min()}/{rgb.max()}/{rgb.mean()}\n")
            if "observation.image.c922_webcam" in obs:
                c922 = obs["observation.image.c922_webcam"]
                f.write(f"C922 shape: {c922.shape} dtype: {c922.dtype}\n")
                f.write(f"C922 Min/Max/Mean: {c922.min()}/{c922.max()}/{c922.mean()}\n")

    def get_observation(self):
        """Get raw observation from sensors"""
        state = self.robot.get_robot_state()[:3]
        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        else:
            state = state.astype(np.float32)
            
        rgb, c922 = self.camera.get_latest_frames()
        # Ensure images are numpy arrays
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb, dtype=np.uint8)
        if not isinstance(c922, np.ndarray):
            c922 = np.array(c922, dtype=np.uint8)
        
        obs = {
            "observation.image.kinect_rgb": rgb,      # Keep as HWC uint8 numpy array
            "observation.image.c922_webcam": c922,    # Keep as HWC uint8 numpy array
            "observation.state": state,               # Keep as numpy array
            "task": np.array([0], dtype=np.float32)   # Keep as numpy array
        }
        
        # Save debug frame every 100 cycles
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
            if self.debug_counter % 100 == 0:
                self.save_debug_frame(obs)
        else:
            self.debug_counter = 0
            self.save_debug_frame(obs)  # Save first frame immediately
        
        return obs

    def run(self):
        print("Starting policy execution...")
        cycle_count = 0
        
        try:
            while self.running:
                start_time = time.time()

                # Get raw observation
                observation = self.get_observation()
                
                # Preprocess observation (convert to tensors, normalize images, etc.)
                observation = preprocess_observation(observation)
                
                # Store observation if needed
                if self.return_observations:
                    self.all_observations.append(deepcopy(observation))
                
                # Move tensors to device
                observation = {
                    key: observation[key].to(self.device, non_blocking=self.device.type == "cuda") 
                    for key in observation
                }
                
                # Add task information from environment if available
                observation["task"] = ["pick up the red block"]

                # Predict actionSSSSSSSS
                with torch.inference_mode():
                    action = self.policy.select_action(observation)
                
                # Convert to CPU / numpy
                action = action.to("cpu").numpy()
                
                # Debug print action
                print(f"Debug - Action type: {type(action)}, shape: {action.shape}")
                print("sending action", action)
                
                # Send action to robot
                action = action.squeeze()
                self.robot.send_tcp_action(action[:3])
                self.robot.control_gripper(action[3])

                if cycle_count % 10 == 0:
                    print(f"Cycle {cycle_count}: Action {action[:3]} Gripper {action[3]:.2f}")

                elapsed = time.time() - start_time
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

                cycle_count += 1

        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up...")
        self.camera.stop()
        self.robot.disconnect_robot()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")

# -------------------- ENTRY POINT -------------------- #

if __name__ == "__main__":
    POLICY_PATH = "smol_29jul/pretrained_model"
    runner = UR10ePolicyRunner(POLICY_PATH)
    runner.run()