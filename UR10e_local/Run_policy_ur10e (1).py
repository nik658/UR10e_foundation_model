#!/usr/bin/env python3

import numpy as np
import cv2
import time
import multiprocessing as mp
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

def _camera_worker(fps, out_q, stop_evt):
    dt = 1.0 / fps
    while not stop_evt.is_set():
        t0 = time.time()
        try:
            rgb, _ = freenect.sync_get_video()
            depth, _ = freenect.sync_get_depth()

            rgb = cv2.resize(rgb, (640, 480))  # Note: cv2.resize uses (width, height)
                
            depth  = cv2.resize(depth, (640, 480))
        except Exception as e:
            print(f"[CAMERA] error: {e}")
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = np.zeros((480, 640), dtype=np.uint16)

        try:
            out_q.put_nowait((rgb, depth))
        except mp.queues.Full:
            pass

        t_sleep = dt - (time.time() - t0)
        if t_sleep > 0:
            time.sleep(t_sleep)

class CameraFeedManager:
    def __init__(self, fps=15, queue_size=2):
        self.fps = fps
        self._queue = mp.Queue(maxsize=queue_size)
        self._stop_evt = mp.Event()
        self._proc = None
        self._latest_rgb = None
        self._latest_depth = None

    def start_camera_thread(self):
        print("[CAMERA] starting capture process...")
        self._proc = mp.Process(
            target=_camera_worker,
            args=(self.fps, self._queue, self._stop_evt),
            daemon=True,
        )
        self._proc.start()

        for _ in range(50):
            if not self._queue.empty():
                self._latest_rgb, self._latest_depth = self._queue.get()
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
            self._latest_rgb, self._latest_depth = self._queue.get_nowait()
        if self._latest_rgb is None:
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640), dtype=np.uint16)
        return self._latest_rgb.copy(), self._latest_depth.copy()

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
    
    # Handle your specific kinect depth observation
    if "observation.image.kinect_depth" in observations:
        img = torch.from_numpy(observations["observation.image.kinect_depth"])
        # Add batch dimension if needed
        if img.ndim == 3:
            img = img.unsqueeze(0)
        # Sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"
        # Sanity check that images are float32
        assert img.dtype == torch.float32, f"expect torch.float32, but instead {img.dtype=}"
        # Convert to channel first
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        return_observations["observation.image.kinect_depth"] = img
    
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

        self.camera = CameraFeedManager(fps)
        self.camera.start_camera_thread()

        self.robot = UR10eJoystickControl("192.168.1.102")
        if not self.robot.connect_robot():
            raise ConnectionError("Robot connection failed")
        self.robot.gripkit_connect()

        self.device = get_safe_torch_device('cuda')
        self.policy = SmolVLAPolicy.from_pretrained(policy_path).to(self.device).eval()
        self.policy.reset()

        self.running = True
        self.return_observations = False  # Set to True if you want to collect observations
        self.all_observations = []  # Store observations if needed

    def normalize_depth_for_lerobot(self, depth_image):
        """FIXED: Properly normalize depth image for LeRobot saving"""
        depth_meters = depth_image.astype(np.float32) / 1000.0
        
        depth_clipped = np.clip(depth_meters, 0, 1.5)
        depth_normalized = depth_clipped / 1.5 # Normalize to [0, 1] range
        
        if len(depth_normalized.shape) == 2:
            depth_3d = np.stack([depth_normalized, depth_normalized, depth_normalized], axis=2)
        else:
            depth_3d = depth_normalized
            
        depth_3d = np.clip(depth_3d, 0.0, 1.0)
        
        return depth_3d.astype(np.float32)

    def save_debug_frame(self, obs, prefix="policy"):
        """Save a single frame for debugging"""
        import os
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save RGB image
        rgb = obs["observation.image.kinect_rgb"]
        cv2.imwrite(f"{debug_dir}/{prefix}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        # Save depth image
        if "observation.image.kinect_depth" in obs:
            depth = obs["observation.image.kinect_depth"]
            # Convert normalized depth back to uint16 for saving
            depth_uint16 = (depth[:, :, 0] * 65535).astype(np.uint16)
            cv2.imwrite(f"{debug_dir}/{prefix}_depth.png", depth_uint16)
        
        # Save state data
        with open(f"{debug_dir}/{prefix}_state.txt", "w") as f:
            f.write(f"State: {obs['observation.state']}\n")
            f.write(f"Task: {obs['task']}\n")
            f.write(f"RGB shape: {rgb.shape} dtype: {rgb.dtype}\n")
            f.write(f"Min/Max/Mean: {rgb.min()}/{rgb.max()}/{rgb.mean()}\n")
            if "observation.image.kinect_depth" in obs:
                depth = obs["observation.image.kinect_depth"]
                f.write(f"Depth shape: {depth.shape} dtype: {depth.dtype}\n")
                f.write(f"Depth Min/Max/Mean: {depth.min()}/{depth.max()}/{depth.mean()}\n")

    def get_observation(self):
        """Get raw observation from sensors"""
        state = self.robot.get_robot_state()[:3]
        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        else:
            state = state.astype(np.float32)
            
        rgb, depth = self.camera.get_latest_frames()
        # Ensure RGB is a numpy array
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb, dtype=np.uint8)
        
        # Process depth using the normalize function
        depth_normalized = self.normalize_depth_for_lerobot(depth)
        
        obs = {
            "observation.image.kinect_rgb": rgb,  # Keep as HWC uint8 numpy array
            "observation.image.kinect_depth": depth_normalized,  # Normalized depth as HWC float32
            "observation.state": state,     # Keep as numpy array
            "task": np.array([0], dtype=np.float32)  # Keep as numpy array
        }
        
        # Debug print to check types

        
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
                observation["task"] = ["pick up the block"]

                # Predict action
                with torch.inference_mode():
                    action = self.policy.select_action(observation)
                
                # Convert to CPU / numpy
                action = action.to("cpu").numpy()
                
                
                # Debug print action
                print(f"Debug - Action type: {type(action)}, shape: {action.shape}")
                print("sending action",action)
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
    POLICY_PATH = "smol_2cam/pretrained_model"
    runner = UR10ePolicyRunner(POLICY_PATH)
    runner.run()
