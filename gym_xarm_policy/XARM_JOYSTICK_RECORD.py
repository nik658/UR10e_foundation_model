#!/usr/bin/env python3

import gymnasium as gym
import gym_xarm
import pygame
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import os
import time
import copy
from pathlib import Path
import shutil
import threading

class XArmDataCollector:
    def __init__(self, data_dir="./xarm_lift_custom_data", fps=15, repo_id=None, push_to_hub=False):
        print("Initializing XArmDataCollector...")
        self.fps = fps
        self.dt = 1.0 / fps
        
        self.repo_id = repo_id
        self.push_to_hub = push_to_hub
        if self.repo_id:
            print(f"LeRobot Hugging Face repo ID: {self.repo_id}, Push to Hub: {self.push_to_hub}")
        else:
            print("No LeRobot repo ID provided. LeRobot features disabled.")
        self.dataset = None
        
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            print("[ERROR] No controller found! Please connect a controller.")
            raise RuntimeError("No controller found!")
        
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        print(f"Controller connected: {self.controller.get_name()}")
        
        print("Initializing Gym environment: gym_xarm/XarmLift-v0...")
        self.env = gym.make("gym_xarm/XarmLift-v0", render_mode="rgb_array", obs_type="pixels_agent_pos")
        print("obs", self.env.observation_space)
        print("Gym environment initialized.")
        
        temp_obs, _ = self.env.reset()
        sample_state = self.get_state_from_observation(temp_obs)
        self.state_dim = sample_state.shape[0]  # Should be 4
        print(f"State dimension: {self.state_dim}")
        
        self.setup_data_directories(data_dir)
        
        self.episode_index = self.get_next_episode_index()
        
        self.recording = False
        self.episode_data = []
        self.episode_images = []
        self.frame_index = 0
        
        self.episode_start_time = 0
        self.target_frame_times = []
        
        self.prev_record_button = False
        self.prev_reset_button = False
        
        print("XArm Lift Data Collector initialized!")
        print("--- Controls ---")
        print("- Left stick: X/Y movement")
        print("- Right stick: Z movement") 
        print("- A/B buttons: Gripper open/close")
        print("- X button: Manual reset")
        print("- Y button: Start/Stop recording")
        print("- Back button: Exit")
        print("----------------\n")
    
    def setup_data_directories(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.video_dir = self.base_dir / "videos"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        print(f"Data will be saved to: {self.base_dir.absolute()}")

    def get_next_episode_index(self):
        if not self.data_dir.exists():
            return 0
            
        existing_files = list(self.data_dir.glob("episode_*.parquet"))
        if not existing_files:
            return 0
            
        episode_numbers = []
        for file in existing_files:
            try:
                episode_num = int(file.stem.split('_')[1])
                episode_numbers.append(episode_num)
            except (IndexError, ValueError):
                print(f"[WARNING] Could not parse episode number from file: {file.name}")
                continue
                
        next_index = max(episode_numbers) + 1 if episode_numbers else 0
        print(f"Resuming local recording from episode {next_index}.")
        return next_index
    
    def create_lerobot_dataset(self):
        if self.dataset is not None:
            return
        if self.repo_id is None:
            return
            
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            lerobot_dir = self.base_dir / "lerobot_data"
            
            if lerobot_dir.exists():
                print(f"Existing LeRobot data directory found. Removing for a fresh dataset: {lerobot_dir}")
                shutil.rmtree(lerobot_dir)
            
            state_dim = self.state_dim  # Should be 4
            
            features = {
                "observation.image": {
                    "dtype": "video",
                    "shape": (84, 84, 3),
                    "names": ["height", "width", "channel"]
                },
                "observation.state": {
                    "dtype": "float32", 
                    "shape": (4,),
                    "names": None
                },
                "action": {
                    "dtype": "float32",
                    "shape": (4,),
                    "names": None
                },
                "timestamp": {
                    "dtype": "float32", 
                    "shape": (1,),
                    "names": None
                },
                "next.reward": {
                    "dtype": "float32", 
                    "shape": (1,),
                    "names": None
                },
                "next.done": {
                    "dtype": "bool", 
                    "shape": (1,),
                    "names": None
                },
            }
            
            print(f"Creating LeRobotDataset for repo '{self.repo_id}' at root '{lerobot_dir}'...")
            self.dataset = LeRobotDataset.create(
                self.repo_id, 
                self.fps, 
                root=str(lerobot_dir),
                robot=None, 
                use_videos=True, 
                features=features
            )
            print(f"LeRobot dataset created successfully for repo: {self.repo_id}")
            
        except ImportError:
            print("[ERROR] LeRobot library (lerobot.common.datasets.lerobot_dataset) not found. Please ensure it's installed.")
            self.dataset = None
        except Exception as e:
            print(f"[ERROR] Critical error during LeRobot dataset creation: {e}")
            import traceback
            traceback.print_exc()
            self.dataset = None
    
    def get_controller_action(self, action_scale=0.4):
        pygame.event.pump()
        action = np.zeros(4, dtype=np.float32)
        
        left_x = -self.controller.get_axis(1)
        left_y = -self.controller.get_axis(0)
        right_y = -self.controller.get_axis(4)
        
        deadzone = 0.1
        action[0] = left_x * action_scale if abs(left_x) > deadzone else 0.0
        action[1] = left_y * action_scale if abs(left_y) > deadzone else 0.0
        action[2] = right_y * action_scale if abs(right_y) > deadzone else 0.0
        
        if self.controller.get_button(0):
            action[3] = -action_scale
        elif self.controller.get_button(1):
            action[3] = action_scale
        
        return action
    
    def check_controller_buttons(self):
        pygame.event.pump()
        
        record_button = self.controller.get_button(3)
        if record_button and not self.prev_record_button:
            print("[ACTION] Record button (Y) pressed.")
            self.toggle_recording()
        self.prev_record_button = record_button
        
        reset_button = self.controller.get_button(2)
        if reset_button and not self.prev_reset_button:
            print("[ACTION] Reset button (X) pressed.")
            self.manual_reset()
        self.prev_reset_button = reset_button
        
        if self.controller.get_button(6):
            print("[ACTION] Exit button (Back) pressed. Signaling to terminate.")
            return False
        
        return True
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        if self.repo_id and self.dataset is None:
            self.create_lerobot_dataset()
            if self.dataset is None:
                 print("[WARNING] LeRobot dataset creation failed or was skipped. Will record locally only for LeRobot parts.")
            
        self.recording = True
        self.episode_data = []
        self.episode_images = []
        self.frame_index = 0
        self.episode_start_time = time.time()
        
        self.target_frame_times = [i * self.dt for i in range(10000)]
        
        lerobot_episode_idx_str = "N/A"
        if self.dataset is not None:
            try:
                self.dataset.episode_buffer = self.dataset.create_episode_buffer(episode_index=None)
                lerobot_episode_idx_str = str(self.dataset.meta.total_episodes)
            except Exception as e:
                print(f"[ERROR] Could not initialize LeRobot episode buffer: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\033[32m[ACTION] START RECORDING Local Episode {self.episode_index} (LeRobot Episode: {lerobot_episode_idx_str})\033[0m")
    
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        lerobot_ep_idx_str = "N/A"
        if self.dataset and hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, 'total_episodes'):
            lerobot_ep_idx_str = str(self.dataset.meta.total_episodes)
        print(f"\033[31m[ACTION] STOP RECORDING Local Episode {self.episode_index} (LeRobot Episode: {lerobot_ep_idx_str})\033[0m")
        
        if len(self.episode_data) > 0:
            self.save_episode_data()
            
            if self.dataset is not None and hasattr(self.dataset, 'episode_buffer') and len(self.dataset.episode_buffer) > 0:
                try:
                    self.dataset.save_episode()
                    saved_lerobot_idx = self.dataset.meta.total_episodes - 1 
                    print(f"LeRobot episode {saved_lerobot_idx} saved successfully.")
                except Exception as e:
                    print(f"[ERROR] Could not save LeRobot episode: {e}")
                    import traceback
                    traceback.print_exc()
            elif self.dataset is not None:
                print("LeRobot episode buffer is empty or not initialized. Skipping LeRobot episode save.")
            
            self.episode_index += 1
        else:
            print("No data frames recorded in this episode. Nothing to save.")
    
    def manual_reset(self):
        print("\n[ACTION] Manual reset triggered via controller.")
        observation, info = self.env.reset()
        print("Environment reset complete.\n")
        return observation, info
    
    

    def save_episode_data(self):
        print(f"Saving local data for episode {self.episode_index} ({len(self.episode_data)} frames)...")
        df = pd.DataFrame(self.episode_data)
        
        episode_filename = f"episode_{self.episode_index:06d}.parquet"
        parquet_path = self.data_dir / episode_filename
        
        try:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, str(parquet_path))
            print(f"Saved local Parquet: {parquet_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to save local Parquet file {parquet_path.name}: {e}")
        
        if len(self.episode_images) > 0:
            video_filename = f"episode_{self.episode_index:06d}.mp4"
            video_path = self.video_dir / video_filename
            self.save_video(self.episode_images, str(video_path))
        else:
            print("No images recorded for local video.")
    
    def save_video(self, images, video_path):
        if not images:
            print("[WARNING] No images provided to save_video. Video not saved.")
            return
        
        height, width = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        if not out.isOpened():
            print(f"[ERROR] Could not open video writer for path: {video_path}. Check codec, permissions, or path.")
            return
        
        print(f"Writing video to {Path(video_path).name} ({len(images)} frames at {self.fps} FPS)...")
        for img in images:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(bgr_img)
        out.release()
    
    def get_state_from_observation(self, observation):
        """
        Extract the agent_pos directly from the observation dictionary.
        The observation now contains 'agent_pos' which is our 4-dimensional state.
        """
        if isinstance(observation, dict) and 'agent_pos' in observation:
            return observation['agent_pos'].astype(np.float32)
        else:
            print(f"[ERROR] Expected observation to be a dict with 'agent_pos' key, got: {type(observation)}")
            if isinstance(observation, dict):
                print(f"Available keys: {list(observation.keys())}")
            # Fallback to zeros if something goes wrong
            return np.zeros(4, dtype=np.float32)
    
    def record_frame(self, observation, action, reward, terminated, truncated, info):
        if self.frame_index < len(self.target_frame_times):
            current_time = self.target_frame_times[self.frame_index]
        else:
            current_time = self.frame_index * self.dt
        
        state = self.get_state_from_observation(observation)
        image = self.env.render()
        success = info.get('is_success', False) or info.get('success', False)
        
        frame_data = {
            'observation.state': state.tolist(),
            'action': action.tolist(),
            'episode_index': self.episode_index,
            'frame_index': self.frame_index,
            'timestamp': current_time,
            'next.reward': reward,
            'next.done': terminated or truncated,
            'next.success': success,
            'index': self.episode_index * 1000 + self.frame_index,
            'task': ['xarm_lift']
        }
        
        self.episode_data.append(frame_data)
        self.episode_images.append(image.copy())
        
        if self.dataset is not None and hasattr(self.dataset, 'episode_buffer'):
            try:
                resized_image = cv2.resize(image, (84, 84))
                lerobot_frame = {
                    'observation.image': resized_image,
                    'observation.state': state.astype(np.float32),
                    'action': action.astype(np.float32),
                    'timestamp': np.array([current_time], dtype=np.float32),
                    'next.reward': np.array([float(reward)], dtype=np.float32),
                    'next.done': np.array([bool(terminated or truncated)], dtype=bool),
                    'task': 'xarm_lift'
                }
                self.dataset.add_frame(lerobot_frame)
            except Exception as e:
                print(f"[ERROR] Failed to add frame to LeRobot dataset buffer: {e}")
                import traceback
                traceback.print_exc()
        
        self.frame_index += 1
        
        if self.frame_index % (self.fps * 2) == 0:
            lerobot_ep_str = "N/A"
            if self.dataset and hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, 'total_episodes'):
                lerobot_ep_str = str(self.dataset.meta.total_episodes)
            print(f"\033[32m[INFO] RECORDING LocalEp {self.episode_index} (LeRobotEp: {lerobot_ep_str}), "
                  f"Frame {self.frame_index}, Time {current_time:.4f}s, Reward: {reward:.3f}, Success: {success}, State shape: {state.shape}\033[0m")
    
    def run(self):
        print("Starting main data collection loop...")
        observation, info = self.env.reset()
        print("Initial environment reset complete.")
        
        running = True
        try:
            while running:
                loop_start_time = time.time()
                
                if not self.check_controller_buttons():
                    running = False
                    continue
                
                action = self.get_controller_action()
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                if self.recording:
                    self.record_frame(observation, action, reward, terminated, truncated, info)
                
                display_image = self.env.render()
                cv2.imshow('XArm Lift Environment', cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' key pressed in OpenCV window. Exiting.")
                    running = False
                
                elapsed_time = time.time() - loop_start_time
                sleep_duration = self.dt - elapsed_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                elif sleep_duration < -0.01:
                    print(f"[WARNING] Loop running {-sleep_duration:.4f}s behind target timing")
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt (Ctrl+C) detected. Initiating graceful shutdown.")
        
        finally:
            print("Exiting main data collection loop.")
            if self.recording:
                print("Recording was active during exit. Stopping and saving final episode data...")
                self.stop_recording()
            self.cleanup()
    
    def cleanup(self):
        print("\nStarting cleanup sequence...")
        
        if self.dataset is not None and self.push_to_hub:
            if self.repo_id:
                print(f"Attempting to push LeRobot dataset to Hugging Face Hub: {self.repo_id}...")
                try:
                    self.dataset.push_to_hub()
                    print(f"LeRobot dataset successfully pushed to Hub: {self.repo_id}")
                except Exception as e:
                    print(f"[ERROR] Failed to push LeRobot dataset to Hub: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("`push_to_hub` is True, but no `repo_id` was configured. Skipping Hub push.")
        elif self.push_to_hub and self.dataset is None:
             print("`push_to_hub` is True, but LeRobot dataset was not initialized. Skipping Hub push.")

        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
        
        cv2.destroyAllWindows()
        pygame.quit()
        print("\nData collection and cleanup completed!")

if __name__ == "__main__":
    DATA_DIR = "./xarm_lift_custom_data"
    FPS = 15
    REPO_ID = "nik658/xarm_lift_custom_dataset50"
    PUSH_TO_HUB = True
    
    print("===================================")
    print("=== XArm Lift Data Collection ===")
    print("===================================")
    print(f"Data directory: {DATA_DIR}")
    print(f"FPS: {FPS}")
    print(f"Repo ID: {REPO_ID}")
    print(f"Push to Hub: {PUSH_TO_HUB}")
    print("-----------------------------------\n")
    
    collector = XArmDataCollector(
        data_dir=DATA_DIR, 
        fps=FPS, 
        repo_id=REPO_ID, 
        push_to_hub=PUSH_TO_HUB
    )
    collector.run()