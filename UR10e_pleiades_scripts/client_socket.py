#!/usr/bin/env python3

import numpy as np
import time
import socket
import pickle
import torch
import traceback
from pathlib import Path

# LeRobot imports
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device



class GPUPolicyClient:
    def __init__(self, policy_path="pretrained_model", host='localhost', port=8080):

        
        self.host = host
        self.port = port
        self.policy_path = policy_path
        self.policy = None
        self.device = get_safe_torch_device('cuda')
        self.socket = None
        self.connected = False
        
        # Connection settings
        self.connection_timeout = 10  # seconds
        self.receive_timeout = 30     # seconds
        self.retry_delay = 3          # seconds between connection attempts
        
        # Initialize GPU and policy
        self._initialize_gpu_and_policy()
        
    def _initialize_gpu_and_policy(self):
        """Initialize GPU device and load policy model"""
        try:
            print(f"[CLIENT] Loading policy from: {self.policy_path}", flush=True)
            self.policy = PI0Policy.from_pretrained(self.policy_path)
            self.policy.to(self.device)

            self.policy.reset()
            print("[CLIENT] Policy loaded and initialized successfully", flush=True)
            
        except Exception as e:
            print(f"[CLIENT] Error initializing GPU/Policy: {e}", flush=True)
            traceback.print_exc()
            raise
    
    def _connect_to_server(self):
        """Establish connection to the robot server"""
        try:
            print(f"[CLIENT] Creating socket...", flush=True)
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.connection_timeout)
            
            print(f"[CLIENT] Attempting to connect to {self.host}:{self.port}...", flush=True)
            self.socket.connect((self.host, self.port))
            
            # Set timeout for data operations
            self.socket.settimeout(self.receive_timeout)
            
            self.connected = True
            print(f"[CLIENT] ✓ Successfully connected to server!", flush=True)
            return True
            
        except socket.timeout:
            print(f"[CLIENT] Connection timeout after {self.connection_timeout}s", flush=True)
            self._cleanup_socket()
            return False
        except ConnectionRefusedError:
            print(f"[CLIENT] Connection refused. Is the server running?", flush=True)
            self._cleanup_socket()
            return False
        except Exception as e:
            print(f"[CLIENT] Connection error: {e}", flush=True)
            self._cleanup_socket()
            return False
    
    def _cleanup_socket(self):
        """Clean up socket connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
    
    def _receive_observation_data(self):
        """Receive observation data from server"""
        try:
            print("[CLIENT] Waiting for observation data from server...", flush=True)
            
            # First, receive the size of the data
            size_data = b""
            while len(size_data) < 4:
                chunk = self.socket.recv(4 - len(size_data))
                if not chunk:
                    raise ConnectionError("Server closed connection")
                size_data += chunk
            
            data_size = int.from_bytes(size_data, byteorder='big')
            print(f"[CLIENT] Expecting {data_size} bytes of observation data", flush=True)
            
            # Receive the actual data
            received_data = b""
            while len(received_data) < data_size:
                chunk = self.socket.recv(min(8192, data_size - len(received_data)))
                if not chunk:
                    raise ConnectionError("Server closed connection during data transfer")
                received_data += chunk
                
                # Progress indicator for large data
                if len(received_data) % 100000 == 0:
                    progress = (len(received_data) / data_size) * 100
                    print(f"[CLIENT] Received {progress:.1f}% of data...", flush=True)
            
            print(f"[CLIENT] Received complete observation data ({len(received_data)} bytes)", flush=True)
            
            # Deserialize the data
            observation_frame = pickle.loads(received_data)
            print(f"[CLIENT] Successfully deserialized observation data", flush=True)
            
            # Log observation info
            for key, value in observation_frame.items():
                if hasattr(value, 'shape'):
                    print(f"[CLIENT] - {key}: shape {value.shape}, dtype {value.dtype}", flush=True)
                else:
                    print(f"[CLIENT] - {key}: {type(value)}", flush=True)
            
            return observation_frame
            
        except socket.timeout:
            print(f"[CLIENT] Timeout while receiving observation data", flush=True)
            raise
        except Exception as e:
            print(f"[CLIENT]  Error receiving observation data: {e}", flush=True)
            raise
    
    def _compute_action(self, observation_frame):
        """Compute action using the policy model"""
        try:
            print("[CLIENT] Computing action with policy...", flush=True)
            
            observation_frame = {
            key: observation_frame[key].to(self.device, non_blocking=self.device.type == "cuda") for key in observation_frame
        }
            
        
            observation_frame["task"] = "pick up the block"

            with torch.inference_mode():
                action_values = self.policy.select_action(observation_frame)

            print("action:", action_values)

            # Convert the output tensor to a numpy array
            if isinstance(action_values, torch.Tensor):
                action = action_values.cpu().numpy().flatten()
                print(f"[CLIENT] Converted action (numpy array): {action}", flush=True)
                print(f"[CLIENT] Action shape: {action.shape}, dtype: {action.dtype}", flush=True)
                return action
            else:
                print(f"[CLIENT] Action is not a tensor: {action_values}", flush=True)
                return action_values

        except Exception as e:
            print(f"[CLIENT] Error computing action: {e}", flush=True)
            traceback.print_exc()
            raise


    
    def _send_action(self, action):
        """Send computed action back to server"""
        try:
            print(f"[CLIENT] Sending action to server: {action}", flush=True)
            
            # Serialize action
            action_data = pickle.dumps(action)
            
            # Send size first
            size_bytes = len(action_data).to_bytes(4, byteorder='big')
            self.socket.sendall(size_bytes)
            
            # Send action data
            self.socket.sendall(action_data)
            
            print(f"[CLIENT] ✓ Action sent successfully ({len(action_data)} bytes)", flush=True)
            
        except Exception as e:
            print(f"[CLIENT] Error sending action: {e}", flush=True)
            raise
    
    def run_single_cycle(self):
        """Run a single prediction cycle"""
        try:
            # Receive observation data
            observation_frame = self._receive_observation_data()
            
            # Compute action
            action = self._compute_action(observation_frame)
            
            if action is None:
                return True
            # Send action back
            self._send_action(action)
            
            return True
            
        except Exception as e:
            print(f"[CLIENT] Error in prediction cycle: {e}", flush=True)
            return False
    
    def run(self):
        """Main execution loop"""
        print("[CLIENT] Starting main execution loop...", flush=True)
        cycle_count = 0
        
        while True:
            try:
                # Ensure connection
                if not self.connected:
                    if not self._connect_to_server():
                        print(f"[CLIENT] Retrying connection in {self.retry_delay} seconds...", flush=True)
                        time.sleep(self.retry_delay)
                        continue
                
                # Run prediction cycle
                success = self.run_single_cycle()
                
                if success:
                    cycle_count += 1
                    if cycle_count % 10 == 0:
                        print(f"[CLIENT] ✓ Completed {cycle_count} prediction cycles", flush=True)
                else:
                    print("[CLIENT] Prediction cycle failed, will retry...", flush=True)
                    self._cleanup_socket()
                    time.sleep(self.retry_delay)
                
            except KeyboardInterrupt:
                print("\n[CLIENT] KeyboardInterrupt received, exiting...", flush=True)
                break
            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                print(f"[CLIENT]  Connection lost: {e}", flush=True)
                print(f"[CLIENT] Reconnecting in {self.retry_delay} seconds...", flush=True)
                self._cleanup_socket()
                time.sleep(self.retry_delay)
            except Exception as e:
                print(f"[CLIENT]  Unexpected error: {e}", flush=True)
                traceback.print_exc()
                print(f"[CLIENT] Retrying in {self.retry_delay} seconds...", flush=True)
                self._cleanup_socket()
                time.sleep(self.retry_delay)
        
        # Cleanup
        self._cleanup_socket()
        print("[CLIENT] ✓ Client shutdown complete", flush=True)

def main():
    print("PI0 POLICY - Starting on Pleiades Cluster", flush=True)

    POLICY_PATH = "outputs/train/pi0_depth_40ep/checkpoints/last/pretrained_model/" 
    HOST = 'pleiades.ieeta.pt' 
    PORT = 8080    
    
    print(f"Configuration:", flush=True)
    print(f"- Policy Path: {POLICY_PATH}", flush=True)
    print(f"- Server: {HOST}:{PORT}", flush=True)
    print(f"- Device: GPU ", flush=True)
    
    try:
        client = GPUPolicyClient(
            policy_path=POLICY_PATH,
            host=HOST,
            port=PORT
        )
        
        print("[MAIN] Starting client execution...", flush=True)
        client.run()
        
    except Exception as e:
        print(f"[MAIN] Fatal error: {e}", flush=True)
        traceback.print_exc()
        return 1
    
    print("[MAIN] ✓ Main execution completed", flush=True)
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)