#!/usr/bin/env python3

import socket
import threading
import time
import pickle
import numpy as np
import traceback

class PleiadesProxyServer:
    def __init__(self, listen_port=8080, robot_host='localhost', robot_port=5000):
        """
        Proxy server that runs on Pleiades
        - Listens on port 8080 for GPU client connections
        - Forwards to robot server through SSH tunnel on port 5000
        """
        self.listen_port = listen_port
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.running = False
        
        print(f"[PROXY] Initializing proxy server")
        print(f"[PROXY] Listen port: {listen_port}")
        print(f"[PROXY] Robot server: {robot_host}:{robot_port}")
        
    def _validate_data_types(self, data, data_type="unknown"):
        """Validate and log data types being transferred"""
        try:
            if isinstance(data, dict):
                print(f"[PROXY] {data_type} - Dictionary with {len(data)} keys:")
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        print(f"[PROXY]   {key}: numpy array, shape={value.shape}, dtype={value.dtype}")
                    elif hasattr(value, 'shape'):
                        print(f"[PROXY]   {key}: tensor-like, shape={value.shape}")
                    else:
                        print(f"[PROXY]   {key}: {type(value)}")
            elif isinstance(data, np.ndarray):
                print(f"[PROXY] {data_type} - numpy array: shape={data.shape}, dtype={data.dtype}")
            elif isinstance(data, (list, tuple)):
                print(f"[PROXY] {data_type} - {type(data).__name__} with {len(data)} elements")
                if len(data) > 0:
                    print(f"[PROXY]   First element type: {type(data[0])}")
            else:
                print(f"[PROXY] {data_type} - {type(data)}: {data}")
                
        except Exception as e:
            print(f"[PROXY] Error validating {data_type}: {e}")
    
    def _receive_data(self, socket_conn, data_type="data"):
        """Receive and deserialize data with error handling"""
        try:
            print(f"[PROXY] Receiving {data_type}...")
            
            # Receive size header (4 bytes)
            size_data = b""
            while len(size_data) < 4:
                chunk = socket_conn.recv(4 - len(size_data))
                if not chunk:
                    raise ConnectionError(f"Connection closed while receiving {data_type} size")
                size_data += chunk
            
            data_size = int.from_bytes(size_data, byteorder='big')
            print(f"[PROXY] Expecting {data_size} bytes of {data_type}")
            
            # Receive actual data
            received_data = b""
            while len(received_data) < data_size:
                remaining = data_size - len(received_data)
                chunk_size = min(8192, remaining)
                chunk = socket_conn.recv(chunk_size)
                
                if not chunk:
                    raise ConnectionError(f"Connection closed while receiving {data_type}")
                received_data += chunk
                
                # Progress for large transfers
                if data_size > 100000 and len(received_data) % 50000 == 0:
                    progress = (len(received_data) / data_size) * 100
                    print(f"[PROXY] {data_type} progress: {progress:.1f}%")
            
            print(f"[PROXY] ✓ Received {len(received_data)} bytes of {data_type}")
            
            # Deserialize
            data = pickle.loads(received_data)
            self._validate_data_types(data, data_type)
            
            return data
            
        except Exception as e:
            print(f"[PROXY] Error receiving {data_type}: {e}")
            raise
    
    def _send_data(self, socket_conn, data, data_type="data"):
        """Serialize and send data with error handling"""
        try:
            print(f"[PROXY] Sending {data_type}...")
            self._validate_data_types(data, data_type)
            
            # Serialize data
            serialized_data = pickle.dumps(data)
            data_size = len(serialized_data)
            
            print(f"[PROXY] Serialized {data_type}: {data_size} bytes")
            
            # Send size header
            size_bytes = data_size.to_bytes(4, byteorder='big')
            socket_conn.sendall(size_bytes)
            
            # Send data
            bytes_sent = 0
            while bytes_sent < data_size:
                chunk_size = min(8192, data_size - bytes_sent)
                chunk = serialized_data[bytes_sent:bytes_sent + chunk_size]
                socket_conn.sendall(chunk)
                bytes_sent += chunk_size
                
                # Progress for large transfers
                if data_size > 100000 and bytes_sent % 50000 == 0:
                    progress = (bytes_sent / data_size) * 100
                    print(f"[PROXY] {data_type} send progress: {progress:.1f}%")
            
            print(f"[PROXY] ✓ Sent {data_type} successfully")
            
        except Exception as e:
            print(f"[PROXY] Error sending {data_type}: {e}")
            raise
    
    def _handle_client_connection(self, client_socket, client_addr):
        """Handle a single GPU client connection"""
        robot_socket = None
        
        try:
            print(f"[PROXY] GPU client connected from {client_addr}")
            
            # Connect to robot server through SSH tunnel
            print(f"[PROXY] Connecting to robot server {self.robot_host}:{self.robot_port}")
            robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            robot_socket.settimeout(10.0)  # Connection timeout
            robot_socket.connect((self.robot_host, self.robot_port))
            robot_socket.settimeout(30.0)  # Data timeout
            
            print(f"[PROXY] ✓ Connected to robot server")
            
            cycle_count = 0
            
            while self.running:
                try:
                    cycle_start = time.time()
                    
                    # Step 1: Receive observation from robot server
                    observation = self._receive_data(robot_socket, "observation")
                    
                    # Step 2: Forward observation to GPU client
                    self._send_data(client_socket, observation, "observation")
                    
                    # Step 3: Receive action from GPU client
                    action = self._receive_data(client_socket, "action")
                    
                    # Step 4: Forward action to robot server
                    self._send_data(robot_socket, action, "action")
                    
                    cycle_count += 1
                    cycle_time = time.time() - cycle_start
                    
                    if cycle_count % 10 == 0:
                        print(f"[PROXY] ✓ Completed {cycle_count} cycles (last: {cycle_time:.3f}s)")
                    
                except socket.timeout:
                    print(f"[PROXY] Socket timeout in cycle {cycle_count}")
                    break
                except ConnectionError as e:
                    print(f"[PROXY] Connection error in cycle {cycle_count}: {e}")
                    break
                except Exception as e:
                    print(f"[PROXY] Error in cycle {cycle_count}: {e}")
                    traceback.print_exc()
                    break
            
        except ConnectionRefusedError:
            print(f"[PROXY] ❌ Could not connect to robot server at {self.robot_host}:{self.robot_port}")
            print(f"[PROXY] Make sure SSH tunnel is active: ssh -R {self.robot_port}:localhost:{self.robot_port}")
        except Exception as e:
            print(f"[PROXY] Error handling client {client_addr}: {e}")
            traceback.print_exc()
        finally:
            # Cleanup connections
            if robot_socket:
                try:
                    robot_socket.close()
                except:
                    pass
            try:
                client_socket.close()
            except:
                pass
            print(f"[PROXY] Client {client_addr} disconnected")
    
    def start(self):
        """Start the proxy server"""
        print(f"[PROXY] Starting proxy server on port {self.listen_port}")
        
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.listen_port))
            server_socket.listen(5)
            
            print(f"[PROXY] ✓ Listening on port {self.listen_port}")
            print(f"[PROXY] Waiting for GPU client connections...")
            print(f"[PROXY] Will forward to robot server at {self.robot_host}:{self.robot_port}")
            
            self.running = True
            
            while self.running:
                try:
                    # Accept GPU client connection
                    client_socket, client_addr = server_socket.accept()
                    print(f"[PROXY] New connection from {client_addr}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client_connection,
                        args=(client_socket, client_addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except KeyboardInterrupt:
                    print("\n[PROXY] KeyboardInterrupt received")
                    break
                except Exception as e:
                    print(f"[PROXY] Error accepting connection: {e}")
                    time.sleep(1)
            
        except Exception as e:
            print(f"[PROXY] Server error: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            try:
                server_socket.close()
            except:
                pass
            print(f"[PROXY] ✓ Proxy server stopped")

def main():
    print("=" * 60)
    print("PLEIADES PROXY SERVER")
    print("=" * 60)
    print()
    
    # Configuration
    LISTEN_PORT = 8080      # Port for GPU client to connect to
    ROBOT_HOST = 'localhost' # Robot server through SSH tunnel
    ROBOT_PORT = 5000       # Robot server port
    
    print(f"Configuration:")
    print(f"  Listen Port:    {LISTEN_PORT} (for GPU client)")
    print(f"  Robot Server:   {ROBOT_HOST}:{ROBOT_PORT} (through SSH tunnel)")
    
    try:
        proxy = PleiadesProxyServer(
            listen_port=LISTEN_PORT,
            robot_host=ROBOT_HOST,
            robot_port=ROBOT_PORT
        )
        
        proxy.start()
        
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
    except Exception as e:
        print(f"[MAIN] Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    print("[MAIN] ✓ Shutdown complete")
    return 0

if __name__ == "__main__":
    exit(main())

