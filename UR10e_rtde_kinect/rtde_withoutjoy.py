#!/usr/bin/env python3
from multiprocessing import Process
import rtde_control
import rtde_receive
import time
import numpy as np
import freenect
import cv2
import socket
from xmlrpc import client
import time


class UR10eJoystickControl:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.rtde_c = None
        self.rtde_r = None
        self.controller = None
        self.is_connected = False
        self.last_target_pose = None # To maintain state between calls for servoL
        self.gripper_server= None
        self.gripper_gid= None
        
    def connect_robot(self):
        """Connects to the robot's RTDE interfaces."""
        try:
            print(f"Connecting to UR10e robot at {self.robot_ip}...")
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            self.is_connected = True
            print("Successfully connected to UR10e RTDE interfaces.")
            # Get initial TCP pose as the starting point for servoing
            self.last_target_pose = self.rtde_r.getActualTCPPose()
            return True
        except Exception as e:
            print(f"Failed to connect to UR10e robot: {e}")
            self.is_connected = False
            return False

    def disconnect_robot(self):
        """Disconnects from the robot's RTDE interfaces."""
        if self.rtde_c:
            self.rtde_c.stopScript() # Ensure any running script is stopped
            self.rtde_c.disconnect()
            self.rtde_c = None
        if self.rtde_r:
            self.rtde_r.disconnect()
            self.rtde_r = None
        self.is_connected = False
        print("Disconnected from UR10e robot.")

    def get_robot_state(self):
        """Get the current TCP pose of the robot"""
        if not self.is_connected or self.rtde_r is None:
            print("Robot not connected - cannot get state")
            return [0.0] * 6 # Return zeros if not connected
        try:
            tcp_pose = self.rtde_r.getActualTCPPose()
            return tcp_pose
        except Exception as e:
            print(f"Failed to get robot state: {e}")
            return [0.0] * 6 # Return zeros on error

    def gripkit_connect(self):
        GRIPPER_IP = "192.168.1.102"  # Default Weiss CR200 IP
        PORT = 44221                    # Default HTTP port
        CONN_STR = f"http://{GRIPPER_IP}:{PORT}/RPC2"
        # Connect to gripper
        server = client.ServerProxy(CONN_STR)

        # Get first available gripper ID
        try:
            gid = server.GetGrippers()[0]
            print(f"Connected to gripper ID: {gid}")
            self.gripper_server = server
            self.gripper_gid = gid
        except Exception as e:
            print(f"Connection failed: {e}")
            exit(1)

        try:
            # Set release limit (80mm) and no-part detection (1mm)
            server.SetReleaseLimit(gid, 1, 80.0)
            server.SetNoPartLimit(gid, 1, 25.0)
            print("Gripper initialized")
        except:
            print("Initialization commands not supported (may be in simulation)")

    def control_gripper(self, command):
        if command > 0.5:  # Open
            self.gripper_server.Release(self.gripper_gid, 1)
            print("Gripper opened")
        elif command < -0.5:  # Close
            self.gripper_server.Grip(self.gripper_gid, 1)
            print("Gripper closed")
            
    def send_tcp_action(self, action_xyz, speed=0.05, acceleration=0.05, blend=0.0):
        """
        Sends a TCP position command to the robot using servoL for smooth control.
        action_xyz: [dx, dy, dz] values to add to current TCP pose.
        """
        if not self.is_connected or self.rtde_c is None:
            print("Robot not connected. Cannot send TCP action.")
            return

        if self.last_target_pose is None:
            # Fallback if last_target_pose somehow isn't set, try to get current pose
            self.last_target_pose = self.get_robot_state()
            if self.last_target_pose is None or np.all(np.array(self.last_target_pose) == 0):
                print("Could not get valid initial robot pose. Cannot send TCP action.")
                return
        
        print("sending TCP action")
        new_pose = self.last_target_pose.copy()
        for i in range(3): # Apply delta to X, Y, Z
            new_pose[i] += action_xyz[i]

        # Ensure orientation (last 3 elements) remains unchanged from initial
        if len(new_pose) == 6 and len(self.last_target_pose) == 6:
            new_pose[3:] = self.last_target_pose[3:] # Keep Rx, Ry, Rz the same

        try:
            # Using servoL with fixed speed/accel/blend values from original script
            self.rtde_c.servoL(new_pose, 0.05, 0.05, 0.02, 0.1, 300)
            self.last_target_pose = new_pose # Update last target pose for next iteration
        except Exception as e:
            print(f"Error sending TCP action via servoL: {e}")
            # If servoL fails, reset last_target_pose to actual robot pose
            self.last_target_pose = self.get_robot_state()

def main():
    ROBOT_HOST = "192.168.1.102"
    print("trying to connect")
    robot = UR10eJoystickControl(ROBOT_HOST)
    print("CONNECT........")
    
    # ✅ Connect to robot first
    if robot.connect_robot():
        print("Robot connected successfully!")
        
        # ✅ Now get robot state
        state = robot.get_robot_state()
        print(f"Robot TCP pose: {state}")

if __name__ =="__main__":
    main()
