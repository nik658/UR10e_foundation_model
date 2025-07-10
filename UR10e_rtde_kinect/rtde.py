#!/usr/bin/env python3
from multiprocessing import Process
import rtde_control
import rtde_receive
import time
import pygame
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

    def initialise_controller(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No controller found!")
            return None
        controller = pygame.joystick.Joystick(0)
        controller.init()
        print(f"Controller connected: {controller.get_name()}")
        self.controller = controller # Store the controller instance
        return controller

    def get_controller_action(self, action_scale=0.001):
        """Return [dx, dy, dz, gripper] and if there was input"""
        if self.controller is None:
            return np.zeros(4, dtype=np.float32), False

        pygame.event.pump()
        action = np.zeros(4, dtype=np.float32)  # [dx, dy, dz, gripper]
        left_x = self.controller.get_axis(0)
        left_y = self.controller.get_axis(1)
        right_y = -self.controller.get_axis(4)
        deadzone = 0.2
        has_input = False

        # X (left_y), Y (left_x), Z (right_y)
        if abs(left_x) > deadzone:
            action[1] = left_x * action_scale
            has_input = True
        if abs(left_y) > deadzone:
            action[0] = left_y * action_scale
            has_input = True
        if abs(right_y) > deadzone:
            action[2] = right_y * action_scale
            has_input = True

        # Gripper: Only A (open) and B (close)
        if self.controller.get_button(0):  # A button
            action[3] = 1.0   # Open
            has_input = True
        elif self.controller.get_button(1):  # B button
            action[3] = -1.0  # Close
            has_input = True
        else:
            action[3] = 0.0

        return action, has_input

    def get_robot_state(self):
        """Get the current TCP pose of the robot"""
        if not self.is_connected or self.rtde_r is None:
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
        #gripper = Gripper('cr200-85', host='10.1.0.2', port=nmap)
        # Connect to gripper
        server = client.ServerProxy(CONN_STR)

        # Get first available gripper ID10.1.
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
        # Or, if you want to allow orientation changes, calculate them from controller input
        # For now, keeping orientation fixed as per your joystick control logic
        # (which only affected XYZ)
        if len(new_pose) == 6 and len(self.last_target_pose) == 6:
            new_pose[3:] = self.last_target_pose[3:] # Keep Rx, Ry, Rz the same

        try:
            # Using servoL with fixed speed/accel/blend values from original script
            # servoL requires control_speed and control_acceleration
            self.rtde_c.servoL(new_pose, 0.05, 0.05, 0.02, 0.1, 300)
            self.last_target_pose = new_pose # Update last target pose for next iteration
        except Exception as e:
            print(f"Error sending TCP action via servoL: {e}")
            # If servoL fails, reset last_target_pose to actual robot pose
            # to avoid large jumps if control regains
            self.last_target_pose = self.get_robot_state()


    # The run_joystick_control method is kept here for reference or standalone testing,
    # but the DataCollector will manage the main loop and robot interaction.
    def run_joystick_control(self):
        controller = self.initialise_controller()
        if controller is None:
            print("Cannot proceed without controller")
            return

        if not self.connect_robot():
            print("Failed to connect to robot. Exiting joystick control.")
            return

        self.gripkit_connect()

        print("\n[SAFETY] Ensure the robot workspace is clear, E-stop is accessible, and no person is in the workspace.")
        print("[SAFETY] Robot and gripper motion will begin now.")

        previous_action = np.zeros(4)
        smoothing_factor = 0.7

        try:
            print("Connected to UR10e")
            print("Use controller to move robot. Press START button to exit.")
            print("Controls:")
            print("  Left stick: X/Y movement")
            print("  Right stick (vertical): Z movement (up/down)")
            print("  A: Open gripper (85 mm)")
            print("  B: Close gripper (26.5 mm)")
            print("  [Emergency Stop] Use robot's E-stop or BACK button on controller.")

            # Initial pose for servoing
            self.last_target_pose = self.rtde_r.getActualTCPPose()

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 7:  # START button
                            print("Exiting...")
                            return
                        elif event.button == 6:  # BACK button
                            print("Emergency stop!")
                            self.rtde_c.stopScript()
                            return

                action, has_input = self.get_controller_action()
                action = smoothing_factor * previous_action + (1 - smoothing_factor) * action
                previous_action = action

                if np.any(np.abs(action[:3]) > 0.0005):
                    self.send_tcp_action(action[:3]) # Use servoL based action
                else:
                    # If no motion input, continuously servo to current actual pose
                    # This helps to keep the robot still without disabling servoL
                    self.rtde_c.servoL(self.rtde_r.getActualTCPPose(), 0.05, 0.05, 0.02, 0.1, 300)
                    self.last_target_pose = self.rtde_r.getActualTCPPose()


                # Gripper control (A: open, B: close)
                if abs(action[3]) > 0.5:  # Only act if button pressed
                    self.control_gripper(action[3])

                time.sleep(0.02) # This sleep determines the loop rate, crucial for servoL

        except KeyboardInterrupt:
            print("\nStopping robot...")

        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            self.disconnect_robot()
            pygame.quit()
            print("Disconnected from robot")

# The main function in this file is only for testing UR10eJoystickControl standalone
def main():
    ROBOT_HOST = "192.168.1.102"
    robot = UR10eJoystickControl(ROBOT_HOST)

    robot.run_joystick_control()

if __name__ == "__main__":
    main()