from xmlrpc import client
import time

# Configuration (Update these values)
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
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

# Initialize gripper (optional but recommended)
try:
    # Set release limit (80mm) and no-part detection (1mm)
    server.SetReleaseLimit(gid, 1, 80.0)
    server.SetNoPartLimit(gid, 1, 25.0)
    print("Gripper initialized")
except:
    print("Initialization commands not supported (may be in simulation)")

# Control functions
def grip():
    return server.Grip(gid, 1)  # Second parameter is mode

def release():
    return server.Release(gid, 1)

def get_status():
    return server.GetState(gid)

def get_position():
    return server.GetPos(gid)


if __name__ == "__main__":
    print("Releasing gripper...")
    release()
    time.sleep(2)
    print(f"Position: {get_position()} mm")
    print(f"Status: {get_status()}")

    print("\nGripping...")
    grip()
    time.sleep(2)
    print(f"Position: {get_position()} mm")
    print(f"Status: {get_status()}")