Here’s a cleaned-up and professional version of your Kinect and UR10e setup instructions:

---

### Kinect + UR10e Setup

#### 1. **Install `libfreenect` and Python Wrapper**

Install the necessary drivers and Python bindings for Kinect:

```bash
sudo apt-get install libfreenect-dev
pip install freenect
```

#### 2. **Test Kinect Feed**

Use `freenect-glview` to verify that the RGB and depth streams are working:

```bash
freenect-glview
```

---

### UR10e + RTDE Control

* **RTDE** is used to control the UR10e robot programmatically.
* Make sure both the **laptop** and **robot** are connected to the **same LAN network**.

#### Network Configuration

| Device | IP Address      |
| ------ | --------------- |
| Laptop | `192.168.1.101` |
| UR10e  | `192.168.1.102` |

> UR10e IP can be set in the robot’s interface under: `Settings → Network`

---

### Gripper Integration

* The gripper used is `wgripper`.

* It is integrated using a fork of the ROS MoveIt implementation:
  [https://github.com/iris-ua/iris\_sami](https://github.com/iris-ua/iris_sami)

* The control logic is in the modified file:
  `src/sami/gripper.py`
  This has been adapted specifically for the UR10e + LeRobot setup.

---

Let me know if you want all of this merged into one complete README or broken into sections like `docs/hardware_setup.md`.
