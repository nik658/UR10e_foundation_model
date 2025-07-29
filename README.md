
# Pi-0 Imitation Learning on UR10e using LeRobot Interface

This project implements and tests the **Pi-0 Imitation Learning model** on a **UR10e robot** available in the **IRIS Lab, University of Aveiro**, using the [LeRobot](https://github.com/openrobotlab/lerobot) interface.

---

## Motivation

To explore and validate state-of-the-art imitation learning models like **Pi-0** and **smol-vla** on real robotic hardware, and build an end-to-end pipeline from **teleoperation-based data collection** to **policy deployment on the UR10e**.

---

## Installation

1. **Clone LeRobot**:
   Follow the installation steps on the [LeRobot GitHub repository](https://github.com/openrobotlab/lerobot).

2. **Clone this repository**:
   This repo contains the UR10e-specific scripts for teleoperation, training, and inference.

---

## Project Stages

### **Stage 1: Simulation Prototyping (Gym-XArm)**

* Implemented imitation learning policies such as **ACT**, **TDMPC**, **smol-vla** and **pi0** in the `gym-xarm` environment.
* Collected datasets using joystick-based teleoperation.
* Datasets are pushed to [HuggingFace Datasets](https://huggingface.co/nik658) under the user ID `@nik658`.

---

### **Stage 2: Real Robot Setup**

* Setup of **RTDE control** and **teleoperation of the UR10e** using a joystick.
* Integration of **Kinect** sensor for visual feedback and observation.
* Later added 1080p camera feed of c99 camera as another camera feed

---

### **Stage 3: Local Policy Execution**

* Developed scripts for:

  * **Recording episodes** via joystick teleoperation.
  * **Pushing collected episodes** to a HuggingFace dataset repo.
  * **Training LeRobot policies** on the pushed data.
  * **Running inference locally** on the trained Pi-0 model.

* **Problem:** Local inference with Pi-0 was computationally heavy (GPU unavailable, CPU too slow).

* **Solution:** Implemented **asynchronous inference** to decouple observation and action steps.

---

### **Stage 4: Remote Inference via Socket**

* Established a **socket connection** between:

  * **Local PC** (runs RTDE control and Kinect)
  * **Remote GPU cluster** (runs the policy inference)

This setup allows real-time execution of high-compute models like Pi-0 while keeping robot control responsive locally.


---


