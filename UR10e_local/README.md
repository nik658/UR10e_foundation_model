---

### Running Inference on UR10e with LeRobot

This setup is intended for running **inference on a UR10e robot**, which is **not natively supported** in LeRobot. The interface is adapted to work using RTDE and custom scripts.

---

#### Requirements

* Ensure the **RTDE control script** is present and executable in the same directory. This script handles low-level communication with the UR10e.
* The UR10e must be powered on, reachable over LAN, and correctly configured (e.g., IP: `192.168.1.102`).

---

#### Script: `run_policy_ur10e.py`

* This script runs inference on a **pretrained policy** using LeRobot.
* Make sure to:

  * **Import the correct policy** (e.g., ACT, Pi0, SmolVLA).
  * **Set the correct path** to the `pretrained_model/` directory, typically obtained from LeRobot’s training output (`outputs/train/...`).
  * The policy folder **must be local** (not from HuggingFace or remote sources).

---

#### Policy-Specific Notes

* **ACT**:

  * Does **not require a task** name to be passed.
  * The observation can be used directly.

* **SmolVLA / Pi0 / Pi0Fast**:

  * A **task name is required** as part of the observation dictionary.

---
