
---

### Remote Async Inference via Pleiades GPU Cluster

This setup enables **asynchronous inference** by splitting observation and action computation between the **local machine** and a **remote GPU server (Pleiades)**.

---

####  Local Machine Responsibilities

* Starts a **server** that:

  * Captures and processes **observations** (images and robot states).
  * Sends observations to the GPU cluster.
  * Receives and executes the predicted **actions**.

---

####  Remote GPU (Pleiades) Inference Flow

1. **SSH Tunnel**
   Set up a tunnel to forward local port `5000` to the remote cluster:

   ```bash
   ssh -L 5000:localhost:5000 username@pleiades.ieeta.pt
   ```

   This forwards communication between:

   * `localhost:5000` on the **local PC**
   * `localhost:5000` on the **Pleiades cluster**

2. **Run `pleiades_proxy`** on Pleiades
   In one terminal on Pleiades, start the bridge:

   ```bash
   python pleiades_proxy.py
   ```

   This script handles communication between the local server and the inference model.

3. **Start Training/Inference via SLURM**
   In another terminal (on Pleiades), launch the actual inference client (e.g., `inference_server.py` or `train.py`) using SLURM:

   ```bash
   sbatch run_server.sh
   ```

   This script connects to the local machine, receives the observations, runs inference on the GPU, and sends back the action.

---
