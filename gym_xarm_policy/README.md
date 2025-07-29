
---

### `gym_xarm` Usage Guide
Follow the steps of https://github.com/huggingface/gym-xarm for the isntallation of the environment in the same conda anvironemt as lerobot
Ensure the following setup for proper simulation and training:

* Set `MUJOCO_GL=glfw` if you encounter rendering issues with MuJoCo.
* Adjust the **scaling factor** to tune the robot’s movement speed during teleoperation or evaluation.

---

#### Dataset Collection

Use the `XARM_JOYSTICK_RECORD` script to record demonstration datasets via joystick control. The data is automatically pushed to the associated HuggingFace repository under the user ID [`nik658`](https://huggingface.co/nik658).

---

#### Model Training

You can train a model using the following example command:

```bash
MUJOCO_GL=glfw python src/lerobot/scripts/train.py \
  --output_dir=outputs/train/smol_UR10_29jul_2cammm \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=nik658/ur10e_2cam \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=nik658/ur10e_150ep_smol_2cam \
  --steps=40000
```

* The model is saved to HuggingFace and training metrics can be monitored via **Weights & Biases (W\&B)**.
* You can specify different policy backbones (e.g., `smolvla_base`, `pi0`, etc.).

---

#### Model Evaluation

Evaluate a trained policy and save performance metrics and videos:

```bash
python src/lerobot/scripts/eval.py \
  --policy.path=xarm_pi0/pretrained_model \
  --output_dir=outputs/eval/pi0_xarm/30k/lift_cube \
  --env.type=xarm \
  --env.task=XarmLift-v0 \
  --eval.n_episodes=5 \
  --eval.batch_size=5 \
  --policy.device=cpu
```

* Output includes **evaluation videos** stored under the specified `output_dir`.
* Logs include **mean reward**, **max reward**, **reward sum**, and **average reward per episode**.
---
