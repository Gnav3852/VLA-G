# SPIKE Findings: OpenVLA + LIBERO Trajectory Extraction

## Path taken

**OpenVLA + LIBERO directly** (not allenai/vla-evaluation-harness).

Reason: The harness decouples model inference from sim via WebSocket and only records
`reward`, `done`, and `success` per step. Per-step EE trajectories and object poses are
available in the sim observation dict but are not exported. Extending the harness would
require subclassing `LIBEROBenchmark` plus Docker + model server — more plumbing than
instrumenting the existing OpenVLA eval loop, which already reads `robot0_eef_pos` each step.

## Setup

- Model: `openvla/openvla-7b-finetuned-libero-spatial` (OpenVLA-7B, LIBERO-Spatial finetune)
- Suite: `libero_spatial`, task_id=0
- Task: pick up the black bowl between the plate and the ramekin and place it on the plate
- Episodes: 20 (episode_idx 0..19 selects LIBERO init states)
- Model seed: 7, env seed: 0

## Results summary

- Rollouts completed: 20
- Successes: 18/20 (90.0%)
- Output CSV: `/Users/gsn89/ReadTeam/spike_rollouts.csv`
- Trajectory files: `/Users/gsn89/ReadTeam/spike_trajectories/seed_*.json`

---

## 1. TRAJECTORY ACCESS

**Yes.**

Per-step end-effector position is in the LIBERO/robosuite observation dict every step:
`obs["robot0_eef_pos"]` → `(x, y, z)` world frame. The spike script appends this after
the settle period and before/after each policy step.

How: read from `obs` inside the rollout loop (same keys OpenVLA already uses for proprio).
Full paths stored as JSON lists in `/Users/gsn89/ReadTeam/spike_trajectories/`; CSV column `ee_trajectory_path`
points to each file. Also logged: `final_ee_x/y/z`, `num_steps`.

---

## 2. OBJECT POSE ACCESS

**Yes.**

After `env.set_init_state(...)` and the 10 settle steps, object positions
are in obs as `{object_name}_pos` (from robosuite object observables, enabled by default
via `use_object_obs=True`). Target object name comes from `env.obj_of_interest[0]` (the
manipulated object in LIBERO BDDL).

This run used target object: `akita_black_bowl_1`.

---

## 3. SEED VARIATION

**Yes** — varying `episode_idx` (selects different entries in `task_suite.get_task_init_states`)
changes initial object layout.

Rough position ranges across 20 episodes (target object, post-settle):

| Axis | Range (cm) | Std (cm) |
|------|------------|----------|
| x | 5.29 | 1.52 |
| y | 3.67 | 1.04 |
| z | 0.06 | 0.01 |

Note: LIBERO eval varies **episode_idx** (init state index), not `env.seed`. The OpenVLA
reference hardcodes `env.seed(0)`. `model_seed` only affects policy stochasticity
(OpenVLA uses greedy decoding here).

---

## Blockers

None for data extraction. This run used **Apple MPS** (float16 + SDPA) as a dev fallback;
on a CUDA machine use the setup in the script header (bf16 + flash-attn-2, no patches needed).
