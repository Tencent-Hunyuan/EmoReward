# Agent kit repository map

The machine entrypoint is [index.html](./index.html). This file preserves the repository layout relocated from the main README; the linked HTML workflows define integration behavior.

## Repository Layout

```text
EmoReward/
├── agent/                           # HTML hub, scoped skills, adapter example, and offline audit
├── post_train.py                     # GRPO training entry
├── run_emoreward.sh                  # one-command pipeline: Reflect (ref run) -> Evolve -> EmoReward
├── run_post_train_multinode.sh       # multi-node launcher
├── scripts/
│   ├── setup_reward_subpackages.sh   # fetch provider code (no weights)
│   ├── post_train/{train_grpo,pdsh_train}.sh
│   └── {build_reward_traces,export_jit_gate_table,offline_gate_ladder,jit_reward_calibration}.py
├── hyvideo/
│   ├── utils/grpo_utils.py           # GRPO rollout + advantages (Eq. 1–2)
│   └── models/reward_models/
│       ├── rewards.py                # reward dispatch (reward_model="emo_reward")
│       └── emo_reward/               # the EmoReward reward model
│           ├── react_reward.py       # Plan / Act / Reflect + EmoRewardScorer
│           ├── reward_memory.py      # Evolve: RewardMemory (Eq. 10–11)
│           ├── emotion_scorers.py    # CLIP-AUTT + R1-Omni providers
│           ├── openface_adapter.py   # OpenFace-3.0 AU/gaze/landmarks
│           ├── motion_naturalness.py # kinetic / motion primitive
│           ├── gemini_planner.py     # optional external Plan backend
│           └── {CLIP-AUTT, R1-Omni, OpenFace-3.0, facial-landmarks-recognition}/  # external (setup script)
├── assets/Visual_Results/            # 25 MP4s, poster frames, method figures, and gallery README
├── REWARD_SUBPACKAGES.md             # provider code + weight download guide
└── checkpoints-download.md           # base model + VideoReward download guide
```
