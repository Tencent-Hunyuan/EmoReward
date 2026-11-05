# Understand and use EmoReward in your project

EmoReward helps a training system decide **which reward evidence to trust, and how much to weight it**. It combines evidence for each generated sample, then reuses experience from a completed run to calibrate the next run. The released implementation targets emotional video-generation RL; the same design can guide a custom adaptation when your project has multiple measurable criteria and a useful outcome signal.

Start with the [HTML integration hub](./index.html). This README explains the idea and maps the repository; the linked HTML contracts and workflows carry the detailed agent instructions. No browser is required to read them.

## What EmoReward can do

- **Plan:** translate an emotional prompt into observable targets, such as facial, body and temporal cues.
- **Act:** evaluate an already generated video with heterogeneous providers. EmoReward supplies a reward signal; your existing generator produces the sample.
- **Reflect:** adjust provider weights for the current sample using context, confidence, conflicts and evidence availability. Normalize over active providers rather than treating missing evidence as a neutral score.
- **Evolve:** estimate contextual provider reliability from a completed run, export bounded weight multipliers, and keep that table frozen throughout the next run. Reward-model parameters remain unchanged.

You can first add the video reward through a small adapter, then add cross-run memory when suitable traces are available. The released scorer accepts video paths; image, text or other modalities require their own observation and scoring boundary. A new domain is an adaptation to validate, not an already supported drop-in API or a guarantee of improvement.

## Adapt the design to a private project

Preserve your project's generator, optimizer, dataset interfaces and launchers. Start at its existing reward callback and choose the smallest useful change:

1. **Define success and observable evidence.** Identify the task, sample IDs, several complementary criteria, their score direction and validity conditions. For emotional video, reuse the released providers where appropriate. For another domain, choose providers that measure its actual requirements.
2. **Connect one scoring boundary.** Match sample order, batch lengths, metadata and worker ownership to your host. Reuse the [video adapter example](./examples/host_adapter.py) only for its supported inputs. Return the intended scalar once, and preserve the host's grouping and advantage computation. See the [reward contract](./contracts/reward.html).
3. **Make reflection task-specific.** Decide when each provider has enough evidence to contribute and which contexts change its usefulness. Keep missingness separate from low quality. For replacement providers, define comparable score scales before fusion; confidence gates should reflect observable support.
4. **Learn between runs.** Log provider scores, validity, context, configuration and a documented reference outcome. After the reference run finishes, estimate reliability by context and export bounded multipliers with support counts and a fallback. Load one version for the next run and leave it unchanged until that run finishes. See the [memory workflow](./workflows/memory.html).
5. **Measure before expanding.** Validate the adapter, real scoring and host training separately. Compare fixed weights, sample-level reflection and frozen memory with matched settings and held-out outcomes. Repeat the completed-run cycle only when the new evidence supports another calibration.

For example, a private product-video project might keep semantic alignment, motion and brand-style providers, condition trust on shot type or subject visibility, and calibrate against its own reviewed outcomes. A document or code project could instead use factuality, format or executable checks, with new observers and contexts. These are proposed applications of the design, not bundled providers or measured results in this release. Preserve hard requirements as constraints rather than allowing a weighted average to hide violations.

The transferable unit is **context → evidence → calibrated fusion → completed-run reliability → next-run weights**. The released seven provider names and face/arousal regimes are video-specific. Changing either requires updating the observer, gates, trace schema, exporter and runtime lookup together; adding a name to a JSON table does not add a provider. The [customization guide](./workflows/port.html#customize) explains which parts to reuse and which to replace.

Keep private data, weights and credentials in your host's existing storage and configuration. This workflow does not require uploading your project or traces. Select local providers if external processing is unsuitable; the optional external planner is a separate backend choice. Record adaptations, source revision, table provenance and observed checks in the [HTML handoff](./templates/handoff.html).

## Start your agent with one instruction

```text
Use https://github.com/Tencent-Hunyuan/EmoReward: read agent/index.html and its task-specific skills, inspect my current project's reward boundary, and adapt EmoReward's Plan/Act/Reflect/Evolve design with minimal changes to the existing training structure. Reuse compatible providers, define any custom evidence and contexts explicitly, and build frozen cross-run memory only from suitable completed-run traces. Keep private artifacts in my project, validate the integration, and report changes and evidence as HTML.
```

Use [emoreward-port](./skills/emoreward-port/SKILL.md) for the host boundary, [emoreward-memory](./skills/emoreward-memory/SKILL.md) for cross-run calibration, and [emoreward-verify](./skills/emoreward-verify/SKILL.md) to check the result. Load only the workflows relevant to your task.

## Repository Layout

Paths below are relative to the repository root. This is a focused map of tracked code and documentation, not a list of installed weights or generated outputs.

```text
EmoReward/
├── agent/                           # HTML hub, skills, adapter example and offline audit
├── post_train.py                    # GRPO training entry
├── run_emoreward.sh                 # reference run -> offline export -> next run
├── run_post_train_multinode.sh      # multi-node launcher
├── scripts/
│   ├── setup_reward_subpackages.sh  # fetch external provider code separately
│   ├── post_train/
│   │   ├── train_grpo.sh
│   │   └── pdsh_train.sh
│   ├── build_reward_traces.py       # completed-run logs -> traces
│   ├── export_jit_gate_table.py     # traces -> frozen multiplier table
│   ├── offline_gate_ladder.py       # offline analysis
│   └── jit_reward_calibration.py    # retrieval/calibration utility
├── hyvideo/
│   ├── utils/grpo_utils.py          # rollout grouping and advantages
│   └── models/reward_models/
│       ├── rewards.py              # reward dispatch
│       └── emo_reward/
│           ├── react_reward.py     # Plan / Act / Reflect + batch scorer
│           ├── reward_memory.py    # traces and memory calculations
│           ├── emotion_taxonomy.py # emotional targets and taxonomy
│           ├── emotion_scorers.py  # CLIP-AUTT and R1-Omni adapters
│           ├── openface_adapter.py # facial evidence adapter
│           ├── motion_naturalness.py
│           └── gemini_planner.py   # optional external Plan backend
├── assets/                         # bundled media and sample CSVs; not the scoring API
├── REWARD_SUBPACKAGES.md            # external provider setup and weight guide
└── checkpoints-download.md          # base model and VideoReward setup
```

External provider repositories are prepared separately by the setup script; checkpoints and generated run artifacts are not implied by this tree. For visual examples, use the [project webpage](https://emoreward-project.github.io/#visual-results). For exact source symbols and snapshot provenance, use the [code map](./references/code-map.html).
