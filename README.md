<p align="center">
  <img src="./assets/Figure/logo_color.png" width="200" height="200" alt="EmoReward logo">
</p>

<h3 align="center">EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL</h3>

[![Project Page](https://img.shields.io/badge/Project-Page-ff6f61.svg)](https://emoreward-project.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2611.06518-b31b1b.svg)](https://arxiv.org/abs/2611.06518) [![Hugging Face Paper](https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E.svg)](https://huggingface.co/papers/2611.06518) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/Transformers-4.57.1-yellow.svg)](https://github.com/huggingface/transformers) [![Diffusers](https://img.shields.io/badge/Diffusers-0.35.0-blue.svg)](https://github.com/huggingface/diffusers) [![Base Model](https://img.shields.io/badge/Base%20Model-HunyuanVideo--1.5-6a5acd.svg)](https://github.com/Tencent-Hunyuan/HunyuanVideo) [![RL Backbone](https://img.shields.io/badge/RL%20Backbone-SAGE--GRPO-0ea5e9.svg)](https://github.com/Tencent-Hunyuan/SAGE-GRPO)

[Mingzhe Zheng](https://mingzhezheng.homes/)<sup>1,2,&ast;</sup>, [Yue Wu](https://scholar.google.com/citations?user=1xTR6qoAAAAJ&hl=en)<sup>2,&Dagger;</sup>, [Jing Tan](https://sparkstj.github.io/)<sup>2</sup>, [Chang Zou](https://shenyi-z.github.io/)<sup>2</sup>, [Dengyang Jiang](https://scholar.google.com/citations?user=tJcxeMoAAAAJ&hl=en)<sup>1</sup>, [Zeyue Tian](https://scholar.google.com/citations?user=dghq4MQAAAAJ)<sup>1</sup>, [Xuanhua He](https://scholar.google.com/citations?user=-bDAN2YAAAAJ)<sup>1</sup>, [Ryan Xu](#)<sup>2</sup>, [Yunyang Ge](https://scholar.google.com/citations?user=bbgjlg0AAAAJ)<sup>2</sup>, [Sihui Ji](https://sihuiji.github.io/)<sup>2</sup>, [Yunlong Lin](https://lyl1015.github.io/)<sup>2</sup>, [Kaihang Pan](https://scholar.google.com/citations?user=lMQADDUAAAAJ)<sup>2</sup>, [Hongru Yan](#)<sup>2</sup>, [Sixiang Chen](https://ephemeral182.github.io/)<sup>2</sup>, [Dianyi Wang](https://scholar.google.com/citations?user=iP2HPFEAAAAJ)<sup>3</sup>, [Julian Zhang](#)<sup>4</sup>, [Weijie Kong](https://scholar.google.com/citations?hl=zh-CN&user=gsOklKAAAAAJ)<sup>2</sup>, [Changlin Li](https://scholar.google.com/citations?user=wOQjqCMAAAAJ)<sup>2</sup>, [Lanqing Guo](https://guolanqing.github.io/)<sup>2</sup>, [Zhao Zhong](https://scholar.google.com/citations?user=igtXP_kAAAAJ&hl=en)<sup>2</sup>, [Qifeng Chen](https://scholar.google.com/citations?user=lLMX9hcAAAAJ&hl=en)<sup>1,&dagger;</sup>, [Harry Yang](https://scholar.google.com/citations?hl=en&user=jpIFgToAAAAJ&view_op=list_works)<sup>1,&dagger;</sup>

<sup>1</sup>HKUST &nbsp; <sup>2</sup>Tencent Hunyuan &nbsp; <sup>3</sup>Fudan University &nbsp; <sup>4</sup>University of Waterloo<br> <sup>&ast;</sup>Equal contribution &nbsp; <sup>&dagger;</sup>Corresponding Authors &nbsp; <sup>&Dagger;</sup>Project Leader<br> Work done during internship at Tencent Hunyuan


EmoReward is a **self-evolving training mechanism** for emotional video-generation RL. It distills *evidence reliability* from reward traces collected during an actual training run, then uses it to guide multi-reward weight allocation in the next run — deciding which emotional evidence to trust and how much to weight it, **without updating any reward model**. Within a video it calibrates heterogeneous cues per instance (**Reflective Reward Gating**); across runs it fixes bounded, experience-derived multipliers (**RewardMemory**). It plugs into SAGE-GRPO training on HunyuanVideo-1.5 as a drop-in reward, fused with general video rewards, and transfers to DanceGRPO and FlowGRPO. More information and a visual gallery are available on the [webpage](https://emoreward-project.github.io/).

  


![EmoReward motivation: calibrating emotional evidence](./assets/Visual_Results/motivation.png)

## Highlights

We cast emotional video-generation RL as a problem of **trusting the right emotional evidence**: emotion is subjective, temporal, and easy to game with fixed reward proxies. Our three contributions turn training experience into reusable reward calibration.

- **EmoReward — a self-evolving training mechanism.** We introduce a mechanism for emotional video-generation RL that calibrates heterogeneous reward providers within videos and across training runs, without updating reward-model parameters. Reflective Reward Gating uses phase, face, conflict, and subtlety cues to decide which evidence to trust for each video.

- **RewardMemory — memory for multi-reward video generation.** We propose a memory that distills contextual evidence reliability from a completed run into bounded fusion-weight multipliers. These multipliers tilt reward allocation in the next run, while the memory table remains frozen during that run.

- **Cross-run self-evolution — reusable training experience.** We introduce a paradigm for multimodal post-training in which a completed training run informs reward allocation in the next, making experience reusable at the run level. Experiments span three video RL algorithms and two image backbones, alongside ablations and human evaluations.

## Quick start for agents

This README is the human-readable guide. For coding agents, our [HTML integration hub](./agent/index.html) provides focused skills for understanding EmoReward's design, adapting its reward interface, and integrating it into your own project. The [repository map](./agent/README.md) lives alongside the agent kit.

Give your agent this instruction in your project:

```text
Use https://github.com/Tencent-Hunyuan/EmoReward: read agent/index.html and follow its linked skills to integrate EmoReward into my current project, preserving the host training loop and validating the reward contract before running training; report the integration and evidence as HTML.
```

## Abstract

Group Relative Policy Optimization (GRPO) has advanced generative model post-training in verifiable
domains such as code and mathematics, but evaluating video quality — especially emotional
expression — often depends on subjective aesthetics and has no uniquely correct target. Existing
approaches typically rely on fixed rewards for visual quality, motion quality, and text alignment
that poorly capture emotional intent and can invite reward hacking, while reinforcement learning
from human feedback (RLHF) is difficult to scale and subject to individual aesthetic differences
that complicate the calibration of facial, body motion, and chromatic cues across narrative
contexts.

We introduce **EmoReward**, a new self-evolving training mechanism that distills evidence
reliability from reward traces collected during an actual training run to adaptively guide
multi-reward weight allocation in subsequent training by determining which emotional evidence to
trust and how much to weight it. EmoReward first **plans** specific grounding cues from the
emotional prompt to guide evaluation, then uses heterogeneous signal primitives to score the
generated video during the **act** stage. *Reflective Reward Gating* calibrates their contributions
through instance-level gates for phase, face confidence, conflict, and subtlety, then renormalizes
weights over the active set to produce the sample's fused reward. *RewardMemory* then estimates each
primitive's reliability from rollout scores and outcomes within each emotional regime in the
finished training run, and fixes bounded multipliers that calibrate fusion weights in the next
training run without updating reward models.

Experiments on HunyuanVideo-1.5 show improvements across multiple RL algorithms and compare
EmoReward with other agentic reward methods under SAGE-GRPO. Ablations and transfer experiments on
SD 3.5 and FLUX further support adaptive reward calibration for visual generation.

## Table of Contents

- [Highlights](#highlights)
- [Quick start for agents](#quick-start-for-agents)
- [Abstract](#abstract)
- [Plan · Act · Reflect · Evolve](#plan--act--reflect--evolve)
- [Installation](#installation)
- [Checkpoint & Reward-Provider Preparation](#checkpoint--reward-provider-preparation)
- [Quickstart: Train with EmoReward](#quickstart-train-with-emoreward)
- [Reward Policies (Ablation Ladder)](#reward-policies-ablation-ladder)
- [Key Parameters](#key-parameters)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

## Plan · Act · Reflect · Evolve

![Four-stage EmoReward framework: Plan, Act, Reflect, and Evolve](./assets/Visual_Results/overview.png)

EmoReward maps a prompt `p` and generated video `x` to a calibrated reward through four stages
(mirrored on the [project page](https://emoreward-project.github.io/)):

```
Plan     p ──▶ EmoScript e(p)              target valence/arousal, AUs, body cues, expression window
Act      (x,e) ──▶ {s_k, u_k}              per-primitive score s_k and evidence validity u_k
Reflect  (s,u,e,x) ──▶ R_emo               instance gates g_k + active-provider renormalization
Evolve   M ──▶ m_k(r) = exp(Δ_k)           memory-derived reliability multipliers (frozen next run)
```

- **Plan — ground the intent.** A prompt is grounded into an *EmoScript*: target valence/arousal,
facial action units (AUs), body cues, and an expression window.
- **Act — observe the video.** Heterogeneous reward primitives score the video — facial/AU, semantic
emotion, body, temporal, CLIP-AUTT, R1-Omni, motion naturalness (chromatic/kinetic as auxiliary
readouts) — each returning a score `s_k` and an evidence-validity `u_k`.
- **Reflect — calibrate this sample.** Phase-, conflict-, face-, and subtlety-aware gates decide
which primitives to trust; the fused reward is renormalized over only the *active* providers `A`.
- **Evolve — learn from a completed run.** From a *finished* reference run's traces, EmoReward
estimates each primitive's reliability per emotional regime and exports bounded, closed-form
multipliers for the *next* run.

Reflect composes the current-video gates with the frozen memory multipliers, then renormalizes over
the active set `A`:

```
R_emo = Σ_{k∈A} (w_k⁰ · g_k · m_k) · s_k  /  (Σ_{k∈A} w_k⁰ · g_k · m_k + ε)
```

The reward fed to GRPO fuses a general video reward `R_vid` (VideoAlign) with `R_emo`:

```
R_total = ( α · R_vid  +  β · R_emo ) / (α + β)
```

**Reflective Reward Gating** uses `m_k = 1` (the webpage's *Reflect, without RewardMemory*);
**EmoReward** uses the frozen table `m_k(r)` from Evolve (the webpage's full *EmoReward*). The
multipliers come from each primitive's reliability advantage over a completed reference run:

```
ρ_k(z)  = Corr(s_k, y)             agreement of primitive k with the reference outcome y, in regime z
Δ_k     = clip( (ρ_k − mean_j ρ_j) / τ , −c, c )
m_k(r)  = exp(Δ_k)                 reliable providers (ρ high) ↑, constant/low-variance providers ↓
```

EmoReward never updates reward-model parameters; only RewardMemory and the closed-form fusion
weights change.

## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/Tencent-Hunyuan/EmoReward.git
cd EmoReward
pip install -r requirements.txt
```

### 2. Fetch the reward sub-package code

The four emotional-reward providers live as external code packages inside the reward module. Fetch
them at their pinned commits (no weights):

```bash
bash scripts/setup_reward_subpackages.sh
# or copy code from an existing local checkout (weights excluded):
# SAGE_GRPO_SRC=/path/to/SAGE-GRPO bash scripts/setup_reward_subpackages.sh
```

## Checkpoint & Reward-Provider Preparation

EmoReward needs three groups of weights, all resolved under `./ckpts` and inside the reward module.

1. **Base model + general video reward** — HunyuanVideo-1.5 and VideoReward (VideoAlign):
   ```bash
    bash download_weights.sh          # VideoReward
   ```

    then follow `checkpoints-download.md` for the HunyuanVideo-1.5 DiT/VAE, text encoders, and
    vision encoder. The VideoAlign path defaults to `./ckpts/VideoReward` (override with
    `VIDEOALIGN_MODEL_PATH`).
2. **Emotional-reward provider weights** — CLIP-AUTT, R1-Omni, OpenFace-3.0, and

   facial-landmarks-recognition. Follow `REWARD_SUBPACKAGES.md` for the exact download commands
    and target paths.

> **Graceful degradation:** any provider whose weights are missing is dropped and its fusion weight
> is redistributed, so the pipeline still imports and runs for a smoke test. Download the providers
> you need for full emotional fidelity.

For the base-model setup, see the upstream [SAGE-GRPO checkpoint guide](https://github.com/Tencent-Hunyuan/SAGE-GRPO/blob/master/checkpoints-download.md). The optional HPSv3/COHP FLUX.1-schnell model uses `./ckpts/FLUX.1-schnell` (override with `FLUX_SCHNELL_MODEL_PATH`); its download command is in [checkpoints-download.md](./checkpoints-download.md#optional-flux1-schnell-for-hpsv3cohp).

### Expected checkpoint layout

```text
ckpts/
├── transformer/            # HunyuanVideo-1.5 DiT
├── vae/
├── text_encoder/           # MLLM + byT5 (see checkpoints-download.md)
├── vision_encoder/siglip/
├── VideoReward/            # general video reward (VideoAlign); or set VIDEOALIGN_MODEL_PATH
└── FLUX.1-schnell/         # optional HPSv3/COHP; or set FLUX_SCHNELL_MODEL_PATH
```

## Quickstart: Train with EmoReward

EmoReward self-evolves across two training runs. A single command chains both:

```bash
bash run_emoreward.sh
```

This runs, end to end:

1. **emoreward without RewardMemory** — a short reference run (100 steps by default) with

   Reflective Reward Gating only (memory multipliers `m_k = 1`). Its rollouts are logged to
    `rl_samples/`. This is the webpage's *Reflect (without RewardMemory)*.
2. **Evolve (offline, GPU-free)** — distill RewardMemory from those traces:

   `build_reward_traces.py` → `traces.jsonl`, then `export_jit_gate_table.py` →
    `jit_gate_table.json` (the regime-keyed reliability-multiplier table).
3. **emoreward with RewardMemory** — a full run with Reflect gates composed with the frozen

   memory table. This is the full **EmoReward**.

Common overrides:

```bash
STAGE1_STEPS=100  bash run_emoreward.sh                      # length of the reference run
STAGE2_STEPS=10000 bash run_emoreward.sh                     # length of the full run
SKIP_STAGE1=1 STAGE1_DIR=SAGE_GRPO_results/<ref_run> \
  bash run_emoreward.sh                                      # reuse an existing reference run
```

If the memory table is missing, Stage 2 falls back to `adaptive` (= Reflect only).

### Single-node / multi-node

```bash
bash run_post_train_multinode.sh            # calls scripts/post_train/pdsh_train.sh "scripts/post_train/train_grpo.sh"
```

Set `REWARD_MODEL=emo_reward` (the default in `train_grpo.sh`) to route rewards through EmoReward.
Edit or export the node list and rendezvous-related environment expected by your cluster launcher
before starting.

## Reward Policies (Ablation Ladder)

The paper's controlled study fixes the backbone and optimization budget and changes **only** the
reward policy. Every arm is reachable from the same `train_grpo.sh` by exporting
`REWARD_GATE_POLICY` (and, for the lower bound, the fusion weights) — no code or script changes.
`run_emoreward.sh` runs the last two arms back to back.

| Reward policy | `REWARD_GATE_POLICY` | What it isolates |
| --- | --- | --- |
| VideoAlign only | `adaptive`, α=1 / β=0 | **lower bound** — emotional reward off |
| Single provider (R1-Omni) | `single` | one provider vs. gating |
| Random routing | `random` | is *smart* selection needed? |
| Round-robin routing | `roundrobin` | non-adaptive fair scheduling |
| Fixed fusion | `fixed` | dense reward, no reflect |
| **Reflect** (without RewardMemory) | `adaptive`, α=β=0.5 | **Reflective Reward Gating** — Stage 1 of `run_emoreward.sh` |
| **EmoReward** (with RewardMemory) | `jit` + table | **+ RewardMemory tilt** — Stage 2 of `run_emoreward.sh` |

> *Reflect* and *EmoReward* differ only in the memory multipliers: Reflect uses `m_k = 1`, EmoReward
> uses the frozen table from Evolve. To reproduce a single ablation arm, export the policy and call
> the launcher directly, e.g.:
>
> ```bash
> REWARD_GATE_POLICY=fixed bash run_post_train_multinode.sh                 # fixed fusion
> VIDEOALIGN_FUSION_ALPHA=1.0 VIDEOALIGN_FUSION_BETA=0.0 \
>   REWARD_GATE_POLICY=adaptive bash run_post_train_multinode.sh            # VideoAlign-only lower bound
> ```

## Key Parameters

### Hardware Recommendation

| Requirement | Recommended |
| --- | --- |
| GPU memory | 80 GB per GPU |
| GPU count | 32 GPUs (4 nodes × 8) |
| OS | Linux |
| PyTorch | 2.6+ |

### Reward fusion & gating

Exported by `run_emoreward.sh`, consumed by `train_grpo.sh` → `post_train.py`:

| Env var / flag | Default | Description |
| --- | --- | --- |
| `REWARD_GATE_POLICY` | `adaptive` | Reward policy: `fixed`/`single`/`random`/`roundrobin`/`adaptive`/`jit`. |
| `REWARD_GATE_JIT_TABLE` | — | Regime→multiplier table for the `jit` (RewardMemory) policy. |
| `VIDEOALIGN_FUSION_ALPHA` / `_BETA` | `0.5` / `0.5` | α/β in `R_total` (VideoAlign-only lower bound uses 1.0 / 0.0). |
| `EMO_REWARD_COMBINE_VIDEOALIGN` | `true` | Fuse VideoAlign Overall with the emotional reward. |
| `EMO_REWARD_PLAN_ENABLED` | `False` | Enable the external Plan (Gemini) backend (off by default). |
| `CLIP_AUTT_WEIGHT` / `R1OMNI_WEIGHT` | `0.0` / `0.12` | Per-provider weights. |
| `MOTION_NATURALNESS_WEIGHT` | `0.24` | Motion-naturalness primitive weight. |

### GRPO / distributed

`sp_size`, `batch_size`, and `num_generations` are the key knobs — the recipe uses batch 2,
group `G=4`. Constraint:

```text
(batch_size * world_size / sp_size) % num_generations == 0
```

> **Planner credentials:** the Gemini Plan backend (optional, off by default) reads gateway keys
> from `EMO_REWARD_PLAN_API_APP_IDS` and `EMO_REWARD_PLAN_API_KEY_<app_id>`. No credentials are
> shipped in this repo.

## Acknowledgements

- Base video model: [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo)
- RL backbone: [SAGE-GRPO](https://github.com/Tencent-Hunyuan/SAGE-GRPO)
- General video reward: [VideoReward / VideoAlign](https://github.com/KwaiVGI/VideoAlign)
- Emotional-reward providers: [CLIP-AUTT](https://github.com/osamazeeshan/CLIP-AUTT),
[R1-Omni](https://github.com/HumanMLLM/R1-Omni),
[OpenFace-3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0),
[facial-landmarks-recognition](https://github.com/italojs/facial-landmarks-recognition)

## License

This project is released under the license in [LICENSE](./LICENSE). Third-party base models, reward
models, and emotional-reward providers remain under their respective upstream licenses — review and
comply with each before use.

## Citation

If you find EmoReward useful, please consider citing:

```bibtex
@misc{zheng2026emoreward,
  title={EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL},
  author={Zheng, Mingzhe and Wu, Yue and Tan, Jing and Zou, Chang and Jiang, Dengyang and
          Tian, Zeyue and He, Xuanhua and Xu, Ryan and Ge, Yunyang and Ji, Sihui and
          Lin, Yunlong and Pan, Kaihang and Yan, Hongru and Chen, Sixiang and Wang, Dianyi and
          Zhang, Julian and Kong, Weijie and Li, Changlin and Guo, Lanqing and Zhong, Zhao and
          Chen, Qifeng and Yang, Harry},
  year={2026}
}
```
