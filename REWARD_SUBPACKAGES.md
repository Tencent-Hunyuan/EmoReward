# EmoReward — reward sub-package weights

The EmoReward reward model uses four external emotional-reward providers. Their **code**
is fetched by `scripts/setup_reward_subpackages.sh`; their **weights** are downloaded
separately as described below. All providers **degrade gracefully** — if a provider's
weights are missing, that provider is dropped and its fusion weight is redistributed, so
the pipeline still runs for a smoke test. Download the weights you need for full fidelity.

All paths below are relative to the reward-module directory:
`hyvideo/models/reward_models/emo_reward/` (called `<MOD>` below).

## 0. Fetch the provider code first

```bash
# CLONE from upstream (default):
bash scripts/setup_reward_subpackages.sh
# …or COPY code from an existing local checkout (weights excluded):
SAGE_GRPO_SRC=/path/to/SAGE-GRPO bash scripts/setup_reward_subpackages.sh
```

---

## 1. CLIP-AUTT  (facial AU / valence, CLIP ViT-B/32)

Trained checkpoint (BioVid pain/neutral), expected at:

```
<MOD>/CLIP-AUTT/WeightFiles/Biovid/lab_srcs77_biovid_ep10_bs8_sql16_str2_vid.pth
```

Obtain the checkpoint from the [CLIP-AUTT upstream](https://github.com/osamazeeshan/CLIP-AUTT)
(training/weights instructions in its README) and place it at the path above. CLIP ViT-B/32
backbone weights are pulled automatically by the `clip` package on first use.

## 2. R1-Omni  (semantic emotion + VAD, HumanOmni-0.5B)

R1-Omni needs its own model plus two encoders and a tokenizer. Download into `<MOD>/R1-Omni/`:

```bash
cd <MOD>/R1-Omni
# main model (HF or ModelScope)
huggingface-cli download StarJiaxing/R1-Omni-0.5B     --local-dir ./R1-Omni-0.5B
# encoders + tokenizer referenced by R1-Omni-0.5B/config.json
huggingface-cli download openai/whisper-large-v3      --local-dir ./whisper-large-v3
huggingface-cli download google/siglip-base-patch16-224 --local-dir ./siglip-base-patch16-224
huggingface-cli download google-bert/bert-base-uncased  --local-dir ./AI-ModelScope/bert-base-uncased
```

> **Important:** edit `R1-Omni-0.5B/config.json` so `mm_audio_tower` and `mm_vision_tower`
> point at your **local** `whisper-large-v3` and `siglip-base-patch16-224` directories
> (the shipped config may contain an absolute path from another machine). The BERT path is
> resolved from `<MOD>/R1-Omni/AI-ModelScope/bert-base-uncased`.

Expected layout:

```
<MOD>/R1-Omni/
├── R1-Omni-0.5B/            # main model + config.json
├── whisper-large-v3/        # audio tower
├── siglip-base-patch16-224/ # vision tower
├── AI-ModelScope/bert-base-uncased/
├── inference.py  humanomni/  src/  yamls/   # (code, from setup script)
```

## 3. OpenFace-3.0  (AU / gaze / landmark extraction)

Place the OpenFace-3.0 model weights in `<MOD>/OpenFace-3.0/weights/`:

```
<MOD>/OpenFace-3.0/weights/
├── MTL_backbone.pth              # multi-task (Emotion/Gaze/AU) head
├── Alignment_RetinaFace.pth      # RetinaFace detector
├── mobilenetV1X0.25_pretrain.tar # RetinaFace backbone
├── Landmark_68.pkl               # STAR landmark model (68-pt)
└── Landmark_98.pkl               # STAR landmark model (98-pt)
```

Download them from the [OpenFace-3.0 upstream](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0)
(see its README for the weights release).

## 4. facial-landmarks-recognition  (dlib 68-pt landmarks)

Used by the OpenFace pipeline. Place the dlib predictor at:

```
<MOD>/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat
```

Download from the [upstream repo](https://github.com/italojs/facial-landmarks-recognition)
or the canonical dlib model release, then decompress the `.dat`.

---

## Base checkpoints (DiT / VAE / text encoders / VideoAlign)

These are the HunyuanVideo base model and the general video-reward model, **not** the
emotion providers — see `checkpoints-download.md` and `download_weights.sh`. The VideoAlign
reward path defaults to `./ckpts/VideoReward` and can be overridden with `VIDEOALIGN_MODEL_PATH`.
