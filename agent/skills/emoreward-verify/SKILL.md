---
name: emoreward-verify
description: Validate an EmoReward port with source, adapter, scoring and training evidence kept distinct, and diagnose fusion or frozen-memory mismatches.
---

Use the [HTML verification workflow](../../workflows/verify.html) and [reward contract](../../contracts/reward.html). Begin with the standard-library source audit linked by the [hub](../../index.html); it runs without loading providers.

Choose meaningful checks for the changed host boundary: batch alignment, real metadata, metric selection, failures, provider support, warmup, table fallback/hash and distributed reward ownership. Use a fake scorer for interface tests; use real clips for claims about real scoring.

Record exact evidence and statuses in an [HTML handoff](../../templates/handoff.html). Do not upgrade static checks into training evidence or internal proxy agreement into independent emotional quality. Follow only the evidence levels supported by the available environment and user-authorized task.
