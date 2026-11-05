---
name: emoreward-memory
description: Build, inspect or attach a frozen RewardMemory calibration table from completed-run reward traces; distinguish the released runtime from retrieval demos.
---

Read the [HTML hub](../../index.html) and [memory workflow](../../workflows/memory.html). Follow its source links before changing trace extraction, outcome selection or runtime gate configuration.

Keep the seven-provider exporter/runtime and the nine-channel retrieval analysis separate. Use explicit trace provenance, finite values, support counts and a higher-is-better outcome with a documented source. Legacy face and VAD proxies are not exact measurements.

A completed reference run precedes export; the next training run reads a frozen table. Detect missing/invalid tables and sparse-regime fallback. Do not turn the released recipe into an online writeback loop or count agreement with its own fused reward as independent validation.

Report table hash, reference identity, outcome source and resolved eta/tau/cap in the [HTML handoff](../../templates/handoff.html). Verify behavior using the [verification workflow](../../workflows/verify.html).
