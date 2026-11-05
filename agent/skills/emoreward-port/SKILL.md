---
name: emoreward-port
description: Integrate EmoReward reward calibration into an existing video-generation or RL project while preserving its scoring contract and the host optimizer.
---

Use this skill when adding EmoReward to a host project, or adapting its reward interface to another trainer, custom evidence pool or modality. Start from the [HTML hub](../../index.html), then read the [reward contract](../../contracts/reward.html) and [port workflow](../../workflows/port.html).

Identify the host callback, modality, grouping, step metadata and device ownership before choosing an adapter. Inspect the relevant source symbols in the [code map](../../references/code-map.html); do not assume the entire training stack or every provider is required.

For new providers or contexts, follow the [customization guide](../../workflows/port.html#customize). Preserve the host structure and define task targets, score semantics, validity and gates before changing the pipeline. A new modality needs an observer; the video scorer is not a universal callback.

Reuse the [adapter example](../../examples/host_adapter.py) only after matching its video-path input and scalar output to the host. Preserve active-set fusion, fallback diagnostics and one application of the general-video mixture. Direct factory configuration and shell launcher flags are different entrypoints.

If cross-run memory is requested, follow the [memory workflow](../../workflows/memory.html) and its linked skill. Finish with the [verification ladder](../../workflows/verify.html) and an [HTML handoff](../../templates/handoff.html). Record what was actually run; preserve the user's training/deployment scope.
