"""Video-path boundary example; real provider loading is explicit and deferred."""
from math import isfinite
from pathlib import Path


class EmoRewardAdapter:
    def __init__(self, score_fn):
        self.score_fn = score_fn

    @classmethod
    def from_upstream(cls, device, config=None):
        # Resolve dependencies and reward-worker ownership before this import.
        from hyvideo.models.reward_models.emo_reward.react_reward import emo_reward_score
        return cls(emo_reward_score(device=device, config=config))

    def __call__(self, video_paths, prompts, metadata=None):
        paths, prompts = list(video_paths), list(prompts)
        if len(paths) != len(prompts):
            raise ValueError("One prompt is required per video")
        metadata = [{} for _ in paths] if metadata is None else list(metadata)
        if len(metadata) != len(paths):
            raise ValueError("One metadata dictionary is required per video")
        if any(not isinstance(row, dict) for row in metadata):
            raise TypeError("Metadata rows must be dictionaries")
        if not paths:
            return []
        paths = [str(Path(path)) for path in paths]
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        scores, _ = self.score_fn(paths, prompts, [dict(row) for row in metadata])
        values = scores["emo_total"]
        if len(values) != len(paths):
            raise ValueError("Scorer returned a truncated or expanded batch")
        values = [float(value) for value in values]
        if not all(isfinite(value) for value in values):
            raise ValueError("Non-finite reward; apply an explicit host failure policy")
        return values
