from __future__ import annotations

from pathlib import Path

import torch


class DirectMLQwen3Backbone:
    name = "torch-directml"

    def __init__(self, config: dict, weights_path: str, prefix_min_tokens: int = 96, device=None):
        from transformers import AutoConfig, AutoModel

        self.torch = torch
        self.device = torch.device(device)
        root = str(Path(weights_path).parent)
        cfg = AutoConfig.from_pretrained(root)
        rope = config.get("rope_parameters") or {}
        cfg.rope_theta = rope.get("rope_theta", config.get("rope_theta", getattr(cfg, "rope_theta", None)))
        cfg.use_cache = False
        self.model = AutoModel.from_pretrained(root, config=cfg, dtype=torch.float16, attn_implementation="sdpa")
        self.model.to(self.device).eval()
        self.prefix_min_tokens = prefix_min_tokens

    def _pad(self, rows, pad):
        lengths = [len(r) for r in rows]
        width = max(lengths)
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row)
            att[i, : len(row)] = 1
        return ids.to(self.device), att.to(self.device), lengths

    def hidden_rows(self, prefix, suffixes, pad):
        rows = [list(prefix) + list(suffix) for suffix in suffixes]
        ids, att, lengths = self._pad(rows, pad)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=att, use_cache=False)
            hidden = out.last_hidden_state.float().cpu().numpy()
        return [hidden[i, :n] for i, n in enumerate(lengths)]
