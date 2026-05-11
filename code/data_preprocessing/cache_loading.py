"""Load cached latent features from JSONL files into PyTorch DataLoaders."""
from __future__ import annotations

import json
import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_latent_parameters_array(
    file_path: str, batch_size: int = 32
) -> DataLoader:
    """Read a JSONL file of latent feature records into a DataLoader.

    Each line must be a JSON list:
    ``[[float, ...], gender, age, abnormal, sample_id]``.

    The first element may also be a ``dict`` of named parameters; keys are
    sorted to ensure consistent ordering across rows.

    Args:
        file_path: Path to the ``.json`` file (extension appended if missing).
        batch_size: Batch size for the returned DataLoader.

    Returns:
        DataLoader with tensor samples. Sample IDs are attached as
        ``.sample_ids`` (``list[str]``).

    Raises:
        ValueError: If any line is missing the required ``sample_id`` field.
    """
    latent_params: list[tuple[torch.Tensor, ...]] = []
    if not file_path.endswith(".json"):
        file_path = file_path + ".json"

    sample_ids: list[str] = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry: list[Any] = json.loads(line)
                vec_or_dict = entry[0]
                if isinstance(vec_or_dict, dict):
                    ordered_vals = [float(vec_or_dict[k]) for k in sorted(vec_or_dict.keys())]
                    latent_vec = torch.tensor(ordered_vals, dtype=torch.float32)
                else:
                    latent_vec = torch.tensor(vec_or_dict, dtype=torch.float32)
                g = torch.tensor(entry[1], dtype=torch.float32)
                a = torch.tensor(entry[2], dtype=torch.float32)
                ab = torch.tensor(entry[3], dtype=torch.float32)
                if len(entry) < 5:
                    raise ValueError(
                        "Latent JSONL missing sample_id; expected 5 items per line."
                    )
                sample_ids.append(str(entry[4]))
                latent_params.append((latent_vec, g, a, ab))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON line: %s", exc)

    logger.info("Loaded %d latent parameters from %s", len(latent_params), file_path)
    loader = DataLoader(latent_params, batch_size=batch_size, shuffle=False)
    loader.sample_ids = sample_ids  # type: ignore[attr-defined]
    return loader
