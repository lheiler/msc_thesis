import argparse
from pathlib import Path
import torch
import numpy as np
import sys


def build_from_gen_dataset(
    root_dir: str,
    output_path: str,
    segment_seconds: int = 60,
    target_sfreq: float = 128.0,
    window_seconds: float | None = 2.0,
    hop_seconds: float | None = None,
    drop_remainder: bool = True,
):
    # Import your project's dataset class
    try:
        from code.utils.gen_dataset import TUHFIF60sDataset
    except Exception:
        # Try to add project root to path: this file is .../testing/GC-VASE/dataset_preparation
        repo_root = Path(__file__).resolve().parents[3]
        sys.path.append(str(repo_root))
        try:
            from code.utils.gen_dataset import TUHFIF60sDataset
        except Exception:
            # Avoid conflict with stdlib 'code' module by loading via file path
            import importlib.util
            gen_path = repo_root / "code" / "utils" / "gen_dataset.py"
            spec = importlib.util.spec_from_file_location("gen_dataset_module", str(gen_path))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            TUHFIF60sDataset = module.TUHFIF60sDataset

    ds = TUHFIF60sDataset(Path(root_dir), segment_len_sec=segment_seconds, target_sfreq=target_sfreq)

    data_list = []
    subjects = []
    tasks = []
    runs = []

    # Simple labeling scheme: each file is one subject; tasks=0; runs=1
    # You can customize this mapping by parsing folder names
    # Prepare windowing setup
    if window_seconds is None:
        window_seconds = 2.0
    if hop_seconds is None:
        hop_seconds = window_seconds
    target_T = int(round(window_seconds * target_sfreq))
    hop_T = int(round(hop_seconds * target_sfreq))

    for idx in range(len(ds)):
        x_np = ds[idx].numpy().astype(np.float32)  # (C, T_total)
        T_total = x_np.shape[1]
        if T_total < target_T:
            if drop_remainder:
                continue
            # pad to at least one window
            pad = target_T - T_total
            x_np = np.pad(x_np, ((0,0),(0,pad)), mode='constant')
            T_total = x_np.shape[1]
        # Slide windows
        end_limit = T_total - target_T
        made_any = False
        for s in range(0, end_limit + 1, hop_T):
            w = x_np[:, s:s+target_T]
            data_list.append(torch.from_numpy(w))
            subjects.append(idx + 1)  # 1-indexed subjects
            tasks.append(0)
            runs.append(1)
            made_any = True
        if not made_any and not drop_remainder:
            # take last centered window
            start = max(0, T_total - target_T)
            w = x_np[:, start:start+target_T]
            data_list.append(torch.from_numpy(w))
            subjects.append(idx + 1)
            tasks.append(0)
            runs.append(1)

    data = torch.stack(data_list, dim=0)  # (N, C, T)

    # Compute dataset statistics
    data_mean = data.mean()
    data_std = data.std().clamp(min=1e-8)

    labels = {0: 'Generic'}

    save_dict = {
        'data': data,
        'subjects': torch.tensor(subjects, dtype=torch.long),
        'tasks': torch.tensor(tasks, dtype=torch.long),
        'runs': torch.tensor(runs, dtype=torch.long),
        'labels': labels,
        'data_mean': data_mean,
        'data_std': data_std,
        'split': 'all',
    }

    torch.save(save_dict, output_path)
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root folder containing .fif files')
    parser.add_argument('--output', type=str, required=True, help='Path to save e.g. generic_data.pt')
    parser.add_argument('--segment_seconds', type=int, default=60)
    parser.add_argument('--sfreq', type=float, default=128.0)
    parser.add_argument('--window_seconds', type=float, default=2.0)
    parser.add_argument('--hop_seconds', type=float, default=None)
    parser.add_argument('--keep_remainder', action='store_true')
    args = parser.parse_args()

    out = build_from_gen_dataset(
        args.root,
        args.output,
        args.segment_seconds,
        args.sfreq,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        drop_remainder=not args.keep_remainder,
    )
    print(f"Saved dataset to {out}")


