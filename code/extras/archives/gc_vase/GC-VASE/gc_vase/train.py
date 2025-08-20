import sys
import os
#import gc_vase modules from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import wandb
from split_model import SplitLatentModel
import torch
import numpy as np
import torch.optim as optim
from utils import get_results, get_eval_results, get_split_latents, split_do_tsne, plot_latents, CustomLoader, fit_knn_fn, fit_etc_fn
from conversion_utils import get_full_conversion_results
from sklearn.decomposition import PCA
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--model_save_dir', type=str, default='./')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=0)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--recon_type', type=str, default='mse')
parser.add_argument('--content_cosine', type=int, default=1)
parser.add_argument('--attn_dim', type=int, default=64)

parser.add_argument('--data_line', type=str, default='simple')

parser.add_argument('--final_div_factor', type=int, default=10)
parser.add_argument('--initial_lr', type=float, default=0.0001)
parser.add_argument('--max_lr', type=float, default=0.0001)
parser.add_argument('--pct_start', type=float, default=0.5)

parser.add_argument('--sub_cross_s_enabled', type=int, default=0)
parser.add_argument('--sub_cross_s_weight', type=float, default=1.0)
parser.add_argument('--task_cross_t_enabled', type=int, default=0)
parser.add_argument('--task_cross_t_weight', type=float, default=1.0)

parser.add_argument('--recon_enabled', type=int, default=0)
parser.add_argument('--recon_weight', type=float, default=1.0)

parser.add_argument('--scramble_permute_enabled', type=int, default=0)
parser.add_argument('--scramble_permute_weight', type=float, default=1.0)

parser.add_argument('--conversion_permute_enabled', type=int, default=0)
parser.add_argument('--conversion_permute_weight', type=float, default=1.0)

parser.add_argument('--quadruplet_permute_enabled', type=int, default=0)
parser.add_argument('--quadruplet_permute_F_enabled', type=int, default=0)
parser.add_argument('--quadruplet_permute_weight', type=float, default=1.0)

parser.add_argument('--sub_contra_s_enabled', type=int, default=0)
parser.add_argument('--sub_contra_s_weight', type=float, default=1.0)
parser.add_argument('--task_contra_t_enabled', type=int, default=0)
parser.add_argument('--task_contra_t_weight', type=float, default=1.0)

parser.add_argument('--latent_permute_s_enabled', type=int, default=0)
parser.add_argument('--latent_permute_s_weight', type=float, default=1.0)
parser.add_argument('--latent_permute_t_enabled', type=int, default=0)
parser.add_argument('--latent_permute_t_weight', type=float, default=1.0)

parser.add_argument('--restored_permute_s_enabled', type=int, default=0)
parser.add_argument('--restored_permute_s_weight', type=float, default=1.0)
parser.add_argument('--restored_permute_t_enabled', type=int, default=0)
parser.add_argument('--restored_permute_t_weight', type=float, default=1.0)

parser.add_argument('--sub_content_enabled', type=int, default=0)
parser.add_argument('--sub_content_weight', type=float, default=1.0)
parser.add_argument('--task_content_enabled', type=int, default=0)
parser.add_argument('--task_content_weight', type=float, default=1.0)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--use_tqdm', type=int, default=1)

parser.add_argument('--group', type=str, default='')

parser.add_argument('--override_seed', type=int, default=None)

parser.add_argument('--extra_tags', type=str, default='')

parser.add_argument('--full_eval', type=int, default=1)

parser.add_argument('--eval_every', type=int, default=70)
parser.add_argument('--sched_mode', type=str, default='max')
parser.add_argument('--sched_patience', type=int, default=5)
parser.add_argument('--sched_factor', type=float, default=0.5)
parser.add_argument('--old_sched', type=int, default=1)
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--conversion_N', type=int, default=2000)
parser.add_argument('--extra_classifiers', type=int, default=1)
parser.add_argument('--conversion_results', type=int, default=1)

args, unknown = parser.parse_known_args()

loss_to_notation = {
    'recon': ['R'],
    'sub_contra_s': ['SL', 'CR:s'],
    'task_contra_t': ['SL', 'CR:t'],
    'latent_permute_s': ['SL', 'LP:s'],
    'latent_permute_t': ['SL', 'LP:t'],
    'restored_permute_s': ['SL', 'RP:s'],
    'restored_permute_t': ['SL', 'RP:t'],
    'sub_content': ['SL', 'C:s'],
    'task_content': ['SL', 'C:t'],
    'sub_cross_s': ['CE:s'],
    'task_cross_t': ['CE:t'],
    'scramble_permute': ['SP'],
    'conversion_permute': ['CP'],
    'quadruplet_permute': ['QP'],
    'quadruplet_permute_f': ['QPf'],
}

if __name__ == '__main__':
    print(args, file=sys.stdout, flush=True)
    if args.random_seed:
        SEED = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    elif args.override_seed is not None:
        SEED = args.override_seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    else:
        SEED = 3242342323 + 1
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    # Infer input channels from data if available; fallback to 30
    NUM_LAYERS = args.num_layers
    KERNEL_SIZE = 4

    USE_TQDM = args.use_tqdm
    OLD_SCHED = bool(args.old_sched)

    # Load data and infer channels before model init
    with torch.no_grad():
        data_dict = torch.load(args.data_dir + f"{args.data_line}_data.pt")
        # Try to infer channels
        try:
            sample = data_dict["data"][0]
            IN_CHANNELS = int(sample.shape[0])
        except Exception:
            IN_CHANNELS = 19
        # Normalize if stats exist
        if "data_mean" in data_dict and "data_std" in data_dict:
            data_dict["data"] = (data_dict["data"] - data_dict["data_mean"]) / data_dict["data_std"]
        preferred_split = 'train' if 'split' not in data_dict else data_dict['split']
        loader = CustomLoader(data_dict, split=preferred_split)
        if getattr(loader, 'size', 0) == 0:
            # Fallback to generic split
            loader = CustomLoader(data_dict, split='all')
        del data_dict

    # Initialize model after knowing IN_CHANNELS
    model = SplitLatentModel(
        IN_CHANNELS,
        args.channels,
        args.latent_dim,
        NUM_LAYERS,
        KERNEL_SIZE,
        recon_type=args.recon_type,
        content_cosine=args.content_cosine,
    )

    # Define losses
    all_losses = [
        "recon", "sub_contra_s", "task_contra_t", "latent_permute_s", "latent_permute_t",
        "restored_permute_s", "restored_permute_t", "sub_content", "task_content",
        "sub_cross_s", "task_cross_t", "scramble_permute", "conversion_permute",
        "quadruplet_permute", "quadruplet_permute_F"
    ]
    losses = []
    loss_weights = defaultdict(lambda: 1.0)
    if args.recon_enabled:
        losses.append("recon")
        loss_weights["recon"] = args.recon_weight
    if args.sub_contra_s_enabled:
        losses.append("sub_contra_s")
        loss_weights["sub_contra_s"] = args.sub_contra_s_weight
    if args.task_contra_t_enabled:
        losses.append("task_contra_t")
        loss_weights["task_contra_t"] = args.task_contra_t_weight
    if args.latent_permute_s_enabled:
        losses.append("latent_permute_s")
        loss_weights["latent_permute_s"] = args.latent_permute_s_weight
    if args.latent_permute_t_enabled:
        losses.append("latent_permute_t")
        loss_weights["latent_permute_t"] = args.latent_permute_t_weight
    if args.restored_permute_s_enabled:
        losses.append("restored_permute_s")
        loss_weights["restored_permute_s"] = args.restored_permute_s_weight
    if args.restored_permute_t_enabled:
        losses.append("restored_permute_t")
        loss_weights["restored_permute_t"] = args.restored_permute_t_weight
    if args.sub_content_enabled:
        losses.append("sub_content")
        loss_weights["sub_content"] = args.sub_content_weight
    if args.task_content_enabled:
        losses.append("task_content")
        loss_weights["task_content"] = args.task_content_weight
    if args.sub_cross_s_enabled:
        losses.append("sub_cross_s")
        loss_weights["sub_cross_s"] = args.sub_cross_s_weight
    if args.task_cross_t_enabled:
        losses.append("task_cross_t")
        loss_weights["task_cross_t"] = args.task_cross_t_weight
    if args.scramble_permute_enabled:
        losses.append("scramble_permute")
        loss_weights["scramble_permute"] = args.scramble_permute_weight
    if args.conversion_permute_enabled:
        losses.append("conversion_permute")
        loss_weights["conversion_permute"] = args.conversion_permute_weight
    if args.quadruplet_permute_enabled:
        if args.quadruplet_permute_F_enabled:
            losses.append("quadruplet_permute_F")
            loss_weights["quadruplet_permute_F"] = args.quadruplet_permute_weight
        else:
            losses.append("quadruplet_permute")
            loss_weights["quadruplet_permute"] = args.quadruplet_permute_weight

    model.set_losses(
        batch_size=args.batch_size,
        losses=losses,
        loader=loader,
        loss_weights=loss_weights,
    )

    numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {numel}", file=sys.stdout, flush=True)

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define steps/epoch based on dataset size and batch size
    steps_per_epoch = max(1, (loader.size + args.batch_size - 1) // args.batch_size)
    BATCHES = args.epochs * steps_per_epoch
    print(f"steps_per_epoch: {steps_per_epoch}", file=sys.stdout, flush=True)

    div_factor = args.max_lr / args.initial_lr
    if OLD_SCHED:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            div_factor=div_factor,
            max_lr=args.max_lr,
            steps_per_epoch=1,
            epochs=BATCHES,
            three_phase=False,
            pct_start=args.pct_start,
            final_div_factor=args.final_div_factor,
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args.sched_mode,
            factor=args.sched_factor,
            patience=args.sched_patience,
        )

    # Early stopping config (fixed, no CLI):
    EARLY_STOP_PATIENCE = 5
    best_epoch_loss = float('inf')
    epochs_without_improve = 0

    # Training loop with clean epochs and epoch loss reporting
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = defaultdict(list)
        with tqdm(range(steps_per_epoch), disable=not USE_TQDM, file=sys.stdout) as pbar:
            for _ in pbar:
                optimizer.zero_grad()
                x, loss_dict = model.losses()
                total_loss = sum((model.loss_weights[v] * loss_dict[v] for v in model.used_losses))
                total_loss.backward()
                optimizer.step()
                if OLD_SCHED:
                    scheduler.step()
                # track losses
                epoch_losses['total'].append(float(total_loss.detach().cpu().item()))
                for k in model.used_losses:
                    epoch_losses[k].append(float(loss_dict[k].detach().cpu().item()))
                pbar.set_postfix({"loss": f"{epoch_losses['total'][-1]:.6f}"})
        # epoch end
        mean_total = float(np.mean(epoch_losses['total'])) if len(epoch_losses['total']) else float('nan')
        per_loss_str = ", ".join([f"{k}:{np.mean(v):.6f}" for k, v in epoch_losses.items() if k != 'total'])
        print(f"Epoch {epoch+1}/{args.epochs} - total:{mean_total:.6f}" + (f" | {per_loss_str}" if per_loss_str else ""), file=sys.stdout, flush=True)
        if not OLD_SCHED:
            scheduler.step(mean_total)

        # Early stopping check
        if mean_total < best_epoch_loss - 0.0:  # min_delta = 0.0
            best_epoch_loss = mean_total
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best total loss: {best_epoch_loss:.6f})", file=sys.stdout, flush=True)
                break

    # Save the model with architecture hints in filename for easy inference matching
    if args.save_model:
        import re
        from pathlib import Path
        Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
        # Sanitize helper
        def s(x):
            return re.sub(r"[^0-9A-Za-z_.-]", "", str(x))
        # Build informative filename
        fname = (
            f"gcvase_"
            f"in{IN_CHANNELS}_c{args.channels}_d{args.latent_dim}_"
            f"L{NUM_LAYERS}_k{KERNEL_SIZE}_"
            f"recon{int(bool(args.recon_enabled))}w{s(args.recon_weight)}_"
            f"scr{int(bool(args.scramble_permute_enabled))}_"
            f"conv{int(bool(args.conversion_permute_enabled))}_"
            f"ep{args.epochs}_bs{args.batch_size}.pt"
        )
        save_as = str(Path(args.model_save_dir) / fname)
        torch.save(model.state_dict(), save_as)
        print(f"Saved model to {save_as}", file=sys.stdout, flush=True)
