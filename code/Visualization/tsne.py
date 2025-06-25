import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def tsne_plot(loader, results_path, color_by="gender", perplexity=30, n_iter=1000):
    """
    loader: DataLoader yielding (X, gender, age, abn)
    color_by: "gender" | "abn"
    """
    Xs, ys = [], []
    for X, g, _, ab in loader:
        Xs.append(X)
        ys.append(g if color_by == "gender" else ab)
    Xs = torch.cat(Xs).numpy()
    ys = torch.cat(ys).numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity)
    Z = tsne.fit_transform(Xs)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=ys, cmap="coolwarm", alpha=0.7, s=8)
    cb = plt.colorbar(scatter, ticks=[ys.min(), ys.max()])
    cb.set_label(color_by)
    plt.title(f"t-SNE of latent JR vectors (colour = {color_by})")
    plt.tight_layout()
    # save a plot of the t-SNE results
    print(f"Saving t-SNE plot to {results_path}tsne_{color_by}.png")
    plt.savefig(f"{results_path}tsne_{color_by}.png")