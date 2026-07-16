import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

STAGE1_PATH = Path("results/stage1_trajectories.json")
OUTPUT_PATH = Path("plots/tsne_trajectories.png")
TARGET_LENGTH = 100


def load_stage1_trajectories(path: Path) -> Dict[str, dict]:
    with path.open("r") as handle:
        return json.load(handle)


def resample_sequence(values: List[float], target_length: int = TARGET_LENGTH) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(target_length, dtype=np.float32)

    if len(values) == target_length:
        return np.asarray(values, dtype=np.float32)

    x_old = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(x_new, x_old, np.asarray(values, dtype=np.float32)).astype(np.float32)


def build_feature_matrix(data: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[str] = []

    for sample_id, info in data.items():
        margins = info.get("margins", [])
        flips = int(info.get("flips", 0))

        features.append(resample_sequence(margins, TARGET_LENGTH))
        labels.append(flips)
        sample_ids.append(sample_id)

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int32), sample_ids


def map_flips_to_class(flips: int) -> str:
    if flips < 10:
        return "Easy"
    if 10 <= flips <= 40:
        return "Ambiguous"
    return "Hard"


def compute_tsne_embeddings(x: np.ndarray) -> np.ndarray:
    n_samples, n_features = x.shape
    n_components = min(50, n_features, n_samples - 1)
    x_reduced = x
    if n_components < n_features:
        x_reduced = PCA(n_components=n_components, random_state=42).fit_transform(x)

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=42,
        verbose=1,
    )
    return tsne.fit_transform(x_reduced)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    categories = np.asarray([map_flips_to_class(int(flips)) for flips in labels])
    df = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "flips": labels,
            "category": categories,
        }
    )

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 9))
    palette = {"Easy": "#2ca02c", "Ambiguous": "#ff7f0e", "Hard": "#d62728"}
    ax = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="category",
        palette=palette,
        s=12,
        alpha=0.55,
        linewidth=0,
    )
    ax.set_title("t-SNE of Stage 1 Margin Trajectories")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="Flips Class", loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    data = load_stage1_trajectories(STAGE1_PATH)
    x, labels, _ = build_feature_matrix(data)
    embeddings = compute_tsne_embeddings(x)
    plot_tsne(embeddings, labels, OUTPUT_PATH)
    print(f"Saved t-SNE plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
