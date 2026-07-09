import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision
from torchvision.transforms import ToPILImage

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_top_ambiguous(data, top_k=10):
    # Sort by number of flips descending, then by variance of margins ascending if margins exist
    items = []
    for sid, info in data.items():
        margins = info.get("margins", [])
        var_margin = np.var(margins) if len(margins) > 0 else 0
        items.append((sid, info, var_margin))
    
    # Sort by flips DESC, then by margin variance DESC
    items.sort(key=lambda x: (x[1].get('flips', 0), x[2]), reverse=True)
    return items[:top_k]

def plot_cartography(stage1_path, stage2_path, data_dir, output_path):
    stage1_data = load_json(stage1_path) if stage1_path else None
    stage2_data = load_json(stage2_path) if stage2_path else None
    
    # Let's decide which stage to use for finding ambiguous images. Default to stage2 if exists, else stage1
    target_data = stage2_data if stage2_data else stage1_data
    if not target_data:
        print("No valid JSON provided!")
        return

    top_items = get_top_ambiguous(target_data, top_k=10)
    sids = [int(x[0]) for x in top_items]

    # Load CIFAR-10 data to extract images
    # By default, PyTorch Lightning loads validation from train split based on indices,
    # but the whole dataset is available in CIFAR10 train=True.
    cifar = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False)
    
    fig, axes = plt.subplots(nrows=len(sids), ncols=3, figsize=(15, 3 * len(sids)))
    sns.set_style("whitegrid")
    
    for row_idx, (sid_str, info, var_m) in enumerate(top_items):
        sid = int(sid_str)
        img, _ = cifar[sid] # Assuming the sample_id matches CIFAR10 train indices
        
        ax_img = axes[row_idx, 0] if len(sids) > 1 else axes[0]
        ax_margin = axes[row_idx, 1] if len(sids) > 1 else axes[1]
        ax_dist = axes[row_idx, 2] if len(sids) > 1 else axes[2]
        
        # Plot Image
        ax_img.imshow(img)
        ax_img.axis("off")
        flips = info.get('flips', 0)
        ax_img.set_title(f"Sample ID: {sid} | Flips: {flips}\nTrue: {info['true_label']}, Noisy: {info['noisy_label']}")

        # Margins Plot
        if stage1_data and sid_str in stage1_data and len(stage1_data[sid_str].get("margins", [])) > 0:
            ax_margin.plot(stage1_data[sid_str]["epochs"], stage1_data[sid_str]["margins"], label="Stage 1", marker='o', markersize=4, alpha=0.7)
        if stage2_data and sid_str in stage2_data and len(stage2_data[sid_str].get("margins", [])) > 0:
            ax_margin.plot(stage2_data[sid_str]["epochs"], stage2_data[sid_str]["margins"], label="Stage 2", marker='x', markersize=4, alpha=0.7)
        
        ax_margin.set_title("Predictive Margin")
        ax_margin.set_xlabel("Epoch")
        ax_margin.set_ylabel("Margin")
        ax_margin.legend()

        # Descriptor Distance Plot
        has_dist = False
        if stage1_data and sid_str in stage1_data and len(stage1_data[sid_str].get("descriptor_distances", [])) > 0:
            ax_dist.plot(stage1_data[sid_str]["epochs"], stage1_data[sid_str]["descriptor_distances"], label="Stage 1", marker='o', markersize=4, alpha=0.7, color='green')
            has_dist = True
        if stage2_data and sid_str in stage2_data and len(stage2_data[sid_str].get("descriptor_distances", [])) > 0:
            ax_dist.plot(stage2_data[sid_str]["epochs"], stage2_data[sid_str]["descriptor_distances"], label="Stage 2", marker='x', markersize=4, alpha=0.7, color='red')
            has_dist = True
        
        if has_dist:
            ax_dist.set_title("Descriptor Distance")
            ax_dist.set_xlabel("Epoch")
            ax_dist.set_ylabel("L2 Distance")
            ax_dist.legend()
        else:
            ax_dist.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center')
            ax_dist.axis('off')
            ax_dist.set_title("Descriptor Distance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Data Cartography.")
    parser.add_argument("--stage1", type=str, help="Path to stage1_trajectories.json", default=None)
    parser.add_argument("--stage2", type=str, help="Path to stage2_trajectories.json", default=None)
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing CIFAR10 data")
    parser.add_argument("--output", type=str, default="cartography_plot.png", help="Output plot image path")
    args = parser.parse_args()

    plot_cartography(args.stage1, args.stage2, args.data_dir, args.output)
