import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_top_ambiguous(data, top_k=10):
    items = []
    for sid, info in data.items():
        margins = info.get("margins", [])
        var_margin = np.var(margins) if len(margins) > 0 else 0
        items.append((sid, info, var_margin))
    
    items.sort(key=lambda x: (x[1].get('flips', 0), x[2]), reverse=True)
    return items[:top_k]

def plot_scatter_cartography(stage1_data, plots_dir):
    """
    1) Классический Data Cartography Scatter Plot
    Ось X - flips, ось Y - AUM (mean_margin).
    Точки разделены на 3 зоны: Easy, Hard, Ambiguous.
    """
    flips = []
    aums = []
    
    for sid, info in stage1_data.items():
        flips.append(info.get('flips', 0))
        # Получаем сохраненный AUM или вычисляем из margins
        if 'aum' in info:
            aums.append(info['aum'])
        else:
            margins = info.get('margins', [])
            aums.append(float(np.mean(margins)) if margins else 0.0)
            
    if not flips or not aums:
        print("Нет данных для Scatter Plot.")
        return
        
    flips = np.array(flips)
    aums = np.array(aums)
    
    # Определяем зоны по перцентилям
    flip_q = np.percentile(flips, 50)
    aum_q = np.percentile(aums, 50)
    
    # Раскраска по зонам
    categories = []
    for f, a in zip(flips, aums):
        if f >= flip_q and a >= aum_q:
            categories.append('Ambiguous')
        elif f < flip_q and a >= aum_q:
            categories.append('Easy')
        else:
            # Если confidence ниже медианы (в том числе f >= flip_q, a < aum_q) - отнесем к Hard
            # Обычно: низкая вариативность, низкий AUM -> Hard. 
            categories.append('Hard')

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=flips, y=aums, hue=categories, palette={'Easy': 'green', 'Hard': 'red', 'Ambiguous': 'orange'}, alpha=0.6, s=20, edgecolor=None)
    
    plt.axvline(flip_q, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(aum_q, color='gray', linestyle='--', alpha=0.5)
    
    plt.title("Data Cartography Map (Stage 1)")
    plt.xlabel("Variability (Flips)")
    plt.ylabel("Confidence (AUM)")
    plt.legend(title="Zone")
    plt.tight_layout()
    
    out_path = plots_dir / "stage1_cartography_scatter_zones.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {out_path}")

def plot_flips_histogram(stage1_data, plots_dir):
    """
    2) Гистограмма распределения метрики flips.
    """
    flips = [info.get('flips', 0) for info in stage1_data.values()]
    
    if not flips:
        print("Нет данных для гистограммы.")
        return
        
    plt.figure(figsize=(8, 6))
    sns.histplot(flips, bins=max(10, min(50, max(flips)-min(flips)+1)), kde=False, color='blue', discrete=True)
    
    plt.title("Flips Distribution (Stage 1)")
    plt.xlabel("Number of Flips")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    
    out_path = plots_dir / "stage1_flips_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved flips histogram to {out_path}")

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
    parser.add_argument("--stage1", type=str, help="Path to stage1_trajectories.json", default="results/stage1_trajectories.json")
    parser.add_argument("--stage2", type=str, help="Path to stage2_trajectories.json", default=None)
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing CIFAR10 data")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Output plots directory")
    parser.add_argument("--output", type=str, default="cartography_plot.png", help="Output plot image path for top-10 ambiguous")
    args = parser.parse_args()
    
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if args.stage1 and Path(args.stage1).exists():
        stage1_data = load_json(args.stage1)
        plot_scatter_cartography(stage1_data, plots_dir)
        plot_flips_histogram(stage1_data, plots_dir)
    else:
        print(f"Внимание: Файл {args.stage1} не найден. Пропуск построения общих графиков Картографии.")

    # Top-10 plot
    output_path = plots_dir / args.output
    plot_cartography(args.stage1, args.stage2, args.data_dir, output_path)
