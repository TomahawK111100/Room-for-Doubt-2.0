import os, json, torch, numpy as np, argparse
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.model import ResNetClassifier
from src.train_cfm_head import CFMTrajectoryModel
from src.utils import set_seed

class DualTransformCIFAR(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.t_s = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.t_p = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img, label = item[0], item[1] if len(item)==2 else item[1]
        return self.t_s(img), self.t_p(img), label

@torch.no_grad()
def generate_trajectories(model, img_features, classes, num_samples=5):
    B, device = img_features.size(0), img_features.device
    img_features = img_features.repeat_interleave(num_samples, dim=0)
    classes = classes.repeat_interleave(num_samples, dim=0)
    x = torch.randn(B * num_samples, 50, device=device)
    dt = 1.0 / 10
    for step in range(10):
        x = x + model.vector_field(x, torch.full((B * num_samples,), step * dt, device=device), img_features, classes) * dt
    return x.view(B, num_samples, 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open(f"results/stage1_trajectories_{args.dataset}_seed{args.seed}.json", 'r') as f: traj = json.load(f)
    X, y = [], []
    for k, v in traj.items():
        if int(k) < 45000:
            m = v.get('margins', [])[:50]
            X.append(m + [0.0] * max(0, 50 - len(m)))
            y.append(1 if v.get('noisy_label',0) != v.get('true_label', v.get('clean_label',0)) else 0)
    lda = LinearDiscriminantAnalysis().fit(np.array(X), np.array(y))

    student = ResNetClassifier.load_from_checkpoint(f"src/checkpoints/best_{args.dataset}n_worse_seed{args.seed}.ckpt", num_classes=100).to(device).eval()
    cfm = CFMTrajectoryModel(num_classes=100).to(device)
    cfm.load_state_dict(torch.load(f"checkpoints/cfm_head_{args.dataset}_seed{args.seed}.pth", map_location=device))
    cfm.eval()

    loader = DataLoader(DualTransformCIFAR(torchvision.datasets.CIFAR100(root='./data', train=False, download=False)), batch_size=64, shuffle=False, num_workers=2)
    
    all_labels, all_c1, all_c2, all_b1, all_b2 = [], [], [], [], []
    print("Evaluating CFM on CIFAR-100N Test Set...")
    for img_s, img_p, labels in loader:
        img_s, img_p, labels = img_s.to(device), img_p.to(device), labels.to(device)
        top2 = torch.topk(student(img_s), k=2, dim=1).indices
        c1, c2 = top2[:, 0], top2[:, 1]
        
        feats = cfm.extract_features(img_p)
        t_c1 = generate_trajectories(cfm, feats, c1).cpu().numpy()
        t_c2 = generate_trajectories(cfm, feats, c2).cpu().numpy()
        
        all_b1.extend(np.mean([lda.predict_proba(t_c1[:, i, :])[:, 1] for i in range(5)], axis=0))
        all_b2.extend(np.mean([lda.predict_proba(t_c2[:, i, :])[:, 1] for i in range(5)], axis=0))
        all_labels.extend(labels.cpu().numpy()); all_c1.extend(c1.cpu().numpy()); all_c2.extend(c2.cpu().numpy())

    all_labels, all_c1, all_c2, all_b1, all_b2 = map(np.array, [all_labels, all_c1, all_c2, all_b1, all_b2])
    base_acc = (all_c1 == all_labels).mean()
    corr_acc = (np.where((all_b1 > 0.5) & (all_b2 < 0.5), all_c2, all_c1) == all_labels).mean()
    
    print(f"Baseline Accuracy:  {base_acc*100:.2f}%")
    print(f"Corrected Accuracy: {corr_acc*100:.2f}% (Gain: +{(corr_acc - base_acc)*100:.2f}%)")

if __name__ == '__main__': main()
