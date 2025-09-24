import argparse, json, os, time, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from datasets.medmnist_loader import get_medmnist_loaders
from models.baseline_cnn import BaselineCNN

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__(); self.a=alpha; self.g=gamma; self.red=reduction
    def forward(self, logits, targets):
        p = torch.sigmoid(logits).clamp(1e-6,1-1e-6)
        y = targets.float()
        pos = -self.a * (1-p).pow(self.g) * torch.log(p) * y
        neg = -(1-self.a) * (p).pow(self.g) * torch.log(1-p) * (1-y)
        loss = pos+neg
        return loss.mean() if self.red=="mean" else loss.sum()

class ASL(nn.Module):
    # 간단 멀티라벨 ASL
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05):
        super().__init__(); self.gp=gamma_pos; self.gn=gamma_neg; self.clip=clip
    def forward(self, logits, targets):
        y = targets.float()
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip>0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = y * torch.pow(1 - xs_pos, self.gp) * torch.log(xs_pos.clamp(1e-6,1-1e-6))
        los_neg = (1 - y) * torch.pow(xs_neg, self.gn) * torch.log((1 - xs_pos).clamp(1e-6,1-1e-6))
        loss = - (los_pos + los_neg)
        return loss.mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chestmnist")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--backbone", default="mobilenetv3_small_100")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--loss", default="bce", choices=["bce","focal","asl"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--outdir", default="results_student")
    args = ap.parse_args()

    train_loader, val_loader, test_loader, meta = get_medmnist_loaders(
        args.dataset, args.batch_size, args.img_size, augment="light"
    )
    n_classes = meta["n_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = BaselineCNN(n_classes, backbone=args.backbone, pretrained=args.pretrained, multi_label=True).to(device)
    if args.loss=="bce": criterion = nn.BCEWithLogitsLoss()
    elif args.loss=="focal": criterion = FocalLoss()
    else: criterion = ASL()

    opt = AdamW(model.parameters(), lr=args.lr)
    auroc = MultilabelAUROC(num_labels=n_classes).to(device)
    auprc = MultilabelAveragePrecision(num_labels=n_classes).to(device)

    best = -1; exp=f"{args.dataset}_{args.backbone}_{args.loss}_ep{args.epochs}"
    od = os.path.join(args.outdir, exp); os.makedirs(od, exist_ok=True)
    ckpt = os.path.join(od,"ckpt.pt"); test_json=os.path.join(od,"test.json")

    for ep in range(1, args.epochs+1):
        model.train(); run=0
        for x,y in train_loader:
            x=x.to(device); y=y.squeeze().to(device)
            opt.zero_grad(set_to_none=True)
            loss=criterion(model(x), y.float()); loss.backward(); opt.step()
            run += loss.item()*x.size(0)
        # val
        model.eval(); auroc.reset(); auprc.reset(); y_t=[]; y_p=[]
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device); yv=y.squeeze().to(device)
                z=model(x); p=torch.sigmoid(z)
                auroc.update(p, yv); auprc.update(p, yv)
                y_t.append(y.squeeze()); y_p.append((p.cpu()>0.5).long())
        v_auroc=float(auroc.compute().item()); v_auprc=float(auprc.compute().item())
        y_t=torch.cat(y_t).numpy(); y_p=torch.cat(y_p).numpy()
        f1=float(f1_score(y_t,y_p,average="macro",zero_division=0))
        print(json.dumps({"ep":ep,"val_auroc":v_auroc,"val_auprc":v_auprc,"val_f1":f1}))
        if v_auprc>best: best=v_auprc; torch.save(model.state_dict(), ckpt)

    # test
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    auroc.reset(); auprc.reset(); y_t=[]; y_p=[]
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device); yv=y.squeeze().to(device)
            z=model(x); p=torch.sigmoid(z)
            auroc.update(p, yv); auprc.update(p, yv)
            y_t.append(y.squeeze()); y_p.append((p.cpu()>0.5).long())
    t_auroc=float(auroc.compute().item()); t_auprc=float(auprc.compute().item())
    y_t=torch.cat(y_t).numpy(); y_p=torch.cat(y_p).numpy()
    f1=float(f1_score(y_t,y_p,average="macro",zero_division=0))
    with open(test_json,"w") as f: json.dump({"auroc":t_auroc,"auprc":t_auprc,"f1_macro":f1}, f, indent=2)
    print("TEST:", json.dumps({"auroc":t_auroc,"auprc":t_auprc,"f1_macro":f1}, indent=2))

if __name__=="__main__":
    main()
