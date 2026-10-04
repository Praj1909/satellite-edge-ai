"""Train SatNet on EuroSAT.  python core/train.py --epochs 10"""
import os, sys, json, time, argparse
import torch, torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import SatNet, save_model
from data.dataset import load_eurosat

LOG_PATH = os.path.join(os.path.dirname(__file__),"..","models","train_log.json")

def train(epochs=10, lr=1e-3, batch_size=32):
    print(f"\n{'='*50}\n SatNet Training — EuroSAT\n{'='*50}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Device: {device}  Epochs: {epochs}\n")
    train_loader, val_loader, _ = load_eurosat(batch_size=batch_size)
    print(f" Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}\n")
    model = SatNet().to(device)
    print(f" Parameters: {model.count_params():,}\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    best_acc, log = 0.0, []
    for epoch in range(1, epochs+1):
        model.train(); tl=tc=tt=0; t0=time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(); out=model(imgs); loss=criterion(out,labels)
            loss.backward(); optimizer.step(); scheduler.step()
            tl+=loss.item()*imgs.size(0); tc+=(out.argmax(1)==labels).sum().item(); tt+=imgs.size(0)
        ta=tc/tt; tl=tl/tt
        model.eval(); vl=vc=vt=0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out=model(imgs); loss=criterion(out,labels)
                vl+=loss.item()*imgs.size(0); vc+=(out.argmax(1)==labels).sum().item(); vt+=imgs.size(0)
        va=vc/vt; vl=vl/vt
        print(f" Epoch {epoch:02d}/{epochs}  train_acc={ta:.3f}  val_acc={va:.3f}  ({time.time()-t0:.0f}s)")
        log.append({"epoch":epoch,"train_acc":round(ta,4),"val_acc":round(va,4),"train_loss":round(tl,4),"val_loss":round(vl,4)})
        if va > best_acc:
            best_acc = va; save_model(model)
            print(f"  ✓ Best model saved (val_acc={best_acc:.3f})")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH,"w") as f: json.dump(log, f, indent=2)
    print(f"\n{'='*50}\n Best val accuracy: {best_acc:.3f}\n Model → models/satnet.pth\n{'='*50}\n")
    return log

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=32)
    a = p.parse_args()
    train(a.epochs, a.lr, a.batch_size)
