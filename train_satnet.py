"""Quick training wrapper. Run: python train_satnet.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.train import train

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=32)
    a = p.parse_args()
    train(a.epochs, a.lr, a.batch_size)
