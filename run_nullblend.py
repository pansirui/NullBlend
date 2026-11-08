"""Train NullBlend with the published configuration."""
import argparse
from pathlib import Path
import torch
from nullblend.config import NullBlendConfig
from nullblend.training import train_nullblend

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('mobil','seam','marmousi'),required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    train_nullblend(NullBlendConfig(dataset=args.dataset,device=args.device,epochs=args.epochs),args.output_root)

if __name__=='__main__':
    main()
