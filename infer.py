"""Reconstruct observations from a saved checkpoint; no reference required."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import torch
from nullblend.config import NullBlendConfig
from nullblend.data import load_observed_dataset
from nullblend.hnsln import HierarchicalNullSpaceLearningNetwork
from nullblend.operators import overlap_counts, minimum_norm_solution, shot_overlap_order, build_tafa_index
from nullblend.training import _reconstruct

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    checkpoint=args.checkpoint.resolve()
    run_dir=checkpoint.parent.parent if checkpoint.parent.name=='checkpoints' else checkpoint.parent
    stored=json.loads((args.config or run_dir/'config.json').read_text(encoding='utf-8'))
    values={f.name:stored[f.name] for f in fields(NullBlendConfig) if f.name in stored}
    values['device']=args.device
    values['shot_shifts']=tuple(values['shot_shifts'])
    cfg=NullBlendConfig(**values)
    bundle=load_observed_dataset(cfg.dataset_dir)
    observed=bundle['observed']
    starts_cpu=bundle['shot_starts']
    nt=int(bundle['metadata']['nt']); length=observed.shape[-1]
    device=torch.device(args.device)
    counts_cpu=overlap_counts(starts_cpu,nt,length)
    particular=minimum_norm_solution(observed,starts_cpu,nt,counts_cpu)
    index,slots=build_tafa_index(starts_cpu,nt,length,cfg.maximum_overlap)
    model=HierarchicalNullSpaceLearningNetwork().to(device)
    model.set_tafa_index(index.to(device),slots.to(device),length)
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(state['hnsln'] if 'hnsln' in state else state)
    prediction=_reconstruct(model=model,particular_cpu=particular,observed_cpu=observed,
        overlap_order=shot_overlap_order(starts_cpu,nt,length).float().to(device),
        shot_starts=starts_cpu.to(device),counts=counts_cpu.to(device),device=device,
        receiver_batch_size=cfg.receiver_batch_size,receiver_radius=cfg.receiver_radius)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(prediction,args.output)
    print(f'Saved student reconstruction: {args.output}',flush=True)

if __name__=='__main__':
    main()
