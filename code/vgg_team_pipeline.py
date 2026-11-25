"""VGG trained on the same Task 2 image pipeline and patient folds as teammates.

Shared sources: code/dataset.py and split.csv from image-pipeline branch.
This script trains ONE VGG model at a time. It does not ensemble outputs.

Safe baseline (no external weights):
    python code/vgg_team_pipeline.py --root data/T2_data --strategy scratch --device auto

Optional pretraining experiments ONLY IF course staff explicitly permit weights
learned on ImageNet (the written brief also says train only on supplied data):
    python code/vgg_team_pipeline.py --root data/T2_data --strategy partial --allow-imagenet-weights

Inference:
    python code/vgg_team_pipeline.py --mode predict --root data/T2_data --csv data/T2_data/test.csv --checkpoint runs/vgg_partial/vgg.pt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def as_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(arch='vgg11', strategy='scratch', allow_imagenet_weights=False, dropout=0.3):
    """Single torchvision VGG with a compact binary head."""
    variants = {
        'vgg11': (models.vgg11, models.VGG11_Weights),
        'vgg11_bn': (models.vgg11_bn, models.VGG11_BN_Weights),
        'vgg16': (models.vgg16, models.VGG16_Weights),
    }
    if arch not in variants:
        raise ValueError(f'Unsupported VGG architecture: {arch}')
    if strategy not in {'scratch', 'head', 'partial', 'full'}:
        raise ValueError(f'Unsupported training strategy: {strategy}')
    if strategy != 'scratch' and not allow_imagenet_weights:
        raise ValueError('ImageNet weights require --allow-imagenet-weights AND permission under course rules.')

    builder, weight_enum = variants[arch]
    weights = weight_enum.IMAGENET1K_V1 if strategy != 'scratch' else None
    model = builder(weights=weights)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, 2),
    )

    if strategy in {'head', 'partial'}:
        for p in model.features.parameters():
            p.requires_grad_(False)

    if strategy == 'partial':
        # Unfreeze final two VGG convolutional blocks.
        pools = [i for i, layer in enumerate(model.features) if isinstance(layer, nn.MaxPool2d)]
        if len(pools) != 5:
            raise RuntimeError('Expected 5 VGG pooling stages.')
        start = pools[2] + 1
        for layer in list(model.features)[start:]:
            for p in layer.parameters():
                p.requires_grad_(True)
    return model


def safe_shared_data_check(root: Path, split_csv: Path, valid_fold: int):
    labels_path = root / 'labels.csv'
    if not labels_path.is_file():
        raise FileNotFoundError(
            f'Missing {labels_path}. The GitHub branch contains code + split.csv, '
            'not the original T2_data/labels.csv or ultrasound PNGs. '
            'Extract the course T2_data.zip into data/T2_data first.'
        )
    if not split_csv.is_file():
        raise FileNotFoundError(f'Missing shared split.csv: {split_csv}')
    labels = pd.read_csv(labels_path, dtype={'sample_id': str, 'patient_group': str})
    split = pd.read_csv(split_csv, dtype={'sample_id': str, 'patient_group': str})
    labels_cols = {'sample_id', 'path', 'label', 'patient_group'}
    split_cols = {'sample_id', 'patient_group', 'fold'}
    if not labels_cols.issubset(labels.columns) or not split_cols.issubset(split.columns):
        raise ValueError('Required labels/split columns are missing.')
    if labels.sample_id.duplicated().any() or split.sample_id.duplicated().any():
        raise ValueError('Duplicate sample_id found in labels or split.')
    if set(labels.sample_id) != set(split.sample_id):
        raise ValueError('labels.csv and split.csv do not cover the same sample_ids.')
    joined = labels.merge(split, on='sample_id', suffixes=('_labels', '_split'), validate='one_to_one')
    if (joined.patient_group_labels != joined.patient_group_split).any():
        raise ValueError('Patient-group mismatch between labels.csv and split.csv.')
    if joined.groupby('patient_group_labels').fold.nunique().gt(1).any():
        raise ValueError('Patient leakage: a patient belongs to more than one fold.')
    if valid_fold not in joined.fold.unique():
        raise ValueError(f'Validation fold {valid_fold} is not in split.csv.')
    train = joined.loc[joined.fold != valid_fold]
    valid = joined.loc[joined.fold == valid_fold]
    if train.label.nunique() < 2 or valid.label.nunique() < 2:
        raise ValueError('Both train and validation need the two label classes.')
    split_sha = hashlib.sha256(split_csv.read_bytes()).hexdigest()
    stats = {
        'rows': len(joined), 'patients': int(joined.patient_group_labels.nunique()),
        'train_images': len(train), 'valid_images': len(valid),
        'train_patients': int(train.patient_group_labels.nunique()),
        'valid_patients': int(valid.patient_group_labels.nunique()),
        'valid_fold': int(valid_fold), 'split_sha256': split_sha,
        'train_label_counts': {str(k): int(v) for k,v in train.label.value_counts().to_dict().items()},
        'valid_label_counts': {str(k): int(v) for k,v in valid.label.value_counts().to_dict().items()},
    }
    return stats


def get_loaders_shared(root, split_csv, valid_fold, batch_size, num_workers, img_h, img_w, aug):
    # dataset.py reads env at module import time; configure it before importing.
    os.environ['IMG_H'] = str(img_h)
    os.environ['IMG_W'] = str(img_w)
    os.environ['AUG'] = aug
    os.environ['FULL'] = '0'  # crucial: never train on validation while evaluating
    import dataset
    dataset.SIZE = (img_h, img_w)
    dataset.AUG = aug
    return dataset.get_loaders(
        str(root), str(split_csv), valid_fold=valid_fold,
        batch_size=batch_size, num_workers=num_workers,
    )


def collect_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.inference_mode():
        for images, y in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
            probs.append(probabilities.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def get_auc(labels, probabilities):
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float('nan')


def metrics(labels, probabilities, threshold):
    pred = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0,1]).ravel()
    return {
        'threshold': float(threshold), 'ba': float(balanced_accuracy_score(labels, pred)),
        'auc': get_auc(labels, probabilities),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def best_threshold_descriptive(p, y):
    from train_utils import best_threshold
    return best_threshold(p, y)


def freeze_batchnorm_as_needed(model, strategy, freeze_bn):
    # Calling model.train() also toggles BN training even if its weights are frozen.
    if strategy == 'head':
        model.features.eval()
    elif strategy == 'partial':
        for module in model.features.modules():
            if isinstance(module, nn.BatchNorm2d) and not any(
                p.requires_grad for p in module.parameters()
            ):
                module.eval()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def append_csv(filename, row):
    exists = filename.exists()
    with filename.open('a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_checkpoint(filename, payload):
    tmp = filename.with_suffix(filename.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(filename)


def train(args):
    root = as_path(args.root)
    split_csv = as_path(args.split_csv)
    run_dir = as_path(args.output_dir or f'runs/vgg_{args.strategy}')
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available; use --device cpu or auto.')
    device = torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    seed_everything(args.seed)
    stats = safe_shared_data_check(root, split_csv, args.valid_fold)
    print('=== SHARED TASK2 DATA ===', json.dumps(stats, indent=2), flush=True)
    if device.type != 'cuda':
        print('WARNING: CPU VGG training at this image size may be very slow.', flush=True)

    train_loader, valid_loader, train_labels = get_loaders_shared(
        root, split_csv, args.valid_fold, args.batch_size,
        args.workers, args.img_h, args.img_w, args.aug,
    )

    model = build_model(args.arch, args.strategy, args.allow_imagenet_weights, args.dropout).to(device)
    model_cfg = {
        'architecture': args.arch,
        'strategy': args.strategy,
        'pretrained_imagenet': bool(args.strategy!='scratch'),
        'dropout': args.dropout,
        'img_h': args.img_h, 'img_w': args.img_w,
        'augmentation': args.aug,
        'valid_fold': args.valid_fold,
        'imagenet_mean': list([0.485,0.456,0.406]),
        'imagenet_std': list([0.229,0.224,0.225]),
    }
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {model_cfg} | trainable_params={trainable:,} | device={device}', flush=True)
    labels_count = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    weights = torch.tensor(labels_count.sum() / (2.0*np.maximum(1,labels_count)), dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    lr = args.lr if args.lr is not None else (3e-4 if args.strategy=='scratch' else 1e-4)
    optimizer = torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,args.epochs))

    best = -1.0
    best_epoch = 0
    wait = 0
    start_epoch=1
    best_path=run_dir/'vgg.pt'
    last_path=run_dir/'last_training_state.pt'
    metrics_path=run_dir/'epoch_metrics.csv'
    if args.resume and last_path.is_file():
        state=torch.load(last_path,map_location=device,weights_only=False)
        if state['config']!=model_cfg or state['split_sha256']!=stats['split_sha256']:
            raise ValueError('Resume state config/split mismatch. Use a new output directory.')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        best=float(state['best_score'])
        best_epoch=int(state['best_epoch'])
        wait=int(state['wait'])
        start_epoch=int(state['epoch'])+1
        print(f'Resumed from epoch {start_epoch}',flush=True)

    for epoch in range(start_epoch,args.epochs+1):
        model.train()
        freeze_batchnorm_as_needed(model,args.strategy,args.freeze_bn)
        total_loss=0.0
        seen=0
        for images,y in train_loader:
            images=images.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            loss=criterion(logits,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            total_loss+=loss.item()*len(y)
            seen+=len(y)
        scheduler.step()
        probs,ytrue=collect_probs(model,valid_loader,device)
        m=metrics(ytrue,probs,0.5)
        # Keep checkpoint selection aligned with the team's fixed BA@0.5 rule.
        improved=m['ba'] > best+1e-6
        if improved:
            best=m['ba'];best_epoch=epoch;wait=0
            save_checkpoint(best_path,{
                'model_state_dict':{k:v.cpu() for k,v in model.state_dict().items()},
                'model_config':model_cfg,
                'split_sha256':stats['split_sha256'],
                'best_epoch':int(epoch),
                'validation_ba_at_05':float(best),
                'threshold':0.5,
                'threshold_selected_on_validation':False,
            })
        else:
            wait+=1
        line={'epoch':epoch,'train_loss':total_loss/max(1,seen),'valid_ba_05':m['ba'],
              'valid_auc':m['auc'],'lr':optimizer.param_groups[0]['lr'],
              'best_epoch':best_epoch,'best_ba_05':best,'no_improvement':wait}
        append_csv(metrics_path,line)
        print('Epoch {}/{} | loss {:.5f} | BA@0.5 {:.5f} | AUC {:.5f} | best {:.5f}@{}'.format(
            epoch,args.epochs,line['train_loss'],m['ba'],m['auc'],best,best_epoch),flush=True)
        save_checkpoint(last_path,{
            'epoch':epoch,'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'best_score':best,'best_epoch':best_epoch,'wait':wait,
            'config':model_cfg,'split_sha256':stats['split_sha256'],
        })
        if wait>=args.patience:
            print('Early stopping.',flush=True)
            break

    if not best_path.exists():
        raise RuntimeError('No best model saved. Check training settings.')
    ckpt=torch.load(best_path,map_location='cpu',weights_only=True)
    check=build_model(ckpt['model_config']['architecture'],'scratch',False,
                      ckpt['model_config']['dropout'])
    check.load_state_dict(ckpt['model_state_dict'],strict=True)
    model.load_state_dict(ckpt['model_state_dict'],strict=True)
    probs,ytrue=collect_probs(model,valid_loader,device)
    threshold,tuned=best_threshold_descriptive(probs,ytrue)
    report={
        'model_config':model_cfg,'split_summary':stats,'best_epoch':best_epoch,
        'ba_at_05':metrics(ytrue,probs,0.5),'threshold_selected_on_validation':float(threshold),
        'ba_at_selected_threshold_descriptive_only':float(tuned),
        'note':'The selected threshold uses validation labels and is optimistic; primary checkpoint selected with BA at 0.5. This is not private-test performance.',
    }
    (run_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('=== FINAL ===',json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('Saved:',best_path,'\nCheckpoint reload test: OK',flush=True)


def predict(args):
    path=as_path(args.checkpoint)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    config=checkpoint['model_config']
    model=build_model(config['architecture'],'scratch',False,config['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()
    from PIL import Image
    from torchvision import transforms as T
    tf=T.Compose([
        T.Resize((config['img_h'],config['img_w'])),
        T.ToTensor(),
        T.Normalize(config['imagenet_mean'],config['imagenet_std']),
    ])
    data=pd.read_csv(as_path(args.csv))
    if 'path' not in data.columns:
        raise ValueError('Inference CSV requires a path column.')
    root=as_path(args.root)
    probs=[]
    with torch.inference_mode():
        for filename in data.path.astype(str):
            image_path=root/filename
            with Image.open(image_path) as img:
                x=tf(img.convert('RGB')).unsqueeze(0)
            probs.append(float(torch.softmax(model(x),dim=1)[0,1]))
    result=pd.DataFrame({'probability_present':probs,'predicted_label':(np.asarray(probs)>=checkpoint.get('threshold',0.5)).astype(int)})
    if 'sample_id' in data.columns:
        result.insert(0,'sample_id',data.sample_id)
    out=as_path(args.output_csv)
    out.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(out,index=False)
    print('Predictions saved:',out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['train','predict'],default='train')
    parser.add_argument('--root',default='data/T2_data')
    parser.add_argument('--split-csv',default='split.csv')
    parser.add_argument('--valid-fold',type=int,default=1)
    parser.add_argument('--arch',choices=['vgg11','vgg11_bn','vgg16'],default='vgg11')
    parser.add_argument('--strategy',choices=['scratch','head','partial','full'],default='scratch')
    parser.add_argument('--allow-imagenet-weights',action='store_true')
    parser.add_argument('--aug',choices=['base','strong'],default='base')
    parser.add_argument('--img-h',type=int,default=384)
    parser.add_argument('--img-w',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--lr',type=float,default=None)
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--dropout',type=float,default=0.3)
    parser.add_argument('--freeze-bn',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--output-dir',default=None)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',default='runs/vgg_scratch/vgg.pt')
    parser.add_argument('--csv',default=None)
    parser.add_argument('--output-csv',default='runs/vgg_predictions.csv')
    args=parser.parse_args()
    if args.mode=='train':
        if args.epochs<1 or args.patience<1 or args.batch_size<1:
            parser.error('epochs, patience and batch-size must be >=1')
        if args.img_h<32 or args.img_w<32:
            parser.error('VGG requires image dimensions >=32')
        train(args)
    else:
        if args.csv is None:
            parser.error('--csv required for predict')
        predict(args)

if __name__=='__main__':
    main()
