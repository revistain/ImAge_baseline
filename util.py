import os
import torch
import shutil
import logging
import numpy as np
from collections import OrderedDict
from datetime import datetime
from os.path import join

def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))

def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # if the model contains the prefix "module" which is appendend by
    # DataParallel, remove it to avoid errors when loading dict
    if list(state_dict.keys())[0].startswith('module'):
        state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in state_dict.items()})
    model.load_state_dict(state_dict)
    return model

def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, weights_only=False)
    start_epoch_num = checkpoint["epoch_num"]+1
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r1_r5 = checkpoint["best_r1_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current best (R@1 + R@5) = {best_r1_r5:.1f}")
    if args.resume.endswith("best_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("best_model.pth", "best_model.pth"), args.save_dir)
    return model, optimizer, best_r1_r5, start_epoch_num, not_improved_num

def Celsius2Raw(celcius_degree):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    raw_value = R / (np.exp(B / (celcius_degree + 273.15)) - F) + O
    return raw_value

# Raw thermal radiation value to tempearture 
def Raw2Celsius(Raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    Celsius = B / np.log(R / (Raw - O) + F) - 273.15
    return Celsius

def save_files(path):
    import subprocess
    models_dir = os.path.join(path, "save_codes")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    file_paths = ['model','Dataset','scripts','commons.py','datasets_ws.py','eval.py',
                  'loss.py',' inference.py',' parser.py',
                  'train.py',' util.py']

    for file_path in file_paths:
        if os.path.exists(file_path):
            if not os.path.exists(os.path.join(models_dir, file_path)):
                subprocess.run(['cp', '-r', file_path, models_dir])
            else:
                print("File already exists", file_path)
                
cached_timestamp = None
def get_timestamp():
    global cached_timestamp
    if cached_timestamp is None:
        cached_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    return cached_timestamp

def get_loss_warmup_scale(current_epoch, warmup_epochs, max_scale=5e-2):
    """
    current_epoch가 warmup_epochs에 도달할 때까지 
    0.0에서 max_scale로 선형 증가합니다.
    """
    # 워밍업을 안 쓰는 경우 바로 최대값 반환
    if warmup_epochs == 0:
        return max_scale
    
    # 0.0 ~ 1.0 사이의 진행률(Ratio) 계산
    ratio = float(current_epoch) / float(warmup_epochs)
    ratio = min(1.0, max(0.0, ratio))
    
    # 진행률에 최대값을 곱해서 최종 가중치 스케일 반환
    return ratio * max_scale