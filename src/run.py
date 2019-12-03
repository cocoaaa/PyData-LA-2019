import os,sys, time
from pathlib import Path
import numpy as np
import random
from skimage import io as skiio
import PIL

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pdb 
# from tqdm.notebook import tnrange, tqdm
from tqdm.autonotebook import tqdm

# My libs
from helpers import load_txt, now2str, append2file
from model.helpers import save_checkpt

from metrics import runningScore, averageMeter
from train import train
from evaluate import evaluate

def run(model, dataloaders, loss_fn, optimizer, lr_scheduler, device, params, memo):
    """
    Runs train and validation every epoch while updating `model`'s parameters using the 
    `loss_fn` and `optimizers`. `optimizer`'s learning rates are updated during the training via `lr_scheduler`.
    
    Before calling this function, put the model in `device`, and *then* set this model's parameters as optimizer's target variables:
    
    ```python
    # Example
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=dummy_lr) # dummy_lr because it will always be 
                                        # overwritten by lr_scheduler's returned value
    dataloaders = {'train': train_dl,
                    'val': val_dl}
    loss_fn = nn.CrossEntropy(...)
    lr_scheduler = TriangleLR(...)
    params = {'max_epoch': , # default: 50, 
              'batch_size': , # default: 8 , 
              'n_classes' : 21, #default: 21
              'fill_value': , # default: 255 , 
              'save_every': , # default: 10 , 
              'print_every': , # default: 5,
              'ignore_idx': 255, # default: 255
              'ignore_bg': True # default: True
              'debug': False # default: False
              }
    
    result = run(model, dataloaders, loss_fn, optimizer, lr_scheduler, device, params)
    ```
    
    Assumes
    - model is already sent to the `device`
    - optimizer has been constructed with model.parameters() as input
    - `device` will be used to send the data (returned from dataloaders) to the `device`
        that is the same device where the input `model` is located
    """
    
    # Get parameters
    max_epoch = params.get('max_epoch', 50)
    batch_size = params.get('batch_size', 8)
    best_iou = params.get('best_iou', -100)
    fill_value = params.get('fill_value', 255)
    ignore_gt = params.get('ignore_gt', 0) # None or, one of the class labels to be removed from acc computation
    save_every = params.get('save_every', 10) # unit of epoch
    print_every = params.get('print_every', 5) # unit of epoch
    log_fn = params.get('log_fn', f'/data/rl/log/{model.name}-fill-{fill_value}-bs-{batch_size}/{now2str()}/log.txt')
    
    # Get dataloaders
    train_dl, val_dl = dataloaders['train'], dataloaders['val']
    assert batch_size == train_dl.batch_size, "batch_size must be the batchsize of train_dl"
    
    # Set log file directory if not existing yet
    if not isinstance(log_fn, Path):
        log_fn = Path(log_fn)
    if not log_fn.parent.exists():
        log_fn.parent.mkdir(parents=True)
        print('Created : ', str(log_fn.parent))
    print('log file: ', log_fn)
    
    # Start experiment
    all_train_losses = [] # from each iteration
    all_train_accs = []
    ep_train_losses = [] # averge losses from each epoch
    ep_train_accs = []
    
    val_losses = [] # from each epoch
    val_accs = []
    val_ious = []
    #best_iou is set above where the parameters are read from the  input `params` dict
    exp_start = time.time()
    for epoch in tqdm(range(max_epoch), desc='Epoch'):
        # Train
        start = time.time()
        train_result = train(model, train_dl, loss_fn, optimizer, lr_scheduler, device, params, memo)
        end = time.time()
        
        # Collect train metrics
        all_train_losses.extend(train_result['train_losses'])
        all_train_accs.extend(train_result['train_accs'])
        ep_train_losses.append(train_result['loss_meter'].avg)
        ep_train_accs.append(train_result['acc_meter'].avg)
        
            # Logging after each train epoch
        append2file(f"{'='*80}"
                    f'\nEpoch: {epoch}/{max_epoch}'
                    f'\n\tTrain took: {(end-start)/60.0:.3f} mins'
                    f'\n\tTrain loss: {ep_train_losses[-1]:9.5f}'
                    f'\n\tTrain acc: {ep_train_accs[-1]:9.5f}%', log_fn)
        ## train_result = {'running_metrics': running_metrics,
#               'loss_meter': loss_meter,
#               'acc_meter': acc_meter,
#              'train_losses': train_losses,
#              'train_accs': train_accs}
        
        
        # Evaluate
        val_result = evaluate(model, val_dl, loss_fn, device, params)
        
        # Collect validation metrics
        val_losses.append(val_result['loss_meter'].avg)
        val_accs.append(val_result['acc_meter'].avg)
                          
        score, class_iou = val_result['running_metrics'].get_scores(ignore_gt=ignore_gt)
        val_ious.append(score["Mean IoU : \t"])
        
        # Log evaluation for this epoch
        mean_acc = score['Mean Acc : \t']
        append2file(
            f"\n\tVal loss: {val_losses[-1]:9.5f}"
            f"\n\tVal acc: {val_accs[-1]:9.5f}%"
            f"\n\t\t vs. {mean_acc}"
            f"\n\tVal mean IOU: {val_ious[-1]:9.5f}", log_fn)
        
        for k, v in score.items():
#             print(k, v) ##printing here
            append2file(f'\tVal_metric/{k} -- {v:9.5f}', log_fn)

        for k, v in class_iou.items():
            append2file(f'\tVal Class IOU/{k} -- {v:9.5f}', log_fn)

        # Save current state if it achieves the best IOU on validation set
        if best_iou < 0 or (score["Mean IoU : \t"] - best_iou)/best_iou > 0.05:
            best_iou = score["Mean IoU : \t"]
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_iou": best_iou,
            }
            torch.save(state, log_fn.parent/'best_state.pth')
            print(f"Updated a new best state. ep: {state['epoch']}, iou: {state['best_iou']}")
            for k, v in score.items():
                print(k, v) ##printing here
            
        # Save the current model if it's time to save intermittently
        if (epoch+1)%save_every==0:
            print("="*80)
            print('Epoch: ', epoch, ' Saved model')
            out_fn = log_fn.parent/f'{model.name}_{epoch}.pth'
            save_checkpt(epoch, model, optimizer, ep_train_losses[-1], out_fn)
                ##     result = {'running_metrics': running_metrics,
#               'loss_meter': loss_meter,
#               'acc_meter': acc_meter}
        
    # Log this experiment's train and val losses
    out_fn = log_fn.parent/(log_fn.stem+'_losses.npz')
    np.savez_compressed(out_fn, 
                        train_losses=all_train_losses, train_accs=all_train_accs, 
                        val_losses=val_losses, val_accs=val_accs)
    print('Saved the losses to...: ', out_fn)
    print(f'Experiement took : {(time.time() - exp_start)/60:.3f} mins')

    result =  {
        'train': {'loss': ep_train_losses, 
                  'acc': ep_train_accs},
        'val': {'loss': val_losses, 
                'acc': val_accs}
    }

    return model, result                            
        
                        
                        
