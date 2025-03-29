import os
import sys
import torch
import tqdm 
import numpy as np
import torch.nn as nn

from test_model_steps import TIME_STEP
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='0' #'0,1'GPU NUM
from model.model import WindPowerModel
from torch.utils.data import (DataLoader)
from datetime import datetime

from model.dataset import  CacheDataset,OnlineCacheDataset,PreprocessCacheData
from model.utils import get_linear_schedule_with_warmup,PrintModelInfo,save_ckpt,CaculateAcc
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
    
LR=0.01
EPOCH=100
BATCH_SIZE=128
CACHE=False
TENSORBOARDSTEP=500
TF_ENABLE_ONEDNN_OPTS=0
TYPE="Light" #Wind Light
SAVE_NAME=f"{TYPE}PowerModel"
SAVE_PATH='./output_/output_model/'
PRETRAINED_MODEL_PATH = SAVE_PATH,SAVE_NAME+'.ckpt'
Pretrain=False if PRETRAINED_MODEL_PATH ==" " else True
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""dataset"""
train_type="train"
data_path_train=f"./dataset/json_train/train.json"
cached_file=f"./dataset/cache/{train_type}.pt"
val_type="val"
data_path_val=f"./dataset/json_train/test.json"
cached_file_val=f"./dataset/cache/{val_type}.pt"
TIME_STEP=1

def CreateDataloader(data_path):
    dataset = OnlineCacheDataset(data_path,shuffle=True)
    num_work = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    loader = DataLoader(dataset=dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True,
                        pin_memory=True,
                        num_workers=num_work,
                        collate_fn=dataset.collate_fn)
    return loader

def quantile_loss(y_pred, y_true, quantile=0.5):
    errors = y_true - y_pred
    return torch.mean(torch.max(quantile * errors, (quantile - 1) * errors))

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                    
def main():
    global_step=0
    """Define Model"""
    model=WindPowerModel(site=1).to(DEVICE)
    model.apply(weights_init)
    model_name=model.__class__.__name__
    PrintModelInfo(model)
    # """Create dataloader"""
    dataloader_train=CreateDataloader(data_path_train)
    total_steps = len(dataloader_train) * EPOCH
    """Loss function"""
    criterion = nn.MSELoss()
    """Optimizer"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    """ Train! """
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * total_steps , total_steps)
    tb_writer = SummaryWriter(log_dir='./output_/tflog/') 
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.dataset.total_num)
    print(f"  Pretrained Model is ")
    print(f"  Save Model as {SAVE_PATH}")
    print("  ****************************************************************")
    start_time=datetime.now()
    best_loss=10
    for epoch_index in range(EPOCH):
        loss_sum=0
        sum_test_accuarcy=0
        model.train()
        torch.cuda.empty_cache()
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, batch in enumerate(train_iterator):
            label,station,month,day,hour,minute,input_data= tuple(t.to(DEVICE) for t in batch)
            optimizer.zero_grad()
            output=model(station,month,day,hour,minute,input_data)  #input[b,1,8*site]---[b,site]
            #accuarcy=CaculateAcc(output,label)
            accuarcy=0
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss_sum=loss_sum+loss.item()
            sum_test_accuarcy=sum_test_accuarcy+accuarcy
            current_lr= scheduler.get_last_lr()[0]
            """ tensorbooard """
            if  global_step % TENSORBOARDSTEP== 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', current_lr, global_step=global_step)
                tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            global_step=global_step+1
            scheduler.step()
            train_iterator.set_description('Epoch=%d, Acc= %3.3f %%,loss=%.6f, lr=%9.7f' 
                                           % (epoch_index,(sum_test_accuarcy/(step+1))*100, loss_sum/(step+1), current_lr))
        """ validation """
        """save model"""
        if loss_sum/(step+1) < best_loss:
            best_loss = loss_sum/(step+1)
            save_ckpt(SAVE_PATH,SAVE_NAME+'.ckpt',model,epoch_index,scheduler,optimizer)
        else:
            save_ckpt(SAVE_PATH,SAVE_NAME+'_last.ckpt',model,epoch_index,scheduler,optimizer) 
    end_time=datetime.now()
    print("Training consume :",(end_time-start_time)/60,"minutes")
    
if __name__=="__main__":
    main()