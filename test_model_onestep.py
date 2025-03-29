import os
import sys
import torch
import tqdm 
import numpy as np
import torch.nn as nn
from collections import deque
import pandas as pd
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='0' #'0,1'GPU NUM
from model.model import WindPowerModel
from torch.utils.data import (DataLoader)
from datetime import datetime

from model.dataset import  CacheDataset,OnlineCacheDataset,PreprocessCacheData
from model.utils import get_linear_schedule_with_warmup,PrintModelInfo,save_ckpt,CaculateAcc
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE="Light" #Wind Light
data_path_test=f"./dataset/json_test/test.json"
MODEL_PATH=f"./output_/output_model/{TYPE}PowerModel.ckpt"
BATCH_SIZE=32
TIME_STEP=1
SITE=10

def CreateDataloader(data_path):
    dataset = OnlineCacheDataset(data_path,shuffle=True)
    num_work = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    loader = DataLoader(dataset=dataset, 
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        pin_memory=True,
                        num_workers=num_work,
                        collate_fn=dataset.collate_fn)
    return loader

def main():
    model=WindPowerModel().to(DEVICE)
    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt['model'])
    PrintModelInfo(model)
    print("model load weight done.")
    dataloader_test=CreateDataloader(data_path_test)
    model.eval()
    with torch.no_grad():
        prediction=[]
        test_iterator = tqdm.tqdm(dataloader_test, initial=0, desc="Iter", disable=False)
        for step, batch in enumerate(test_iterator):
            label,station,month,day,hour,minute,input_data= tuple(t.to(DEVICE) for t in batch)

            step_predict = model(station,month,day,hour,minute,input_data)  
            prediction.append(step_predict)

    final_predictions = torch.cat(prediction, dim=0)  # (总Batch数 * B, 预测步数, 5)
    final_predictions_np = final_predictions.cpu().numpy()  # 转换为 NumPy 数组，确保使用 cpu()
    final_predictions_flattened = final_predictions_np.reshape(-1, SITE) #5663,5
    for i in range(SITE):
        output=final_predictions_flattened[:, i]
        start_time = pd.to_datetime("2025-01-01 00:00:00")
        time_index = pd.date_range(start=start_time, periods=len(output), freq='15T')
        df_output = pd.DataFrame({'': time_index, 'power': output})  
        df_output.to_csv(f'./output/output{i+1}.csv', index=False)
        

    
if __name__=="__main__":
    main()