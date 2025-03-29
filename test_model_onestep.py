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
TYPE="Light"
data_path_test=f"./dataset/json_test/test.json"
MODEL_PATH=f"./output_/output_model/{TYPE}PowerModel.ckpt"
BATCH_SIZE=32
TIME_STEP=1

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
    model=WindPowerModel(site=5).to(DEVICE)
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
            label_wind,label_light,timestamp,input_data= tuple(t.to(DEVICE) for t in batch)
            if TYPE=='Wind':
                input_data=input_data[:,:,:40]
            elif TYPE=="Light":
                input_data=input_data[:,:,40:]
            step_predict = model(input_data)  
            prediction.append(step_predict)

    final_predictions = torch.cat(prediction, dim=0)  # (总Batch数 * B, 预测步数, 5)
    final_predictions_np = final_predictions.cpu().numpy()  # 转换为 NumPy 数组，确保使用 cpu()
    final_predictions_flattened = final_predictions_np.reshape(-1, 5) #5663,5
    output0 = final_predictions_flattened[:, 0]  # 获取第一列
    output1 = final_predictions_flattened[:, 1]  # 获取第二列
    output2 = final_predictions_flattened[:, 2]  # 获取第三列
    output3 = final_predictions_flattened[:, 3]  # 获取第四列
    output4 = final_predictions_flattened[:, 4]  # 获取第五列
    
    start_time = pd.to_datetime("2025-01-01 00:00:00")
    time_index = pd.date_range(start=start_time, periods=len(output0), freq='15T')
    
    df_output0 = pd.DataFrame({'': time_index, 'power': output0})
    df_output1 = pd.DataFrame({'': time_index, 'power': output1})
    df_output2 = pd.DataFrame({'': time_index, 'power': output2})
    df_output3 = pd.DataFrame({'': time_index, 'power': output3})
    df_output4 = pd.DataFrame({'': time_index, 'power': output4})
    if TYPE=="Wind":
        df_output0.to_csv('./output/output1.csv', index=False)
        df_output1.to_csv('./output/output2.csv', index=False)
        df_output2.to_csv('./output/output3.csv', index=False)
        df_output3.to_csv('./output/output4.csv', index=False)
        df_output4.to_csv('./output/output5.csv', index=False)
    elif TYPE=="Light":
        df_output0.to_csv('./output/output6.csv', index=False)
        df_output1.to_csv('./output/output7.csv', index=False)
        df_output2.to_csv('./output/output8.csv', index=False)
        df_output3.to_csv('./output/output9.csv', index=False)
        df_output4.to_csv('./output/output10.csv', index=False)

    
if __name__=="__main__":
    main()