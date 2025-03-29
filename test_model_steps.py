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
data_path_test=f"./dataset/json_test/test.json"
MODEL_PATH="./output/output_model/WindPowerModel_last.ckpt"
BATCH_SIZE=1
TIME_STEP=24

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
        test_iterator = tqdm.tqdm(dataloader_test, initial=0, desc="Iter", disable=False)
        total_steps = dataloader_test.dataset.total_num  # 目标预测总时间步
        all_predictions = []  # 存储整个 batch 的预测结果
        prediction=[]
        power_queue = deque(maxlen=TIME_STEP)
        for step, batch in enumerate(test_iterator):
            _, input_data = tuple(t.to(DEVICE) for t in batch)
            indices = [0, 9, 18, 27, 36]
            if step==0:
                power_data = input_data[:, :, indices]
                for t in range(TIME_STEP):
                    power_queue.append(power_data[:, t, :])  # 每次添加形状 [B, 1, 5]
            else:
                weather_indices = []
                updated_power = torch.cat(list(power_queue), dim=0).unsqueeze(0)#形状 [B, T, 5]
                for site_start in indices:
                    weather_indices.extend(range(site_start + 1, site_start + 9))
                weather_data = input_data[:, :, weather_indices]  # 形状 [B, T, 40]
                weather_reshaped = weather_data.view(1, TIME_STEP, 5, 8)  # 5站点 × 8特征
                combined_per_site = torch.cat([updated_power.unsqueeze(-1), weather_reshaped], dim=-1)
                input_data = combined_per_site.reshape(1, TIME_STEP, 5 * 9)  # 5×9=45

            """"""
            step_predict = model(input_data)  # 预测下一时间步 (B, 5)
            prediction.append(step_predict)
            power_queue.append(step_predict)

    final_predictions = torch.cat(prediction, dim=0)  # (总Batch数 * B, 预测步数, 5)
    final_predictions_np = final_predictions.cpu().numpy()  # 转换为 NumPy 数组，确保使用 cpu()
    final_predictions_flattened = final_predictions_np.reshape(-1, 5)
    df = pd.DataFrame(final_predictions_flattened, columns=[f'power_{i+1}' for i in range(5)])
    df.to_csv('predictions_wind.csv', index=False)
    print(final_predictions.shape)  # 确保输出符合预期
    
if __name__=="__main__":
    main()