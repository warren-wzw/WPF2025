import xarray as xr
import pandas as pd
from datetime import datetime
import os

"""
NMP_1: * channel  (channel) object 'ghi' 'poai' 'sp' 't2m' 'tcc' 'tp' 'u100' 'v100'
NMP_2: * channel  (channel) object 'ghi' 'msl' 'poai' 't2m' 'tcc' 'tp' 'u100' 'v100'
NMP_3: * channel  (channel) object 'ghi' 'poai' 'sp' 't2m' 'tcc' 'tp' 'u100' 'v100'
"""


class NetCDF_Translation(object):
    def __init__(self, file_path1="", file_path2="", folder_path="", channel_choose="NWP_1", save_path=""):
        self.file_path1 = file_path1  # 文件路径
        self.file_path2 = file_path2  # 文件路径
        self.folder_path = folder_path  # 文件夹的路径
        self.channel_choose = channel_choose  # 通道的选择，判断是NWP_1、3还是NWP_2
        self.save_path = save_path

    def showNCdata(self):
        ds = xr.open_dataset(self.file_path1)  # 此处的path应该为.nc文件路径
        print(ds)
        print("\n数据变量维度：")
        print(ds['data'].dims)

    # 将单个相应位置的nc文件按要求处理后保存
    def translate_NetCDF_toCSV_single(self):
        ds = xr.open_dataset(self.file_path1)  # 此处的path应该为.nc文件路径
        print(ds)
        print("\n数据变量维度：")
        print(ds['data'].dims)
        # 定义目标位置（Matlab 的 6x6 → Python 的 5x5）
        lon_idx = 5
        lat_idx = 5
        # 维度顺序： time × hour × channel × lat × lon
        target_data = ds['data'][0, :, :, lat_idx, lon_idx].values  # shape: (24, 11)

        # 创建列名（根据 channel 名称）
        channel_names = ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100']

        # 创建 DataFrame
        df = pd.DataFrame(
            data=target_data,  # 数据形状为 (24, 11)
            columns=channel_names,
            index=pd.Index(range(24), name="hour")
        )

        # 从文件路径中提取日期
        date_str = self.file_path1.split('\\')[-1].split('.')[0]  # 提取文件名部分并去除扩展名
        date_format = "%Y%m%d"
        date = datetime.strptime(date_str, date_format)

        # 修改 hour 列为时间格式
        df.index = pd.date_range(start=date, periods=24, freq='H')

        # 保存到 CSV
        output_csv = "D:\\FengDian\\train_dataset\\new_train_dataset\\output_data1.csv"
        df.to_csv(output_csv, index=True, index_label="hour")

    # 将同一个文件夹中的所有nc转换为csv
    def translate_NetCDF_toCSV_Folder(self):
        # 获取文件夹中所有.nc文件的文件名
        nc_files = [f for f in os.listdir(self.folder_path) if f.endswith('.nc')]
        nc_files.sort()  # 按文件名排序，确保按顺序处理
        # 遍历每个.nc文件
        for file_name in nc_files:
            # 构造文件路径
            folder_path = os.path.join(self.folder_path, file_name)

            # 读取NetCDF文件
            ds = xr.open_dataset(folder_path)

            # 提取6x6位置的数据（Python使用0-based索引）
            lon_idx = 5
            lat_idx = 5
            target_data = ds['data'][0, :, :, lat_idx, lon_idx].values  # shape: (24, 8)

            # 定义新的channel顺序并舍弃tcc列
            # channel_names = ['ghi', 'poai', 'sp', 't2m', 'tp', 'u100', 'v100']
            # target_data = target_data[:, [0, 2, 3, 4, 5, 6, 7]]
            if self.channel_choose == "NWP_1" or self.channel_choose == "NWP_3":
                channel_names = ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100']
            else:
                channel_names = ['ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100']
            # print(self.channel_choose)
            """
            假设原始channel顺序为 'ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100'
            根据新的channel顺序和舍弃tcc的要求，重新排列和选择数据
            这里需要根据实际情况调整索引，假设新的顺序对应的数据在原始数据中的列索引为 [0, 2, 3, 4, 6, 7]
            如果实际顺序不同，需要修改以下索引 
            """
            # 创建DataFrame
            df = pd.DataFrame(
                data=target_data,
                columns=channel_names,
                index=pd.Index(range(24), name="hour")
            )

            # 从文件名中提取日期并设置为时间索引
            date_str = file_name.split('.')[0]
            date_format = "%Y%m%d"
            date = datetime.strptime(date_str, date_format)
            df.index = pd.date_range(start=date, periods=24, freq='H')

            # 构造输出CSV文件路径
            output_csv = os.path.join(self.folder_path, f"{date_str}.csv")

            # 保存为CSV文件
            df.to_csv(output_csv, index=True, index_label="hour")

    # 用于将nc类型保存的csv文件合并
    def merge_csv_baseFolder(self):
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        csv_files.sort()  # 按文件名排序，确保按顺序合并

        # 初始化一个空的DataFrame，用于存储合并后的数据
        merged_df = pd.DataFrame()

        # 遍历每个CSV文件
        for file_name in csv_files:
            # 构造文件路径
            file_path = os.path.join(self.folder_path, file_name)

            # 读取CSV文件
            df = pd.read_csv(file_path, index_col='hour', parse_dates=True)

            # 将数据追加到合并后的DataFrame中
            merged_df = pd.concat([merged_df, df])

        # 构造输出合并后的CSV文件路径
        output_csv = os.path.join(self.folder_path, "merged_data.csv")
        # 保存合并后的数据到CSV文件
        merged_df.to_csv(output_csv, index=True, index_label="hour")

    # 两个csv文件的右侧拼接
    def merge_csv_LR(self):
        # 读取 a.csv 和 b.csv
        df_a = pd.read_csv(self.file_path1, encoding='latin-1')
        df_b = pd.read_csv(self.file_path2, encoding='latin-1')

        # 确保两个文件的行数一致
        if len(df_a) != len(df_b):
            raise ValueError("两个文件的行数不一致，无法拼接")

        # 将 b.csv 的内容拼接到 a.csv 的右侧
        merged_df = pd.concat([df_a, df_b], axis=1)

        # 保存合并后的数据到新的 CSV 文件
        output_file = r"D:\FengDian\train_dataset\merge_data\train_55888\merged_result.csv"  # 替换为输出文件的路径
        merged_df.to_csv(output_file, index=False)

        print(f"合并后的文件已保存至：{output_file}")
        # # 确保 a_data 的行数是 4 的倍数
        # if len(a_data) % 4 != 0:
        #     raise ValueError("file_path2.csv 的行数不是 4 的倍数")
        #
        # # 计算每四行的平均值（忽略 0 值）
        # preP_values = []
        # for i in range(0, len(a_data), 4):
        #     chunk = a_data.iloc[i:i + 4]['功率(MW)']
        #     non_zero_values = chunk[chunk != 0]
        #     if len(non_zero_values) == 0:
        #         preP_values.append(0)
        #     else:
        #         preP_values.append(non_zero_values.mean())
        #
        # # 将 preP_values 转换为与 merged_data 行数一致的列表
        # # 假设 merged_data 的行数与 preP_values 的长度一致
        # if len(preP_values) != len(merged_data):
        #     raise ValueError("merged_data 的行数与 preP_values 的长度不一致")
        #
        # # 将 preP_values 添加到 merged_data 的最后一列
        # merged_data['preP'] = preP_values
        #
        # # 保存合并后的数据到新的 CSV 文件
        # save_path = self.save_path  # "
        # # .csv"
        # merged_data.to_csv(save_path, index=True, index_label="hour")

    def merge_normalizationAnd_addStation(self):
        # 获取文件夹中所有.csv文件的文件名
        csv_files = [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        csv_files.sort()  # 按文件名排序，确保按顺序合并

        # 初始化一个空的DataFrame，用于存储合并后的数据
        merged_df = pd.DataFrame()

        # 遍历每个CSV文件
        for i, file_name in enumerate(csv_files):
            # 构造文件路径
            file_path = os.path.join(self.folder_path, file_name)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 添加主关键字列，列名为 'station'
            df['station'] = i + 1  # 主关键字，1-10

            # 将数据追加到合并后的DataFrame中
            merged_df = pd.concat([merged_df, df])

        # 构造输出合并后的CSV文件路径
        output_csv = os.path.join(self.folder_path, "merged_normalization.csv")

        # 保存合并后的数据到CSV文件
        merged_df.to_csv(output_csv, index=False)

        print(f"合并后的CSV文件已保存至：{output_csv}")

    def avg_mergeNormalizationCSV(self):
        df = pd.read_csv(self.file_path1, encoding='latin-1')

        # 确保行数是 4 的倍数
        if len(df) % 4 != 0:
            raise ValueError("文件的行数不是 4 的倍数")

        # 计算第二列和第三列每四行的平均值（忽略 0 值）
        col2_values = []
        col3_values = []

        for i in range(0, len(df), 4):
            chunk = df.iloc[i:i + 4]

            # 计算第二列的平均值
            col2_chunk = chunk.iloc[:, 1]  # 第二列的索引是 1
            non_zero_col2 = col2_chunk[col2_chunk != 0]
            if len(non_zero_col2) == 0:
                col2_values.append(0)
            else:
                col2_values.append(non_zero_col2.mean())

            # 计算第三列的平均值
            col3_chunk = chunk.iloc[:, 2]  # 第三列的索引是 2
            non_zero_col3 = col3_chunk[col3_chunk != 0]
            if len(non_zero_col3) == 0:
                col3_values.append(0)
            else:
                col3_values.append(non_zero_col3.mean())

        # 创建一个新的 DataFrame 来存储结果
        result_df = pd.DataFrame({
            'P': col2_values,  # 功率
            'station': col3_values  # 站点
        })

        # 保存结果到新的 CSV 文件
        output_file_path = r"D:\FengDian\train_dataset\fact_data\merged_avg.csv"  # 替换为你的输出文件路径
        result_df.to_csv(output_file_path, index=False)

        print(f"结果已保存至：{output_file_path}")

    '''需要修改，路径需要优化'''

    def merge3NMP(self):
        for time in range(1, 11):
            # 设置包含CSV文件的文件夹路径
            folder_path = fr"D:\FengDian\train_dataset\POWER_PRED_train\{time}"  # 替换为你的文件夹路径

            # 获取文件夹中所有.csv文件的文件名
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            csv_files.sort()  # 按文件名排序，确保按顺序合并

            # 初始化一个空的DataFrame，用于存储合并后的数据
            merged_df = pd.DataFrame()

            # 遍历每个CSV文件
            for i, file_name in enumerate(csv_files):
                # 构造文件路径
                file_path = os.path.join(folder_path, file_name)

                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 重命名第2-9列，添加编号
                new_columns = df.columns.tolist()
                for col_idx in range(1, 9):  # 第2列到第9列的索引是1到8
                    new_columns[col_idx] = f"{df.columns[col_idx]}_{i + 1}"

                df.columns = new_columns

                # 如果是第一个文件，直接作为基础数据
                if i == 0:
                    merged_df = df
                else:
                    # 对于其他文件，只取第2-9列，并将其添加到合并后的DataFrame中
                    merged_df = pd.concat([merged_df, df.iloc[:, 1:9]], axis=1)

            # 确保第一列不重复
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

            # 构造输出合并后的CSV文件路径
            output_csv = os.path.join(folder_path, fr"merged_data{time}.csv")

            # 保存合并后的数据到CSV文件
            merged_df.to_csv(output_csv, index=False)

            print(f"合并后的CSV文件已保存至：{output_csv}")




# a = NetCDF_Translation()
# a.file_path1=fr"D:\FengDian\train_dataset\new_train_dataset\nwp_data_train\6\NWP_1\20240101.nc"
# a.translate_NetCDF_toCSV_single()
#
# a.showNCdata()
# for time in range(1, 11):
#     for channel in range(1, 4):
#         path = fr"D:\FengDian\train_dataset\POWER_PRED_train\{time}\NWP_{channel}"
#         a = NetCDF_Translation()
#         a.folder_path = path
#         a.channel_choose = "NWP_" + str(channel)
#
#         # 将每个子目录下的nc文件进行转换
#         # a.translate_NetCDF_toCSV_Folder()
#
#         # 合并
#         # a.merge_csv1()
#
#         # 进一步合并
#         a.file_path1 = fr"D:\FengDian\train_dataset\POWER_PRED_train\{time}\NWP_{channel}\merged_data.csv"
#         a.file_path2 = fr"D:\FengDian\train_dataset\fact_data\{time}_normalization.csv"
#         a.save_path = fr"D:\FengDian\train_dataset\merge_data\train{time}_NWP_{channel}.csv"
#         a.merge_csv2()
