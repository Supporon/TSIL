import pandas as pd
import glob

# patterns = ["fluctuation", "rise", "fall", "extreme"]
# merged_data = []

# for pattern in patterns:
#     all_files = glob.glob(f"./FinTSB_cn/{pattern}/dataset_*.pkl")
#     for source_id, file_path in enumerate(sorted(all_files)):
#         df = pd.read_pickle(file_path)
    
#         df = df.reset_index()
#         df['source'] = source_id
    
#         df = df.set_index(['datetime', 'instrument'])
#         # print(df)
#         # break
#         merged_data.append(df)

# print(len(merged_data))
# final_df = pd.concat(merged_data)
# print(final_df.shape)
# final_df.to_pickle("./FinTSB_cn/merged_dataset.pkl")

import qlib
from qlib.data import D
import pandas as pd

# # 1. 初始化QLib（确保provider_uri指向数据根目录）
# qlib.init(provider_uri="./qlib_data/cn_data", region="cn")

# # 2. 获取所有instruments（ticker列表）
# instruments = D.instruments(market='all')

# # 3. 定义需加载的字段（与dump_bin.py的--include_fields对应）
# fields = ['$y10', '$main_ret_slp', '$ret', '$close_adj', '$high_adj', '$low_adj', '$open_adj', '$tr', '$capvol0', '$volume' ] # 一定要$
# # instruments = D.list_instruments(instruments=instruments, start_time='2010-01-01', end_time='2023-12-31', as_list=True)
# # 4. 加载数据（核心步骤）
# df = D.features(
#     instruments=instruments,
#     fields=fields,
#     start_time='2010-01-01',
#     end_time='2023-12-31',
#     freq='day'
# )

# # 5. 查看结果（此时df应为非空DataFrame）
# print(df.head())
# # 5. 保存
# df.to_pickle("/home/dmz-ai/zhengzengrong/FinTSB-main/data/FinTSB_cn/merged_dataset.pkl")



# fea = pd.read_pickle('/home/dmz-ai/zhengzengrong/FinTSB-main/tra/feature.pkl')
# label = pd.read_pickle('/home/dmz-ai/zhengzengrong/FinTSB-main/tra/label.pkl')
# ret = pd.read_pickle('/home/dmz-ai/zhengzengrong/FinTSB-main/tra/ret.pkl')
# df = pd.concat([fea, label], axis=1)
# df.to_pickle('/home/dmz-ai/zhengzengrong/FinTSB-main/data/FinTSB_cn/merged_dataset.pkl')



