# import torch
# print(torch.__version__)           # 查看PyTorch版本
# print(torch.cuda.is_available())   # 应输出True
# print(torch.cuda.get_device_name(0))  # 显示GPU型号
import numpy as np
a = np.array([1, 2, 3])
print(a.shape[1])
# from pymongo import MongoClient
#
# # 数据库连接信息
# db_connection_info = {
#     'connection': 'mongodb://localhost:27017/',
#     'db_name': 'bm',
#     'collection': 'bm1_rms'
# }
#
# # 连接到MongoDB
# client = MongoClient(db_connection_info['connection'])
# db = client[db_connection_info['db_name']]
# collection = db[db_connection_info['collection']]
#
# # 使用aggregate和$sample阶段随机抽取一条记录
# random_record = list(collection.aggregate([{'$sample': {'size': 1}}]))
#
# # 关闭连接
# client.close()
#
# if random_record:
#     print("随机抽取的记录为：", random_record[0])
# else:
#     print("未找到任何记录。")