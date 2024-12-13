import requests
import json
import numpy as np
from general_functions.database_data import DatabaseFinder
import os


# from general_functions.functions import single_hysteresis_alarm
# from general_functions import functions
#
# 示例使用
def cross_entropy(p, q):
    p, q = np.array(p), np.array(q)
    assert p.shape == q.shape, "The two distributions must have the same shape."
    return -np.sum(p * np.log(q + 1e-10))  # 添加一个小值避免log(0)


def rename_file(target_file, new_file_name):
    if os.path.exists(target_file):
        os.rename(target_file, new_file_name)

rename_file('./test/test3.txt', 'test/test4.txt')


from pymongo import MongoClient

# 数据库的连接信息
connection_string = "mongodb://localhost:27017/"
db_name = "bm"
collection_name = "bm1_rms"
from pymongo import MongoClient

# 数据库连接信息
db_config = {
    "connection": "mongodb://localhost:27017/",
    "db_name": "bm",
    "collection": "bm1_rms"
}

# 输入
# input_query = "bm1_rms>sensors.sensor1.r_rms"
#
client = MongoClient(db_config["connection"])
db = client[db_config["db_name"]]
collection = db[collection_name]
#
# 构建查询条件
query = {"is_running": None}

# 构建投影
projection = {"shot": 1, "_id": 0}  # 需要包含_id以便排序
# 执行查询，并按"shot"字段排序
results = collection.find(query, projection)  # 1 表示升序，-1 表示降序
# client.close()
for result in results:
    print(result)
# # # 构建投影
# # query = {"is_running": False}
# # # projection = {"shot": 1, "_id": 0}  # 需要包含_id以便排序
# # # 执行查询，并按"shot"字段排序
# # results = list(collection.find(query, projection))  # 1 表示升序，-1 表示降序
# # # client.close()
# # print(len(results))
# query = {"is_running": True}
#
# projection = {"shot": 1, "_id": 0}  # 需要包含_id以便排序
# # 执行查询，并按"shot"字段排序
# results = list(collection.find(query, projection))  # 1 表示升序，-1 表示降序
# # for result in results:
# #     print(result)
# # client.close()
# print(len(results))
#
# query = {"shot": {"$gt": -1, "$lte": 1110400}}
# training_data = collection.find(query).sort("shot", -1).limit(5000)
# training_data = list(training_data)
# print(len(training_data))
# client.close()
# database_finder = DatabaseFinder(input_query, 1108210, 2, db, "bm1_rms")
# print(database_finder.data)
# def get_mongodb_data(db_config, input_query, limit=10):
#     # 解析输入
#     collection_name, field_path = input_query.split('>')
#
#     # 建立数据库连接
#     client = MongoClient(db_config["connection"])
#     db = client[db_config["db_name"]]
#     collection = db[collection_name]
#
#     # 构建查询条件
#     query = {"is_running": True}
#
#     # 构建投影
#     projection = {field_path: 1, "shot": 1, "_id": 0}  # 需要包含_id以便排序
#     # 执行查询，并按"shot"字段排序
#     results = list(collection.find(query, projection).sort("shot", -1).limit(limit))  # 1 表示升序，-1 表示降序
#     client.close()
#     return results
#
#
# # 调用函数并打印结果
# results = get_mongodb_data(db_config, input_query, limit=5)  # 指定返回5条记录
# for result in results:
#     print(result)
# # 输入字符串
# input_string = "sensors.sensor1.axis_2_rms"
# fields_to_query = [input_string, "shot"]
# print(fields_to_query)
# 连接到 MongoDB
# client = MongoClient(connection_string)
#
# # 选择数据库和集合
# db = client[db_name]
# collection = db[collection_name]
#
# # 构建查询条件
# # query = {}
# query = {"shot": 1108200}
# # projection = {field: 1 for field in fields_to_query}
# # projection['id'] = 0
# # 查询文档
# documents = collection.find(query)
# # 处理查询结果
# for document in documents:
#     print("找到的文档:", document)

# 关闭客户端连接
# client.close()
# temp = "temp"
# print(temp.split("_")[0])
# # 创建 MongoDB 客户端
# client = MongoClient('mongodb://localhost:27017/')
#
# # 选择数据库
# db = client['bm']
#
# # 选择集合
# collection = db['bm1_rms']
#
# # 定义要查询的 shot 值
# shot_value = 1109201
#
# # 查询文档
# document = collection.find_one({"shot": shot_value})
#
# if document:
#     print("找到的文档:", document)
#     print(type(document))
# else:
#     print("未找到匹配的文档")
# if document:
#     print("可")
# else:
#     print("No")
# 关闭客户端连接
# client.close()

# # 删除数据
# delete_result = collection.delete_one({"name": "Alice"})
# print("删除的文档数:", delete_result.deleted_count)
# 关闭客户端连接
# from datetime import datetime
#
# # 获取当前的日期和时间
# current_datetime = datetime.now()
#
# # 使用 strftime 方法格式化输出，不包含毫秒部分
# formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
#
# # 打印当前的日期和时间
# print("当前的日期和时间是:", formatted_datetime)

# 如果你只想获取当前的时间，可以这样做：
# current_time = current_datetime.time()
# print("当前的时间是:", current_time)
#
# single_sensor_rms={'r_rms':1, 'z_rms':2, 'temp':3}
# index_key = ['r_rms', 'z_rms']
# sensor_index = {k: single_sensor_rms[k] for k in index_key if k in single_sensor_rms}
# print(sensor_index)
# temp = 'temp'
# print(temp.split("_"))
# p = np.array([0.1, 0.2, 0.3])
# q = np.array([0.1, 0.2, 0.3])
# r = np.array([0.2, 0.4, 0.6])
# t = np.array([0.3, 0.5, 0.7])
# print(cross_entropy(p, q))
# print(cross_entropy(r, q))
# print(cross_entropy(t, q))
# print(cross_entropy(r+t, q))
# values = [False, False, False]
# result = any(values)
# print(result)  # 输出: True
# a = ['sensor1_x_minor', 'sensor1_x_major', 'sensor1_x_fatal', 'sensor1_y_minor']
# b = ['sensor1_x_1', 'sensor1_x_2', 'sensor1_x_3', 'sensor1_y_1']
# c = functions.remove_lower_alarm(b)
# d = functions.return_final_format_alarms(c)
# print(c)
# print(d)
# a = [1 ,2, 3, 4, 5, 6]
# for i in a:
#     for j in a:
#         if i == j:
#             print("相等", i, j)
# if single_hysteresis_alarm(a, 2.9, 6, 5, 4, True):
#     print("异常")
# is_running = None
# if is_running:
#     print('等同True')
# else:
#     print('等同False')
# re = {
#                 "rms_mean": [1],
#                 "rms_x": [2],
#                 "rms_y": [3],
#                 "rms_z": [4],
#             }
# for r in re:
#     print(r)
# 需要传递的数据
# data = {
#     "bail_mill_name": "bm3",
#     "shot": "2220000",
#     "is_running": False,
#     "data_time": "2024-08-24 08:45"
# }
# shots = np.arange(100, 105)
# print(shots)
#
# data = {"bail_mill_name": "bm1", "shot": "1110000", "is_running": None, "data_time": "2024-08-23 03:09", "results": [{"senor": "sensor1", "alarm": None, "rms_mean": 0.013464680591177195, "rms_x": 0.017936836754939994, "rms_y": 0.014681936742392864, "rms_z": 0.007775268276198724}, {"senor": "sensor2", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}, {"senor": "sensor3", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}, {"senor": "sensor4", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}, {"senor": "sensor5", "alarm": None, "rms_mean": 0.015473393350410869, "rms_x": 0.007772049454963081, "rms_y": 0.010270217144423853, "rms_z": 0.02837791345184567}, {"senor": "sensor6", "alarm": None, "rms_mean": 0.0, "rms_x": 0.0, "rms_y": 0.0, "rms_z": 0.0}]}
#
# # 发送 POST 请求到 Flask 应用的 API 路由
# url = "http://localhost:5000/api/update_data"
# headers = {'Content-Type': 'application/json'}  # 请求头，指定数据格式为 JSON
#
# response = requests.post(url, headers=headers, data=json.dumps(data))
#
# # 输出服务器的响应
# print(response.status_code)  # 状态码
# print(response.json())       # 响应的JSON内容
#
# # 目标URL（Flask 应用程序的 API 端点）
# url = "http://localhost:5000/api/get_data"
#
# # 发送 GET 请求到 Flask 应用
# response = requests.get(url)
#
# # 检查请求是否成功
# if response.status_code == 200:
#     # 将响应内容解析为JSON格式
#     data = response.json()
#     print("Data received from server:", data)
#     print(type(data))
# else:
#     print("Failed to retrieve data. Status code:", response.status_code)
