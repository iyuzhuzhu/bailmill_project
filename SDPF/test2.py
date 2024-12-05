from matplotlib import pyplot as plt
import matplotlib
import numpy as np
# print(np.arange(0, 10, 1))
# a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
#
# for i in a:
#     if i == 'SimHei':
#         print(i)


import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# 示例数据
# data = {
#     'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-06', '2024-01-09'],
#     'value': [10, 15, 13, 17, 18]
# }
data = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-06', '2024-01-09']
value = [10, 15, 13, 17, 18]
# 将数据转换成 DataFrame 并解析日期
df = pd.DataFrame(data)
print(df)
df[0] = pd.to_datetime(df[0])  # 将日期列转换为 datetime 类型
print(df[0][0])
# 创建图形
fig, ax = plt.subplots()

# 绘制数据
ax.plot(df[0], value, marker='o')

# 设置日期格式
date_form = mdates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(date_form)
# 使用自动日期定位器
date_locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
ax.xaxis.set_major_locator(date_locator)
# 自动调整日期标签，防止重叠
fig.autofmt_xdate()

# 添加标题和标签
plt.title('Time Series Trend')
plt.xlabel('Date')
plt.ylabel('Value')

# 显示图形
plt.show()
# 配置日志
# logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
#
# def some_function():
#     try:
#         # 你的代码
#         1 / 0  # 示例：引发一个除零错误
#     except Exception as e:
#         logging.error(f"An error occurred: {e}")
#         print(f"An error occurred: {e}")
#
# some_function()
# input = "$bm$_rms>sensors.sensor1.r_rms"
# input.replace('$bm$', 'bm1')
# print(input)
#
# db_config = {
#     "connection": "mongodb://localhost:27017/",
#     "db_name": "bm",
#     "collection": "bm1_rms"
# }
# print(db_config['connection', 'db_name'])