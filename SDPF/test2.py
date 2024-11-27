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
input = "$bm$_rms>sensors.sensor1.r_rms"
input.replace('$bm$', 'bm1')
print(input)