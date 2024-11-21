from matplotlib import pyplot as plt
import matplotlib
import numpy as np
print(np.arange(0, 10, 1))
a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

for i in a:
    if i == 'SimHei':
        print(i)