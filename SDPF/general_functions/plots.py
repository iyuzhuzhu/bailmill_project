from matplotlib import pyplot as plt
import functions
import numpy as np


def limit_x_y(x_min='-', x_max="-", y_min="-", y_max="-"):
    if x_min != '-':
        plt.xlim(left=x_min)
    if x_max != '-':
        plt.xlim(right=x_max)
    # 设置 y 轴范围
    if y_min != '-':
        plt.ylim(bottom=y_min)
    if y_max != '-':
        plt.ylim(top=y_max)


def plot(x, y, x_label, y_label, title, save_path, x_min='_', x_max="_", y_min="_", y_max="_",
         font='SimHei', linewidth=2, linestyle='_', color='r'):
    """
    基础绘制图像
    :param y_max: 图像上y轴最大显示值
    :param y_min: 图像上y轴最小显示值
    :param x_max: 图像上x轴最大显示值
    :param x_min: 图像上x轴最小显示值
    :param x: 横坐标
    :param y: 纵坐标
    :param x_label: 横轴标签
    :param y_label:纵轴标签
    :param title:图像标题
    :param save_path: 图像保存地址
    :param font: 字体
    :param linewidth: 线宽
    :param linestyle: 线型
    :param color: 颜色
    :return:
    """
    plt.rcParams['font.family'] = font  # 替换为你选择的字体
    # 绘制折线图
    plt.plot(x, y, lw=linewidth, ls=linestyle, c=color)
    # 添加标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    limit_x_y(x_min, x_max, y_min, y_max)
    # 显示图形
    plt.savefig(save_path)


def plot_single_channel_basic(single_channel_basic_config, single_sensor_raw_data, data_source, shot, name, sensor
                              ):
    pass


if __name__ == "__main__":
    x = np.arange(0, 1, 0.1)
    y = np.sin(5*x)
    plt.plot(x, y)
    limit_x_y(x_max=0.5)
    plt.show()

