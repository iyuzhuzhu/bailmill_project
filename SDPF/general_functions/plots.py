from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
from general_functions import functions
import yaml


def is_number(value):
    """检查 value 是否为数值类型"""
    return isinstance(value, (int, float))


def limit_x_y(x_min='-', x_max="-", y_min="-", y_max="-"):
    # 设置 x 轴范围
    if is_number(x_min) and is_number(x_max):
        plt.xlim(x_min, x_max)
    elif is_number(x_min):
        plt.xlim(left=x_min)
    elif is_number(x_max):
        plt.xlim(right=x_max)
    # 设置 y 轴范围
    if is_number(y_min) and is_number(y_max):
        plt.ylim(y_min, y_max)
    elif is_number(y_min):
        plt.ylim(bottom=y_min)
    elif is_number(y_max):
        plt.ylim(top=y_max)


def plot(x, y, x_label, y_label, title, save_path, x_min='-', x_max="-", y_min="-", y_max="-",
         font='SimHei', linewidth=1.5, bwith=3, linestyle='-', color='r', size=(12, 6), unicode_minus=False):
    """
    基础绘制图像
    :param y_max: 图像上y轴最大显示值
    :param y_min: 图像上y轴最小显示值
    :param x_max: 图像上x轴最大显示值
    :param x_min: 图像上x轴最小显示值
    :param x: 横坐标变量
    :param y: 纵坐标变量
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
    plt.figure(figsize=size)
    plt.rcParams['axes.unicode_minus'] = unicode_minus  # False正常显示负号
    # 绘制折线图
    plt.plot(x, y, lw=linewidth, ls=linestyle, c=color)
    # 添加标题和标签
    plt.tick_params(width=bwith)
    plt.xticks(fontproperties='Times New Roman', weight="bold", fontsize=25)
    plt.yticks(fontproperties='Times New Roman', weight="bold", fontsize=25)
    plt.xlabel(x_label, weight="bold", fontsize=25)
    plt.ylabel(y_label, weight="bold", fontsize=25)
    plt.title(title, weight="bold", fontsize=30, pad=10)
    limit_x_y(x_min, x_max, y_min, y_max)
    ax1 = plt.gca()
    ax1.spines["top"].set_linewidth(bwith)
    ax1.spines["right"].set_linewidth(bwith)
    ax1.spines["bottom"].set_linewidth(bwith)
    ax1.spines["left"].set_linewidth(bwith)
    plt.tight_layout()
    # c保存图形
    plt.savefig(save_path)
    plt.close()


def plot_config(x, y, plot_data, output_folder):
    """
    按照配置文件，输入的x,y进行绘图
    """
    line_color = plot_data['line_color']
    x_max, x_min, y_max, y_min = plot_data['x_max'], plot_data['x_min'], plot_data['y_max'], plot_data['y_min']
    x_label, y_label, title = plot_data['x_label'], plot_data['y_label'], plot_data['title']
    output, drop_last, enable_detail = plot_data['output'], plot_data['drop_last'], plot_data['enable_detail']
    output_plot_path = os.path.join(output_folder, output)
    plot(x, y, x_label, y_label, title, output_plot_path, color=line_color)
    if enable_detail:
        detail_plot = output.split('.')[0] + "_detail" + output.split('.')[1]
        detail_plot_path = os.path.join(output_folder, detail_plot)
        plot(x, y, x_label, y_label, title, detail_plot_path, x_min, x_max, y_min, y_max,
             color=line_color)


def copy_file(source_file, destination_dir):
    """
    将source_file文件路径对应的文件复制到目标地址文件夹destination_dir
    :param source_file: 被复制文件
    :param destination_dir: 目标文件夹
    :return:
    """
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    # 构建目标文件的完整路径
    destination_file = os.path.join(destination_dir, os.path.basename(source_file))
    # 复制文件
    try:
        shutil.copy2(source_file, destination_file)  # 使用 copy2() 保留元数据
    except Exception as e:
        pass


if __name__ == "__main__":
    x = np.arange(0, 1, 0.1)
    y = np.sin(5 * x)
    plt.plot(x, y)
    limit_x_y(x_max=0.5)
    plt.show()
