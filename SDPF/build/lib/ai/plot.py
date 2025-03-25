import os
import matplotlib.pyplot as plt
from general_functions.plots import limit_x_y, adjust_whole_picture


def set_plot(x, y1, y2, x_label, y_label, title, line_color1, line_color2, line_width, font='SimHei',
             num_font='Times New Roman', bwith=3, size=(12, 6), back_color='white', unicode_minus=False, x_rotation=0):
    """
    基础绘制图像
    :param back_color: 背景颜色
    :param y_max: 图像上y轴最大显示值
    :param y_min: 图像上y轴最小显示值
    :param x_max: 图像上x轴最大显示值
    :param x_min: 图像上x轴最小显示值
    :param x: 横坐标变量
    :param y: 纵坐标变量
    :param x_label: 横轴标签
    :param y_label:纵轴标签
    :param title:图像标题
    :param save_path: 图像保存地址（包括图像名称）
    :param font: 字体
    :param linewidth: 线宽
    :param linestyle: 线型
    :param color: 颜色
    :return:
    """
    # 检查并确保 size 是一个包含两个数值的元组
    plt.rcParams['font.family'] = font  # 替换为你选择的字体
    fig, ax = plt.subplots(figsize=size, facecolor=back_color)
    plt.rcParams['axes.unicode_minus'] = unicode_minus  # False正常显示负号
    # 添加标题和标签
    plt.tick_params(width=bwith)
    plt.xticks(fontproperties=num_font, weight="bold", fontsize=25, rotation=x_rotation)
    plt.yticks(fontproperties=num_font, weight="bold", fontsize=25)
    plt.xlabel(x_label, weight="bold", fontsize=25)
    plt.ylabel(y_label, weight="bold", fontsize=25)
    plt.title(title, weight="bold", fontsize=30, pad=10)
    plt.plot(x, y1, color=line_color1, label='原始信号', lw=line_width)  # 绘制原始数据
    plt.plot(x, y2, color=line_color2, label='重构信号', lw=line_width)  # 绘制重构数据
    plt.fill_between(x, y1, y2, color='lightcoral')
    plt.legend(labels=["原始信号", "重构信号"])
    adjust_whole_picture(bwith)


def plot_rec(x, y1, y2, plot_data, output_folder):
    """
    修改后的函数，可以接收两个y值列表（y1和y2），并将其在同一张图上对比展示。
    """
    line_color1 = plot_data['line_color1']
    line_color2 = plot_data['line_color2']
    x_max, x_min, y_max, y_min = plot_data['x_max'], plot_data['x_min'], plot_data['y_max'], plot_data['y_min']
    x_label, y_label, title = plot_data['x_label'], plot_data['y_label'], plot_data['title']
    output, enable_detail = plot_data['output'], plot_data['enable_detail']
    line_width = plot_data['line_width']
    output_plot_path = os.path.join(output_folder, output)
    set_plot(x, y1, y2, x_label, y_label, title, line_color1, line_color2, line_width)

    plt.savefig(output_plot_path)  # 保存图像

    if enable_detail:
        detail_plot = output.split('.')[0] + "_detail" + '.' + output.split('.')[1]
        detail_plot_path = os.path.join(output_folder, detail_plot)
        limit_x_y(x_min, x_max, y_min, y_max)
        plt.savefig(detail_plot_path)  # 保存详细视图图像

    plt.close()