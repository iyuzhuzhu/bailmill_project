import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from h5py import File
import yaml
import argparse
import os
import json


def replace_placeholders(string, replacement_params):
    for key, value in replacement_params.items():
        placeholder_key = f"${key}$"
        string = string.replace(placeholder_key, value)
    return string


def create_output_folder(output_path, shot):
    output_path = os.path.join(output_path, shot)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        return output_path
    else:
        return output_path


# 读取配置文件
def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f.read(), Loader=yaml.FullLoader)
        data_path, output_path = config_data['data_path'], config_data['output_path']
        return data_path, output_path


# 读取hdf5文件数据（data[key]数据类型为numpy数组）
def read_hdf5(data_path):
    with File(data_path, 'r') as f:
        sample_data, data = {}, {}
        for key in f.keys():
            if key == 'Attribute':
                for name, value in f[key].attrs.items():
                    sample_data[name] = value
                    # print(f"{key}: {value}")
            else:
                data[key] = f[key][:]
                # print(type(data[key]))
    return sample_data, data


# read_hdf5(r'E:\ROS\Bailmill_project\1000020.1.hdf5')

def plot(x, y, x_label, y_label, title, output_path):
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.plot(x, y)
    # x,y坐标轴名称设置
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # 显示曲线图像
    plt.savefig(output_path)
    plt.close()


def plot_raw(count_num, fs, key, value, output_path):
    x = np.arange(0, count_num / fs, 1 / fs)
    title = key + "数据原始图像"
    file_name = key + '_raw' + '.png'
    output_path = os.path.join(output_path, file_name)
    plot(x, value, "时间", "电压", title, output_path)


def plot_fft(x, y, x_label, y_label, output_path, channel):
    file_name = channel + '_fft' + '.png'
    output_path = os.path.join(output_path, file_name)
    title = channel + "FFT"
    plot(x, y, x_label, y_label, title, output_path)


def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    N = len(data)  # 信号长度
    # N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data)) / N * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    # print(result)
    return axisFreq, result


# 得到FFT分析的最大幅值对应的频率，将其视为基波频率
def find_fundamental_frequency(axis_freq, result, summary, channel):
    fundamental_frequency_index = np.argmax(result)
    fundamental_frequency = axis_freq[fundamental_frequency_index]
    summary[channel] = fundamental_frequency
    return summary


def save_summary_json(summary, output_path):
    output_path = os.path.join(output_path, 'summary.json')
    summary_string = json.dumps(summary)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(summary_string)


def main():
    # 输入参数
    parser = argparse.ArgumentParser(description='FFT')
    parser.add_argument('config_path', type=str, help='config path', default='')
    parser.add_argument('--shot', '-s', type=str, help='shot', default="")
    args = parser.parse_args()
    config_path = args.config_path
    shot = args.shot
    # config_path = './config.yml'
    # shot = '1000020.1'
    summary = {}
    # 读取配置文件，得到数据文件地址与输出地址
    data_path, output_path = read_config(config_path)
    data_path = data_path.replace('$shot$', shot)
    # 根据炮号创建输出存放的文件
    output_path = create_output_folder(output_path, shot)
    # 读取数据与传感器参数
    sample_data, data = read_hdf5(data_path)
    count_num, fs = sample_data['SampleCount'], sample_data['SampleRate']
    for key, value in data.items():
        # 画原始数据图像
        plot_raw(count_num, fs, key, value, output_path)
        # fft
        axis_freq, result = FFT(fs, value)
        plot_fft(axis_freq, result, '频率/hz', '幅值', output_path, key)
        # 得到基波幅值写入summary字典
        summary = find_fundamental_frequency(axis_freq, result, summary, key)
    # 保存json文件
    save_summary_json(summary, output_path)


if __name__ == '__main__':
    main()
