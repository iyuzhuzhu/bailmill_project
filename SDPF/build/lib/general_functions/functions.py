from h5py import File
import yaml
import os
import json
import argparse
import numpy as np
from datetime import datetime
import requests
from pymongo import MongoClient
from ruamel.yaml import YAML


def get_input_params(description):
    """
    读取命令行输入参数
    :param description: 输入描述
    :return: 输入的配置文件地址和炮号
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config_path', type=str, help='config path', default='')
    parser.add_argument('--shot', '-s', type=str, help='shot', default="")
    args = parser.parse_args()
    # print("参数输入完成")
    return args.config_path, args.shot


def read_config(config_path):
    """
    :param config_path: 配置文件地址
    :return: 配置文件信息
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config_data


def get_prefix_address(path, split='$bm$'):
    """
    将配置文件中的data_source的$split$前的前缀地址取出
    :param path: 地址
    :param split:切割处
    :return:切割处前的代码
    """
    prefix_address = ""
    while split in path:
        prefix_address, _ = os.path.split(path)
        path = prefix_address
    return prefix_address


def get_bail_names(path):
    """
    得到球磨机的名称与数目，此处默认球磨机命名bm1 bm2....
    :param path: config文件中的data_source地址
    :return:球磨机名称列表
    """
    prefix_address = get_prefix_address(path, split='$bm$')
    entries = os.listdir(prefix_address)
    # 检查每个条目是否为目录，并收集它们
    bail_names = [entry for entry in entries if entry[:2] == 'bm']
    return bail_names


def replace_path(data_source, shot, bm, sensor):
    """
    :param data_source: config文件中的datasource,Inference等数据存放地
    :param shot: 炮号
    :param bm: 球磨机名称 如bm1
    :param sensor: 传感器名称 如sensor1
    :return: 替换了所有$str$的datasource
    """
    shot_2 = int(int(shot) / 100)
    data_source = data_source.replace('$shot$', shot)
    data_source = data_source.replace('$shot_2$', str(shot_2))
    data_source = data_source.replace('$bm$', bm)
    data_source = data_source.replace('$sensor$', sensor)
    # print(data_source)
    return data_source


def read_hdf5(data_path):
    """
    :param data_path: hdf5文件地址
    :return:
        sample_data: 采集设置的参数
        data: 数据
    """
    with File(data_path, 'r') as f:
        sample_data, data = {}, {}
        for key in f.keys():
            if key == 'Attribute':
                for name, value in f[key].attrs.items():
                    sample_data[name] = value
                    # print(f"{key}: {value}")
            else:
                data[key] = f[key][:]  # data[key]数据类型为numpy数组
    return sample_data, data


def get_single_sensor_data(data_source, shot, bm, sensor):
    """
    返回单个sensor采集的hdf5文件的信息
    :param data_source: config文件中的datasource即数据存放地
    :param shot: 炮号
    :param bm: 球磨机名称 如bm1
    :param sensor: 传感器名称 如sensor1
    :return:
        sample_data: 采集设置的参数
        data: 数据
    """
    data_source = replace_path(data_source, shot, bm, sensor)
    sample_data, data = read_hdf5(data_source)
    return sample_data, data


def channel_to_axis(channels_to_axis, channel):
    """
    输入channel0，1，2与x,y,z的对应关系和channel，返回该channel对应的轴向
    :param channels_to_axis: channel0，1，2与x,y,z的对应关系 如config['channels']
    :param channel: 需要确定对应轴向的channel
    :return: 输入的channel对应的x,y,z轴
    """
    for channel_axis_dict in channels_to_axis:
        channel_, _ = channel_axis_dict.copy().popitem()
        if channel == channel_:
            return channel_axis_dict[channel]


def return_datatime(data_time_float):
    """
    将输入的时间浮点数转化为时间戳, 若hdf5中时间浮点数不存在，则返回None
    :param data_time_float: 时间浮点数 np.float64
    :return: 时间类型数据
    """
    try:
        # 将数字转换为字符串
        data_str = str(data_time_float)
        # 使用 strptime 方法解析日期时间，只到分钟
        date_time = datetime.strptime(data_str[:12], "%Y%m%d%H%M")
        formatted_date = date_time.strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        formatted_date = None
    return formatted_date


def save_json(json_data, json_name, output_path):
    """
    :param json_data: 需要以json形式保存的数据
    :param json_name: 保存json文件的名字
    :param output_path: json文件的保存地址
    :return:
    """
    output_path = os.path.join(output_path, json_name)
    json_string = json.dumps(json_data)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_string)


def replace_placeholders(string, replacement_params):
    """
    :param string: 需要被替换的字符串，替换$key$
    :param replacement_params: 用于替换的键值对，key表示原字符串被替换的部分，value表示替换的部分
    :return:
    """
    for key, value in replacement_params.items():
        placeholder_key = f"${key}$"
        string = string.replace(placeholder_key, value)
    return string


def create_output_folder(output_path, shot, bm):
    """
    创建输出的文件夹
    :param output_path: 输出地址（含被替换部分）
    :param shot:
    :param bm:
    :return:
    """
    shot_2 = int(int(shot) / 100)
    output_path = output_path.replace('$shot$', shot)
    output_path = output_path.replace('$shot_2$', str(shot_2))
    output_path = output_path.replace('$bm$', bm)
    # output_path = replace_path(output_path, shot, bm, sensor)
    try:
        os.makedirs(output_path, exist_ok=True)
        return output_path
    except Exception as e:
        pass


# def single_summary(bm, shot, datatime_float, results, is_running=None):
#     """
#     将单独炮号的信息模型总结
#     :param bm: 球磨机名称 bm1 bm2
#     :param shot: 炮号
#     :param datatime_float: 时间的浮点数 如202408261140
#     :param results: 该炮号的分析结果
#     :param is_running: 该炮号采集时，球磨机是否正在运行
#     :return: 单独炮号分析总结
#     """
#     single_summary = dict()
#     single_summary['bail_mill_name'] = bm
#     single_summary['shot'] = shot
#     single_summary['data_time'] = return_datatime(datatime_float)
#     single_summary['is_running'] = is_running
#     single_summary['results'] = results
#     return single_summary


# def single_summary_save(bm, shot, datatime_float, results, output_path, model_name, is_running=None):
#     """
#     保存单独炮号数据分析数据
#     :param bm: 球磨机名称 bm1 bm2
#     :param shot: 炮号
#     :param datatime_float: 时间的浮点数 如202408261140
#     :param results: 该炮号的分析结果
#     :param output_path: config中的输出路径 如Inference_path
#     :param model_name: 模型名称 如rms fft ai
#     :param is_running: 数据采集时， 球磨机是否正在运行
#     :return: 单独炮号分析总结
#     """
#     single_summary_dict = single_summary(bm, shot, datatime_float, results, is_running)
#     output_path = create_output_folder(output_path, shot, bm)
#     json_name = 'single_summary' + '_' + model_name + '.json'
#     save_json(single_summary_dict, json_name, output_path)
#     return single_summary_dict


def return_single_summary_data(output_path, shot, bm, model_name):
    """
    返回对应single_summary_modelname.json文件数据
    :param output_path: 输出地址（含被替换部分）config['Inference_path']部分
    :param shot: 返回的炮号
    :param bm: 球磨机名称
    :param model_name: 模型名称 如rms fft
    :return: json文件数据
    """
    output_path = create_output_folder(output_path, shot, bm)
    json_name = 'single_summary' + '_' + model_name + '.json'
    single_summary_path = os.path.join(output_path, json_name)
    with open(single_summary_path, 'r') as file:
        single_summary_data = json.load(file)
    return single_summary_data


def get_web_is_running(url, data_key='is_running'):
    """
    由web_api得到当前炮号is_running的数据
    :param url: 获取数据的api的url 如"http://localhost:5000/api/get_data"
    :param data_key: 返回得到的数据字典的数据对应的key
    :return: 或许到的is_running数据
    """
    response = requests.get(url)
    # 检查请求是否成功
    if response.status_code == 200:
        # 将响应内容解析为JSON格式
        data = response.json()
        return data[data_key]
    else:
        print("Failed to retrieve data. Status code:", response.status_code)


def update_web_data(data, url=None, headers=None):
    """
    更新网页数据
    :param data: 需要更新上去的数据
    :param url: 更新数据的api的url 如"http://localhost:5000/api/update_data"
    :param headers: 请求头，指定数据格式为 JSON
    :return:
    """
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    if url is None:
        url = "http://localhost:5000/api/update_data"
    requests.post(url, headers=headers, data=json.dumps(data))


def get_alarm_config(alarm_config_path, bm):
    """
    得到报警的配置文件信息
    :param: bm 球磨机名称
    :return:
    """
    alarm_config_path = alarm_config_path.replace('$bm$', bm)
    alarm_config, ruamel_yaml = load_yaml(alarm_config_path)
    return alarm_config, ruamel_yaml, alarm_config_path


def load_yaml(file_path):
    """
    得到配置文件信息，便于更新
    :param file_path:
    :return:
    """
    ruamel_yaml = YAML()
    yaml.preserve_quotes = True  # 保留引号
    ruamel_yaml.default_flow_style = False  # 使用块样式而非流样式
    with open(file_path, 'r', encoding='utf-8') as file:
        data = ruamel_yaml.load(file)
    return data, ruamel_yaml


def save_yaml(data, file_path, ruamel_yaml):
    """
    保存修改好的配置文件数据
    :param data: 配置文件数据
    :param file_path: 保存地址
    :param ruamel_yaml: 之前读取创建的YAML对象
    :return:
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        ruamel_yaml.dump(data, file)


def alarm_config_axis(axis, data_list, h, hh, sensor_threshold):
    """
    根据已有的数据列表与异常分位点，得到异常阈值列表
    :param hh: 2级阈值分位点
    :param h: 1级阈值分位点
    :param axis: 轴向数据
    :param data_list:
    :param sensor_threshold:异常阈值字典
    :return:
    """
    h_threshold = np.percentile(data_list, 100 * h)
    hh_threshold = np.percentile(data_list, 100 * hh)
    sensor_threshold['of_h_' + axis.split("_")[0]] = h_threshold
    sensor_threshold['of_hh_' + axis.split("_")[0]] = hh_threshold
    return sensor_threshold


def order_alarm_dict(sensor_threshold_dict, round_num=5):
    """
    将异常阈值字典按照次序排列输出，并将字典内的数值类数据变为可录入yaml文件格式
    :param round_num: 保留的数据小数点位数
    :param sensor_threshold_dict:
    :return:
    """
    for key in sensor_threshold_dict.keys():
        if key == 'sensor_prefix':
            continue
        else:
            sensor_threshold_dict[key] = convert_to_serializable(sensor_threshold_dict[key], round_num)
            sensor_threshold_dict[key] = convert_floats_to_strings(sensor_threshold_dict[key])
    ordered_data = {
        "sensor_prefix": sensor_threshold_dict['sensor_prefix'],
        "of_h_r": sensor_threshold_dict["of_h_r"],
        "of_hh_r": sensor_threshold_dict["of_hh_r"],
        "of_h_z": sensor_threshold_dict["of_h_z"],
        "of_hh_z": sensor_threshold_dict["of_hh_z"],
        "of_h_temp": sensor_threshold_dict["of_h_temp"],
        "of_hh_temp": sensor_threshold_dict["of_hh_temp"]
    }
    return ordered_data


def convert_to_serializable(data, round_num):
    """递归地将数据转换为可序列化的格式."""
    if isinstance(data, float):
        return round(data, round_num)  # 截断为可接受的精度
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value, round_num) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item, round_num) for item in data]
    return data


def convert_floats_to_strings(data):
    """递归地将浮点数转换为字符串格式，以避免序列化问题."""
    if isinstance(data, float):
        return str(data)  # 将浮点数转换为字符串
    elif isinstance(data, dict):
        return {key: convert_floats_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_floats_to_strings(item) for item in data]
    return data


def get_last_alarm(output_path, shot, bm, model_name, sensor):
    """
    返回当前炮号的上一炮的对应的传感器results的alarm:对应的值
    :param output_path: 输出地址（含被替换部分）config['Inference_path']部分
    :param shot: 返回的炮号
    :param bm: 球磨机名称
    :param model_name: 模型名称 如rms fft
    :param sensor: 传感器名称
    :return:
    """
    try:
        shot = str(int(shot) - 1)
        last_shot_summary = return_single_summary_data(output_path, shot, bm, model_name)
        last_results = last_shot_summary['results']
        for last_result in last_results:
            if last_result['sensor'] == sensor:
                return last_result['alarm']
            else:
                continue
    except Exception as e:
        return None


def alarm_str_to_level(alarm_str):
    """
    将报警字符返回报警等级 如sensor1_x_major返回2
    :param alarm_str:
    :return:
    """
    if alarm_str[-5:] == 'minor':
        return 1
    elif alarm_str[-5:] == 'major':
        return 2
    elif alarm_str[-5:] == 'fatal':
        return 3


def transform_alarm_to_bool(alarm, alarm_threshold_key):
    """
    通过json数据读取出来的alarm列表，返回输入的alarm-config中的sensor_threshold中的阈值对应异常级别是否在上次是异常
    :param alarm:
    :param alarm_threshold_key: alarm_config中的sensor_threshold 如of_hh_x
    :return:
    """
    if not alarm:
        return False
    for alarm_ in alarm:
        if alarm_.split('_')[1] == alarm_threshold_key.split('_')[2]:
            if alarm_str_to_level(alarm_) >= alarm_threshold_key.count('h'):
                return True
    return False


def get_last_alarm_state(output_path, shot, bm, model_name, sensor, alarm_threshold_key):
    """
    返回输入的alarm-config中的sensor_threshold中的阈值对应异常级别是否在上次是异常 True表示异常
    :param output_path: 输出地址（含被替换部分）config['Inference_path']部分
    :param shot: 返回的炮号
    :param bm: 球磨机名称
    :param model_name: 模型名称 如rms fft
    :param sensor: 传感器名称
    :param alarm_threshold_key:
    :return:
    """
    last_alarm = get_last_alarm(output_path, shot, bm, model_name, sensor)
    last_alarm_state = transform_alarm_to_bool(last_alarm, alarm_threshold_key)
    return last_alarm_state


def single_hysteresis_alarm(data, threshold, window, on, off, last_alarm_state=None):
    """
    针对于单个传感器的轴，进行滞环报警，如果已有数据不足则不报警
    :param data: 异常指数数据
    :param threshold: 阈值
    :param window: 数据需要的窗口
    :param on: 上一炮未报警时，异常指数序列中异常数阈值
    :param off: 上一炮报警时，异常指数序列中异常数阈值
    :param last_alarm_state: 上一炮的报警状态
    :return: 报警，若为True则报警，反之则不报警
    """
    if len(data) < window:
        return None
    alarm_shot_num = 0
    for i in data:
        if i >= float(threshold):
            alarm_shot_num += 1
    # print(alarm_shot_num)
    if last_alarm_state:
        if alarm_shot_num >= off:
            return True
        else:
            return False
    else:
        if alarm_shot_num >= on:
            return True
        else:
            return False


def transform_alarm_bool(alarm_state, alarm_level, sensor, axis):
    """
    根据报警情况，以及报警等级，传感器与轴，返回报警字符串 如sensor1_x_major
    :param alarm_state: 是否报警 True or False or None(None表示目前无法判断是否报警）
    :param alarm_level: 报警等级 minor major fatal
    :param sensor: 传感器名称 senor1
    :param axis:
    :return:
    """
    if alarm_state:
        alarm = sensor + '_' + axis + '_' + alarm_level
        return alarm
    else:
        return False


def get_sensor_alarm_threshold(sensors_threshold, sensor):
    """
    取出alarm_config对应传感器的sensor_threshold
    :param sensors_threshold: alarm_config的sensors_threshold
    :param sensor: 需要进行判别是否异常的传感器
    :return:
    """
    for sensor_threshold in sensors_threshold:
        if sensor_threshold['sensor_prefix'] == sensor:
            sensor_threshold_ = {k: v for k, v in sensor_threshold.items() if k != 'sensor_prefix'}
            return sensor_threshold_
        else:
            continue


def return_final_format_alarms(alarms):
    """
    将报警中本来的1，2，3级别报警替换成minor，major，fatal,并按照xyz轴顺序排列故障
    :param alarms: 报警列表
    :return: 字符 char 在字符串 s 中出现的次数
    """
    alarms_ = []
    for alarm in alarms:
        if alarm[-1] == '1':
            alarm_ = alarm[:-1] + 'minor'
        elif alarm[-1] == '2':
            alarm_ = alarm[:-1] + 'major'
        elif alarm[-1] == '3':
            alarm_ = alarm[:-1] + 'fatal'
        alarms_.append(alarm_)
    sorted_alarms = sorted(alarms_, key=lambda element: element.split('_')[1])
    return sorted_alarms


def single_sensor_alarm(output_path, shot, bm, model_name, data, sensor_threshold, window, on, off):
    """
    得到异常状态的列表
    :param output_path: 输出地址（含被替换部分）config['Inference_path']部分
    :param shot: 返回的炮号
    :param bm: 球磨机名称
    :param model_name: 模型名称 如rms fft
    :param data: 当前炮号开始的window
    :param sensor_threshold: alarm_config中sensors_threshold的列表单个字典元素
    :param window:
    :param on:
    :param off:
    :return:
    """
    alarms, sensor = [], sensor_threshold['sensor_prefix']
    for key, threshold in sensor_threshold.items():
        if key == 'sensor_prefix' or key[-1] == 'g':
            continue
        else:
            for axis, data_list in data.items():
                if axis.split('_')[1] == key.split('_')[2]:
                    last_alarm_state = get_last_alarm_state(output_path, shot, bm, model_name, sensor, key)
                    alarm_state = single_hysteresis_alarm(data_list, threshold, window, on, off, last_alarm_state)
                    if alarm_state:
                        alarm_level = str(key.count('h'))  # 通过h数量判定几级异常阈值
                        alarm = transform_alarm_bool(alarm_state, alarm_level, sensor, key[-1])
                        alarms.append(alarm)
                    else:
                        continue
    alarms = remove_lower_alarm(alarms)
    alarms = return_final_format_alarms(alarms)
    return alarms


def remove_lower_alarm(alarms):
    """
    报了高级的警则将低级的警剔除
    :param alarms:
    :return:
    """
    alarms_remove = []
    for alarm in alarms:
        # print(alarm[-1])
        for alarm_ in alarms:
            if alarm[:-1] == alarm_[:-1] and int(alarm[-1]) < int(alarm_[-1]):
                alarms_remove.append(alarm)
    alarms, alarms_remove = set(alarms), set(alarms_remove)
    alarms = alarms - alarms_remove
    return list(alarms)


def single_shot_summary(bm, shot, sensors, datatime_float, is_running=None):
    """
    将单独炮号的信息模型总结，新设计
    :param sensors: summary sensors信息汇总字典
    :param bm: 球磨机名称 bm1 bm2
    :param shot: 炮号
    :param is_running: 该炮号采集时，球磨机是否正在运行
    :return: 单独炮号分析总结
    """
    single_summary = dict()
    single_summary['ball_mill_name'] = bm
    single_summary['shot'] = int(shot)
    # 获取当前的日期和时间
    # current_datetime = datetime.now()
    # # 使用 strftime 方法格式化输出，不包含毫秒部分
    # formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    single_summary['time'] = return_datatime(datatime_float)
    single_summary['is_running'] = is_running
    single_summary['sensors'] = sensors
    return single_summary


def save_single_summary_mongodb(single_summary, address, collection_name, shot, database_name='bm'):
    """
    将数据保存到MongoDB数据库,如果当前炮号已被分析，则对已有的结果进行替换
    :param shot: 炮号
    :param single_summary: 分析得到的数据
    :param address: 数据库域名地址
    :param collection_name: 集合名称
    :param database_name: 数据库名称
    :return:
    """
    shot = int(shot)
    # 创建 MongoDB 客户端
    client = MongoClient(address)
    # 选择数据库
    db = client[database_name]
    # 选择集合
    collection = db[collection_name]
    # 创建唯一索引
    try:
        collection.create_index("shot", unique=True)
    except Exception as e:
        pass
    # 检查文档是否已经存在
    existing_document = collection.find_one({"shot": shot})
    if existing_document:
        # 替换现有文档
        result = collection.replace_one({"shot": shot}, single_summary)
    else:
        # 插入新文档
        result = collection.insert_one(single_summary)
    # 关闭客户端连接
    client.close()


def create_and_save_single_summary(bm, shot, sensors, address, collection_name, database_name, is_running=None):
    single_summary = single_shot_summary(bm, shot, sensors, is_running)
    save_single_summary_mongodb(single_summary, address, collection_name, database_name)


def save_alarm_config(file_path, alarm_config_data):
    pass
