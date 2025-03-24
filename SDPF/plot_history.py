import json
import os.path
from general_functions import functions, plots, database_data
from general_functions.database_data import split_string
import numpy as np
import argparse
from alarmSystem.Data.db import mongoDBInit
import sys


OUTPUT_PATH = r"/home/jtext103/test/history/$shot$/$shot_num$/$bm$/"


def split_input_collection(input, separate_identifier='>'):
    """
    将数据库输入进行切割
    :param input: 数据库查询输入 如bm1_rms>sensors.sensor1.r_rms
    :param separate_identifier: 划分input的collection的辨识符 如>
    :return: 集合名称，集合内要取出的数据对应的关键字
    """
    input_collection = split_string(input, separate_identifier)
    collection = input_collection[0]
    input = input_collection[1]
    return collection, input


def create_save_dict(sensor_num: int=6, axis=None, keys=None):
    if axis is None:
        axis = ['r_rms', 'z_rms', 'temp']
    if keys is None:
        keys = ['shot', 'value']
    save_dict = {}
    nums = np.arange(1, sensor_num+1, 1)
    for num in nums:
        sensor_name = "sensor" + str(num)
        save_dict[sensor_name] = {}
        for ax in axis:
            save_dict[sensor_name][ax] = {}
            for key in keys:
                save_dict[sensor_name][ax][key] = []
    return save_dict


class CollectionDB:
    def __init__(self, db, collection_name):
        assert db is not None
        # self.is_running = is_running
        self.collection = mongoDBInit.get_collection(db, collection_name)

    def find_latest_n_records(self, n: int, min_shot=-1, max_shot=sys.maxsize, is_running: bool=True) -> list:
        assert self.collection is not None
        if is_running:
            query = {"shot": {"$gt": min_shot, "$lte": max_shot}, "is_running": is_running}
        else:
            query = {"shot": {"$gt": min_shot, "$lte": max_shot}}
        latest_model_cursor = self.collection.find(query).sort("shot", -1).limit(n)
        latest_model_list = list(latest_model_cursor)
        return latest_model_list


class DatabaseFinder(CollectionDB):
    def __init__(self, input, shot, shot_num, db, collection_name, is_running, separate_identifier='.'):
        """
        返回数据库的一定数目的记录
        :param input: 查询字符  如sensors.sensor1.r_rms
        :param shot: 查询炮号
        :param shot_num: 需要查询的记录数目
        :param db: 已经连接到的数据库
        :param collection_name: 查询的集合名称
        :param separate_identifier: 输入的查询关键字间的划分符号 如sensors.sensor1.r_rms的划分符号为.
        """
        super().__init__(db, collection_name)
        self.input_list = split_string(input, separate_identifier)
        self.shot = shot
        self.shot_num = shot_num
        self.data = self.get_data(is_running=is_running)
        # print(self.data)

    def find_documentation_list(self, is_running):
        documentation_list = super().find_latest_n_records(self.shot_num, max_shot=self.shot, is_running=is_running)
        # print(documentation_list)
        return documentation_list

    def from_documentation_get_data(self, documentation):
        data = documentation
        for key in self.input_list:
            data = data[key]
        return data

    def get_data(self, is_running):
        data_list = []
        documentation_list = self.find_documentation_list(is_running=is_running)
        for documentation in documentation_list:
            data = self.from_documentation_get_data(documentation)
            data_list.append(data)
        return data_list


class Trend:
    def __init__(self, config_path, name, shot, shot_num, is_running, threshold, ignore_anomaly, include_collection=True,
                 collection_separate_identifier='>', input_separate_identifier='.',
                 collection=None):
        self.config = functions.read_config(config_path)
        self.threshold = threshold
        self.is_running = is_running
        self.ignore_anomaly = ignore_anomaly
        self.name = name
        self.shot = int(shot)
        self.shot_num = shot_num
        self.collection_separate_identifier, self.input_separate_identifier = (collection_separate_identifier,
                                                                               input_separate_identifier)
        self.plot_trends()

    def connect_mongodb_database(self):
        connection = self.config['db']['connection']
        db_name = self.config['db']['db_name']
        client, db = database_data.connect_mongodb_database(connection, db_name)
        return client, db

    def get_data(self, input, shot_num, db):
        """
        输入如collection>sensors.sensor1.r_rms，取出对应数据
        :param input: 输入字符串 如collection>sensors.sensor1.r_rms
        :param shot_num: 需要取出的数据炮数
        :param db: 已连接的数据库
        :return:
        """
        collection_name, data_key = split_input_collection(input)
        collection_name = functions.replace_ball_mill_name(collection_name, self.name)
        data = DatabaseFinder(data_key, self.shot, self.shot_num, db, collection_name, self.is_running,
                              self.input_separate_identifier).data
        return data

    def get_x_y(self, input, shot_num, db):
        """
        根据输入的字符串取出趋势图的x或者y数据
        """
        input = functions.replace_ball_mill_name(input, self.name)
        data = self.get_data(input, self.shot_num, db)
        return data

    def get_trend_data(self, db, y_input, output, time_input="$bm$_rms>time", x_input="$bm$_rms>shot"):
        """
                根据配置文件中的inputs和outputs的单个配置信息绘制单个趋势图
                :param x_input: 配置文件中x_input对应项
                :param db: 连接的数据库
                :param y_input: 配置文件中x_input对应项
                :param output: 配置文件中outputs的单个配置项
                :param save_folder_path: 趋势图的保存的文件夹
                :return:
                """
        file_name, title = output['file_name'], output['title']
        point_max, point_min = output['point_max'], output['point_min']
        x_label, y_label = output['x_label'], output['y_label']
        line_width, line_color, back_color = output['line_width'], output['line_color'], output['back_color']
        # save_path = os.path.join(save_folder_path, file_name)
        # x_input = "$bm$_rms>shot"  # 绘制图像的横坐标
        x_data = self.get_x_y(x_input, point_max, db)
        y_data = self.get_x_y(y_input, point_max, db)
        time_data = self.get_x_y(time_input, point_max, db)
        return x_data, y_data, time_data

    def plot_trend(self, x_data, y_data, output, save_folder_path):
        """
        根据配置文件中的inputs和outputs的单个配置信息绘制单个趋势图
        :param x_input: 配置文件中x_input对应项
        :param db: 连接的数据库
        :param y_input: 配置文件中x_input对应项
        :param output: 配置文件中outputs的单个配置项
        :param save_folder_path: 趋势图的保存的文件夹
        :return:
        """
        file_name, title = output['file_name'], output['title']
        point_max, point_min = output['point_max'], output['point_min']
        x_label, y_label = output['x_label'], output['y_label']
        line_width, line_color, back_color = output['line_width'], output['line_color'], output['back_color']
        save_path = os.path.join(save_folder_path, file_name)
        if len(y_data) >= point_min:  # 不小于绘制趋势图最小的需求点数，才绘制趋势图
            plots.plot_trend(x_data, y_data, x_label, y_label, title, save_path, linewidth=line_width, color=line_color,
                             back_color=back_color, is_x_time=True, x_rotation=-10)
        # return x_data, y_data

    @staticmethod
    def find_anomaly_rms_shot(shots, y_data, threshold, sensor, axis, save_dict):
        anomaly_index = []
        for index, shot in enumerate(shots):
            if axis == 'temp':
                break
            try:
                if y_data[index] >= threshold:
                    save_dict[sensor][axis]['shot'].append(shots[index])
                    save_dict[sensor][axis]['value'].append(y_data[index])
                    anomaly_index.append(index)
            except Exception as e:
                # print(e)
                continue
        return save_dict, anomaly_index

    def save_dict_as_json(self, save_dict, save_path, json_name='anomaly_shot_summary.json'):
        save_file_path = os.path.join(save_path, json_name)
        # 将字典保存为JSON文件
        with open(save_file_path, 'w') as json_file:
            json.dump(save_dict, json_file, ensure_ascii=False, indent=4)

    @staticmethod
    def del_index_list(lst: list, indices_to_remove):
        for i in sorted(indices_to_remove, reverse=True):
            del lst[i]
        return lst

    def plot_trends(self):
        """
        绘制所有的趋势图
        """
        client, db = self.connect_mongodb_database()
        save_folder_path = self.create_save_folder()
        save_dict = create_save_dict()
        # self.copy_rename_plot_desc(save_folder_path)
        for i, y_input in enumerate(self.config['inputs']):
            # x_input = self.config['x_input']
            output = self.config['outputs'][i]
            shots, y_data, time_data = self.get_trend_data(db, y_input, output)
            sensor, axis = y_input.split('.')[1], y_input.split('.')[2]
            save_dict, anomaly_index = self.find_anomaly_rms_shot(shots, y_data, self.threshold, sensor, axis, save_dict)
            if self.ignore_anomaly:
                # shots = self.del_index_list(shots,anomaly_index)
                time_data = self.del_index_list(time_data, anomaly_index)
                y_data = self.del_index_list(y_data, anomaly_index)
            self.plot_trend(time_data, y_data, output, save_folder_path)

        save_dict['threshold'] = self.threshold
        self.save_dict_as_json(save_dict, save_folder_path)
        client.close()

    def create_save_folder(self):
        """
        将配置文件中的保存文件夹地址替换并创建对应文件夹
        :return:
        """
        save_folder_path = OUTPUT_PATH
        save_folder_path = functions.replace_ball_mill_name(save_folder_path, self.name)
        save_folder_path = functions.replace_string(save_folder_path, str(self.shot), 'shot')
        save_folder_path = functions.replace_string(save_folder_path, str(self.shot_num), 'shot_num')
        functions.create_folder(save_folder_path)
        return save_folder_path

    def copy_rename_plot_desc(self, save_path):
        plots.copy_and_rename_file(self.config['gui_config_file'], save_path, self.config['plot_desc_file'])


def get_input_params2(description):
    """
    读取命令行输入参数
    :param description: 输入描述
    :return: 输入的配置文件地址和炮号
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config_path', type=str, help='config path', default='')
    # parser.add_argument('--name', '-n', type=str, help='ball_mill_name', default='')
    parser.add_argument('--shot', '-s', type=str, help='shot', default="")
    parser.add_argument('--shot_num', '-m', type=int, help='shot_num', default=10**10)
    parser.add_argument('--is_running', '-i', type=bool, help='is_running', default=True)
    parser.add_argument('--threshold', '-t', type=float, help='threshold', default=2)
    parser.add_argument('--ignore_anomaly', '-a', type=bool, help='is_ignore_anomaly', default=True)
    args = parser.parse_args()
    # print("参数输入完成")
    return args.config_path, args.shot, args.shot_num, args.is_running, args.threshold, args.ignore_anomaly


def main():
    names = ['bm1', 'bm2', 'bm3', 'bm4']
    config_path, shot, shot_num, is_running, threshold, ignore_anomaly = get_input_params2('trend')
    shot_num = int(shot_num)
    # config_path, name, shot = r'./config.yml', 'bm1', 1110400
    for name in names:
        Trend(config_path, name, shot, shot_num, is_running, threshold, ignore_anomaly)
        print(name)


if __name__ == '__main__':
    main()
    # print(create_save_dict())
