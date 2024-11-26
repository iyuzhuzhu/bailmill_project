from sklearn.cluster import KMeans
import numpy as np
from pymongo import MongoClient
from general_functions import functions, plots
import os
from matplotlib import pyplot as plt


def calculate_rms(data):
    # 计算rms
    rms = np.sqrt(np.mean(data ** 2))
    return rms


class Rms:
    def __init__(self, name, config_path, shot, model_name='rms'):
        self.date_time, self.start_axis, self.start_sensor, self.is_running, self.temp = None, None, None, None, None
        self.name, self.shot, self.model_name, self.config_path = name, shot, model_name, config_path
        self.config, self.ruamel_yaml = functions.load_yaml(self.config_path)
        self.data_source, self.output_path, self.sensors = (self.config['data_source'], self.config['Inference_path'],
                                                            self.config['sensors'])
        self.rms()

    def rms(self):
        self.training_model()
        sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name,
                                                               self.sensors)
        self.single_shot_summary(sensors_data, sample_data)
        self.plot_model(sensors_data, sample_data)
        print(self.shot)

    def training_model(self):
        """
        根据新的历史数据更新判断是否运行的阈值和报警阈值，需要手动修改is_training=True
        """
        if self.config['is_training']:
            self.train_rms_model()
            self.config['is_training'] = False
            try:
                functions.save_yaml(self.config, self.config_path, self.ruamel_yaml)
            except Exception as e:
                pass

    def create_training_data_summary(self):
        """
        返回训练数据汇总的字典模板
        :return:
        """
        training_data_summary = {}
        for sensor in self.sensors:
            training_data_summary[sensor] = {
                "r_rms": [],
                "z_rms": [],
                "temp": [],
            }
        return training_data_summary

    def get_training_data_summary(self, shot_num=None, used_training=True):
        """
        得到训练数据，也可用于取当前炮号前一炮的任意炮数据
        :return:
        """
        training_data_summary = self.create_training_data_summary()
        # 若无报错，前者为遍历过程中球磨机未运行时的采集数据炮数，后者为遍历到的运行时的炮数
        not_running_shot_num, training_shot_num = 0, 0
        if used_training:
            shot_num = self.config['training_shots']
        try:
            # 创建 MongoDB 客户端
            client = MongoClient(self.config['db']['mongodb_address'])
            # 选择数据库
            db = self.config['db']['db_name']
            # 选择集合
            collection = self.config['db']['collection']
            while training_shot_num < shot_num:
                # 报错时,跳过报错的那炮数据(如遇到采集到的有效炮数小于配置文件中的训练炮数，终止循环)
                try:
                    # 训练从前一炮的数据开始取
                    training_shot = int(self.shot) - training_shot_num - 1 - not_running_shot_num
                    single_summary_data = collection.find_one({"shot": training_shot})
                    if single_summary_data['is_running']:
                        training_data_summary = self.single_data_training_summary(single_summary_data['sensors'],
                                                                                  training_data_summary)
                        training_shot_num += 1
                    else:
                        not_running_shot_num += 1
                except Exception as e:
                    # print(e)
                    training_shot_num += 1
        except Exception as e:
            return False
        # print(training_data_summary['sensor1'])
        return training_data_summary

    def get_alarm_config(self):
        """
        得到报警的配置文件信息
        :return:
        """
        alarm_config_path = self.config['alarm_config_path'].replace('$bm$', self.name)
        alarm_config = functions.read_config(alarm_config_path)
        # self.window, self.on, self.off = alarm_config['window'], alarm_config['on'], alarm_config['off']
        return alarm_config

    def train_rms_model(self):
        """
        将得到的阈值，并写入yaml文件
        :return:
        """
        try:
            training_data_summary = self.get_training_data_summary()
            # 确认正确返回数据
            if training_data_summary:
                threshold_config, ruamel_yaml, alarm_config_path = functions.get_alarm_config(
                    self.config['threshold_config_path'], self.name)
                threshold_config = self.update_rms_start(training_data_summary, threshold_config)  # 更新判断是否启动阈值
                h, hh = threshold_config['h'], threshold_config['hh']
                sensors_rms_threshold = {}
                for sensor in training_data_summary.keys():
                    try:
                        sensor_threshold = {}
                        for rms_axis, rms_list in training_data_summary[sensor].items():
                            sensor_threshold = functions.alarm_config_axis(rms_axis, rms_list, h, hh,
                                                                           sensor_threshold)
                    except Exception as e:
                        # print(e)
                        sensor_threshold = threshold_config['sensors_threshold'][sensor]
                    sensor_threshold = functions.order_alarm_dict(sensor_threshold, 8)
                    sensors_rms_threshold[sensor] = sensor_threshold
                threshold_config['sensors_threshold'] = sensors_rms_threshold
                functions.save_yaml(threshold_config, alarm_config_path, ruamel_yaml)
        except Exception as e:
            pass
        # # print(sensors_rms_threshold)

    @staticmethod
    def update_rms_start(training_data_summary, threshold_config):
        """
        修改config文件中的ball_mills中的min_rms_start
        :return:
        """
        try:
            min_rms_start = []
            start_sensor = threshold_config['is_running_threshold']['start_sensor']
            start_axis = threshold_config['is_running_threshold']['start_axis']
            for start_sensor in start_sensor:
                rms_start_list = training_data_summary[start_sensor][start_axis]
                data = np.array(rms_start_list).reshape(-1, 1)
                # 使用 K-means 聚类
                kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0).fit(data)
                # 获取每个簇的中心
                centers = kmeans.cluster_centers_
                # 排序中心值
                sorted_centers = sorted(centers.flatten())
                # 计算两个中心之间的中点作为阈值
                threshold = (sorted_centers[0] + sorted_centers[1]) / 2
                # 输出结果
                threshold = functions.convert_to_serializable(threshold, 8)
                threshold = functions.convert_floats_to_strings(threshold)
                min_rms_start.append(threshold)
            # print(min_rms_start)
            threshold_config['is_running']['min_rms_start'] = min_rms_start
        except Exception as e:
            pass
        return threshold_config

    @staticmethod
    def single_data_training_summary(single_summary_data_results, training_data_summary):
        """
        将球磨机正在运行时采集到的数据导入训练数据
        :param single_summary_data_results: single_summary json数据中的results部分
        :param training_data_summary: 训练数据汇总
        :return: 训练数据汇总
        """
        for sensor, result in single_summary_data_results.items():
            for key in training_data_summary[sensor].keys():
                if result[key] is not None:
                    training_data_summary[sensor][key].append(result[key])
                else:
                    continue
        return training_data_summary

    def get_plot_config(self):
        """
        读取绘图配置文件
        """
        plot_config_path = self.config['plot_config_path']
        plot_config_data = functions.read_config(plot_config_path)
        return plot_config_data

    @staticmethod
    def get_plot_x_y(fs, channel_data, drop_last=True):
        """
        得到需要绘制的振动波形图像的x,y
        """
        time, vibration = None, None
        if channel_data is not None:
            if drop_last:
                vibration = channel_data[:-1]
            else:
                vibration = channel_data
            time = np.arange(0, len(vibration), 1) / fs  # 得到时间参数
        return time, vibration

    def plot_model(self, sensors_data=None, sample_data=None):
        """
        绘制振动波形图像
        """
        # sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name, self.sensors)
        plot_config_data = self.get_plot_config()
        save_plot_folder = plot_config_data['save_plot_folder']
        output_path = functions.create_output_folder(self.config['Inference_path'], self.shot, self.name)
        output_folder = os.path.join(output_path, save_plot_folder)
        functions.create_folder(output_folder)

        for plot_data in plot_config_data['plots']:
            sensor = plot_data['name']
            channel = plot_data['plot_channel']
            drop_last = plot_data['drop_last']
            if sensors_data[sensor] is not None:
                x, y = self.get_plot_x_y(sample_data['SampleRate'], sensors_data[sensor][channel], drop_last)
                # print(x, y)
                plots.plot_config(x, y, plot_data, output_folder)

    def calculate_single_sensor_rms(self, sensor_data):
        """
        :param sensor_data: 传感器采集到的hdf5振动数据 如sensor1
        :return: 单个传感器的(channel0 1 2)与轴向和径向的rms字典和温度
        :return: rms数据字典
        """
        single_sensor_rms = {}
        data = sensor_data
        channel_num, r_rms, r_rms_num, z_rms, z_rms_num, temp = 1, 0, 0, 0, 0, 0
        for channel, value in data.items():
            # axis = functions.channel_to_axis(self.config['channels'], channel)
            axis = 'axis' + '_' + str(channel_num) + '_' + self.model_name  # axis_1_rms
            rms = calculate_rms(value[:-1])  # 计算rms
            temp += value[-1]
            single_sensor_rms[axis] = rms
            if self.config['channels'][channel_num - 1]['channel' + str(channel_num)] == 'r':
                r_rms += rms
                r_rms_num += 1
            elif self.config['channels'][channel_num - 1]['channel' + str(channel_num)] == 'z':
                z_rms = rms
                z_rms_num += 1
            channel_num += 1
        single_sensor_rms['r_' + self.model_name] = r_rms / r_rms_num
        single_sensor_rms['z_' + self.model_name] = z_rms / z_rms_num
        single_sensor_rms['temp'] = temp / (r_rms_num + z_rms_num)
        single_sensor_rms = self.default_single_sensor_template(single_sensor_rms, self.model_name)
        return single_sensor_rms

    def confirm_threshold_exceeded(self, single_sensor_rms, sensor_threshold):
        """
        确定rms与温度是否超出阈值，并且实现超出高级阈值会覆盖超出低级阈值
        :param single_sensor_rms: 计算得到的rms和温度
        :param sensor_threshold: 阈值
        :return:
        """
        for key, threshold in sensor_threshold.items():
            threshold = float(threshold)
            if key.split('_')[2] == 'temp':
                if single_sensor_rms['temp'] >= threshold and key.count('h') > single_sensor_rms['temp_alarm']:
                    single_sensor_rms['temp_alarm'] = key.count('h')
            else:
                sensor_index = key.split('_')[2] + "_" + self.model_name
                alarm_index = key.split('_')[2] + "_" + self.model_name + "_alarm"
                if single_sensor_rms[sensor_index] >= threshold and key.count('h') > single_sensor_rms[alarm_index]:
                    single_sensor_rms[alarm_index] = key.count('h')
        return single_sensor_rms

    def get_alarm(self, sensor, single_sensor_rms):
        """
        更新的alarm部分
        :param sensor:传感器名称
        :param single_sensor_rms:
        :return:
        """
        threshold_config, _, _ = functions.get_threshold_config(self.config['threshold_config_path'], self.name)
        # window, on, off = alarm_config['window'], alarm_config['on'], alarm_config['off']
        sensor_threshold = threshold_config['sensors_threshold'][sensor]
        single_sensor_rms = self.confirm_threshold_exceeded(single_sensor_rms, sensor_threshold)
        return single_sensor_rms

    def calculate_single_sensors_rms(self, sensors_data):
        """
        得到所有传感器的rms值和温度
        :return:
        """
        sensors_rms = {}
        # sample_data, data = functions.get_single_sensor_data(self.data_source, self.shot, self.name, sensor)
        for sensor in self.sensors:
            try:
                sensor_data = sensors_data[sensor]
                single_sensor_rms = self.calculate_single_sensor_rms(sensor_data)
                sensors_rms[sensor] = single_sensor_rms
            except Exception as e:
                # print(e)
                sensors_rms[sensor] = self.get_single_sensor_result(err=True)
        return sensors_rms

    def single_shot_sensors_summary(self, sensors_data):
        """
        将模型单独炮的数据分析结果汇总
        :return:
        """
        sensors = {}
        sensors_rms = self.calculate_single_sensors_rms(sensors_data)
        self.get_is_running(sensors_rms)
        for sensor, single_sensor_rms in sensors_rms.items():
            # 防止出现传感器出现故障，导致数据没有被采集的故障报错导致程序中断运行
            try:
                if self.is_running:
                    single_sensor_rms = self.get_alarm(sensor, single_sensor_rms)
                sensors[sensor] = single_sensor_rms
            except Exception as e:
                # print(e)
                sensors[sensor] = self.get_single_sensor_result(err=True)
        return sensors

    def single_shot_summary(self, sensors_data, sample_data):
        """
        汇总当前summary信息
        :return:
        """
        sensors = self.single_shot_sensors_summary(sensors_data)
        self.date_time = functions.get_sample_time(sample_data)
        single_shot_summary = functions.single_shot_summary(self.name, self.shot, sensors, self.date_time
                                                            , self.is_running)
        address = self.config['db']['connection']
        collection_name = self.config['db']['collection']
        functions.save_single_summary_mongodb(single_shot_summary, address, collection_name, self.shot,
                                              database_name=self.config['db']['db_name'])
        return single_shot_summary

    def get_config_start_rms(self):
        """
        返回配置文件中的判断是否运行的rms阈值，以及确定使用的传感器以及传感器的轴
        :return:
        """
        threshold_config, _, _ = functions.get_threshold_config(self.config['threshold_config_path'], self.name)
        return threshold_config['is_running']['min_rms_start']

    def get_is_running(self, sensors_rms):
        """
        得到该炮是否球磨机is_running的结果
        :param sensors_rms:
        :return:
        """
        try:
            threshold_config, _, _ = functions.get_threshold_config(self.config['threshold_config_path'], self.name)
            is_running_config = threshold_config['is_running_threshold']
            start_sensor, start_axis = is_running_config['start_sensor'], is_running_config['start_axis']
            min_rms_start = is_running_config['min_rms_start']
            is_running_results = []
            for index, start_sensor in enumerate(start_sensor):
                single_sensor_rms = sensors_rms[start_sensor]
                if single_sensor_rms[start_axis] >= float(min_rms_start[index]):
                    is_running_results.append(True)
                else:
                    is_running_results.append(False)
            self.is_running = any(is_running_results)
            # print(sensor, self.is_running)
        except Exception as e:
            # print(e)
            self.is_running = None

    @staticmethod
    def default_single_sensor_template(single_sensor_rms, model_name):
        """
        预设置所有的报警等级为0
        :param model_name: 模型名称
        :param single_sensor_rms:单个传感器的所有rms和温度值
        :return:
        """
        single_sensor_rms['r_' + model_name + "_alarm"] = 0
        single_sensor_rms['z_' + model_name + "_alarm"] = 0
        single_sensor_rms['temp_alarm'] = 0
        return single_sensor_rms

    @staticmethod
    def get_single_sensor_result(single_sensor_rms=None, err=False):
        """
        返回单个sensor计算rms的结果，如果sensor计算rms报错则返回各个结果为None的字典
        :param alarm:
        :param sensor: 传感器名称
        :param single_sensor_rms: 正常情况下，没有计算报错得到的sensor_rms数据
        :param err: 有没有发生计算故障 False为未发生故障
        :return: 单个sensor的result
        """
        sensor_result = {
            "axis_1_rms": None,
            "axis_2_rms": None,
            "axis_3_rms": None,
            "r_rms": None,
            "z_rms": None,
            "temp": None,
            "r_rms_alarm": None,  # 无警报，H=1,HH=2
            "z_rms_alarm": None,  # 1级警报
            "temp_alarm": None
        }
        if not err:
            sensor_result.update(single_sensor_rms)
        return sensor_result


def all_ball_mills_rms():
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    name = 'bm1'
    shot = '1108200'
    config = functions.read_config(config_path)
    # Rms(name, config_path, shot)
    # 得到bail_mill中的bail_name
    ball_names = [mill['name'] for mill in config['ball_mills']]
    for name in ball_names:
        Rms(name, config_path, shot)


def test_shots_calculate():
    # # 输入参数
    # config_path, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    config = functions.read_config(config_path)
    # 得到bail_mill中的bail_name
    bail_names = [mill['name'] for mill in config['ball_mills']]

    shots = np.arange(1108200, 1110400)
    for shot in shots:
        shot = str(shot)
        for name in bail_names:
            Rms(name, config_path, shot)


def main():
    # test_shots_calculate()
    all_ball_mills_rms()


if __name__ == '__main__':
    main()
