import os.path
from sklearn.cluster import KMeans
import numpy as np
from general_functions import functions
from matplotlib import pyplot as plt


def calculate_rms(data):
    # 计算rms
    rms = np.sqrt(np.mean(data ** 2))
    return rms


def sensor_rms_start(shot_start, shot_end, sensor='sensor1', config_path='./config.yml', rms_axis='rms_x'):
    # # 输入参数
    # config_path, shot = functions.get_input_params('rms')
    shots = np.arange(shot_start, shot_end)
    config = functions.read_config(config_path)
    # 得到bail_mill中的bail_name
    bail_names = [mill['name'] for mill in config['ball_mills']]
    rms_start_dict = {name: [] for name in bail_names}
    for shot in shots:
        for name in bail_names:
            try:
                rms_class = Rms(name, config, str(shot))
                sensor1_rms = rms_class.calculate_single_sensor_rms(sensor='sensor1')
                rms_start_dict[name].append(sensor1_rms[rms_axis])
            except:
                continue
    for name in bail_names:
        summary_path = config['summary_path'].replace('$bm$', name)
        file_name = name + '_' + 'rms' + '_' + 'start' + '.png'
        save_path = os.path.join(summary_path, file_name)
        # print(rms_start_dict[name])
        rms_start_dict[name].sort()
        plt.plot(rms_start_dict[name])
        plt.title("RMS Sort Chart")
        plt.ylabel(rms_axis)
        plt.savefig(save_path)
        plt.close()
        # 将数据转换为二维数组，以便 K-means 可以处理
        data = np.array(rms_start_dict[name]).reshape(-1, 1)
        # 使用 K-means 聚类
        kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0).fit(data)
        # 获取每个簇的中心
        centers = kmeans.cluster_centers_
        # 排序中心值
        sorted_centers = sorted(centers.flatten())
        # 计算两个中心之间的中点作为阈值
        threshold = (sorted_centers[0] + sorted_centers[1]) / 2
        # 输出结果
        print("Threshold:", name, threshold)


class Rms:
    def __init__(self, name, config, shot, model_name='rms'):
        self.date_time, self.start_axis, self.start_sensor, self.is_running = None, None, None, None
        self.name, self.config, self.shot, self.model_name = name, config, shot, model_name
        self.data_source, self.output_path, self.sensors = (self.config['data_source'], self.config['Inference_path'],
                                                            self.config['sensors'])

        # self.train_rms_model()
        self.save_single_summary()

    def create_training_data_summary(self):
        """
        返回训练数据汇总的字典模板
        :return:
        """
        training_data_summary = {}
        for sensor in self.sensors:
            training_data_summary[sensor] = {
                "rms_mean": [],
                "rms_x": [],
                "rms_y": [],
                "rms_z": [],
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
        while training_shot_num < shot_num:
            # 报错时,跳过报错的那炮数据(如遇到采集到的有效炮数小于配置文件中的训练炮数，终止循环)
            try:
                # 训练从前一炮的数据开始取
                training_shot = str(int(self.shot) - training_shot_num - 1 - not_running_shot_num)
                single_summary_data = functions.return_single_summary_data(self.config['Inference_path'], training_shot,
                                                                           self.name, self.model_name)
                if single_summary_data['is_running']:
                    training_data_summary = self.single_data_training_summary(single_summary_data['results'],
                                                                              training_data_summary)
                    training_shot_num += 1
                else:
                    not_running_shot_num += 1
            except Exception as e:
                # print(e)
                training_shot_num += 1
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
        alarm_config, ruamel_yaml, alarm_config_path = functions.get_alarm_config(self.config['alarm_config_path'],
                                                                                  self.name)
        minor, major, fatal = alarm_config['minor'], alarm_config['major'], alarm_config['fatal']
        training_data_summary = self.get_training_data_summary()
        sensors_rms_threshold = []
        for sensor in training_data_summary.keys():
            try:
                sensor_threshold = {'sensor_prefix': sensor}
                for rms_axis, rms_list in training_data_summary[sensor].items():
                    sensor_threshold = functions.alarm_config_axis(rms_axis, rms_list, minor, major, fatal,
                                                                   sensor_threshold)
                # print(sensor_threshold)
            except Exception as e:
                # print(e)
                for sensor_threshold_ in alarm_config['sensors_threshold']:
                    if sensor_threshold_['sensor_prefix'] == sensor:
                        sensor_threshold = sensor_threshold_
            sensor_threshold = functions.order_alarm_dict(sensor_threshold, 6)
            sensors_rms_threshold.append(sensor_threshold)
        alarm_config['sensors_threshold'] = sensors_rms_threshold
        functions.save_yaml(alarm_config, alarm_config_path, ruamel_yaml)
        # print(sensors_rms_threshold)

    @staticmethod
    def single_data_training_summary(single_summary_data_results, training_data_summary):
        """
        将球磨机正在运行时采集到的数据导入训练数据
        :param single_summary_data_results: single_summary json数据中的results部分
        :param training_data_summary: 训练数据汇总
        :return: 训练数据汇总
        """
        for result in single_summary_data_results:
            for key in training_data_summary[result['sensor']].keys():
                if result[key] is not None:
                    training_data_summary[result['sensor']][key].append(result[key])
                else:
                    continue
        return training_data_summary

    def calculate_single_sensor_rms(self, sensor, drop_last=True):
        """
        :param sensor: 传感器名称 如sensor1
        :return: 单个传感器的x,y,z(channel0 1 2)的rms字典
        :param sensor:传感器 如sensor1 sensor2
        :param drop_last: 确定是否剔除最后一个温度点
        :return: rms数据字典
        """
        single_sensor_rms = {}
        sample_data, data = functions.get_single_sensor_data(self.data_source, self.shot, self.name, sensor)
        self.date_time = sample_data['CreateTime']
        # print(self.date_time)
        for channel, value in data.items():
            axis = functions.channel_to_axis(self.config['channels'], channel)
            axis = self.model_name + '_' + axis
            if drop_last:
                rms = calculate_rms(value[:-1])
            else:
                rms = calculate_rms(value)
            single_sensor_rms[axis] = rms
        rms_mean = sum(single_sensor_rms.values()) / len(single_sensor_rms)
        single_sensor_rms[self.model_name + '_' + 'mean'] = rms_mean
        # self.single_sensor_rms = single_sensor_rms
        return single_sensor_rms, sample_data

    def get_alarm(self, sensor, single_sensor_rms):
        """
        得到result的alarm部分
        :param sensor:
        :param single_sensor_rms:
        :return:
        """
        alarm_config, _, _ = functions.get_alarm_config(self.config['alarm_config_path'], self.name)
        window, on, off = alarm_config['window'], alarm_config['on'], alarm_config['off']
        sensor_threshold = functions.get_sensor_alarm_threshold(alarm_config['sensors_threshold'], sensor)
        last_summary = self.get_training_data_summary(window - 1, used_training=False)
        last_sensor_summary = last_summary[sensor]
        for key in last_sensor_summary.keys():
            last_sensor_summary[key].append(single_sensor_rms[key])
        alarm = functions.single_sensor_alarm(self.config['Inference_path'], self.shot, self.name,
                                              self.model_name, last_sensor_summary, sensor_threshold, window, on, off)
        return {'alarm': alarm}

    def save_single_summary(self):
        """
        将模型单独炮的数据分析结果汇总
        :return:
        """
        results = []
        for sensor in self.config['sensors']:
            # 防止出现传感器出现故障，导致数据没有被采集的故障报错导致程序中断运行
            try:
                single_sensor_rms, sample_data = self.calculate_single_sensor_rms(sensor)
                self.get_is_running(sensor, single_sensor_rms)
                if self.is_running:
                    alarm = self.get_alarm(sensor, single_sensor_rms)
                else:
                    alarm = {'alarm': []}
                sensor_result = self.get_single_sensor_result(sensor, single_sensor_rms, alarm)
                results.append(sensor_result)
            except Exception as e:
                # print(e)
                sensor_result = self.get_single_sensor_result(sensor, err=True)
                results.append(sensor_result)
        single_summary_dict = functions.single_summary_save(self.name, self.shot, self.date_time,
                                                            results, self.config['Inference_path'], self.model_name,
                                                            is_running=self.is_running)
        functions.update_web_data(single_summary_dict)
        return single_summary_dict

    def get_config_start_rms(self):
        """
        返回配置文件中的判断是否运行的rms阈值，以及确定使用的传感器以及传感器的轴
        :return:
        """
        for ball_mill in self.config['ball_mills']:
            if ball_mill['name'] == self.name:
                self.start_axis = ball_mill['start_axis']
                self.start_sensor = ball_mill['start_sensor']
                return ball_mill['min_rms_start']
            else:
                continue

    def get_is_running(self, sensor, single_sensor_rms):
        try:
            min_rms_start = self.get_config_start_rms()
            if sensor == self.start_sensor:
                if single_sensor_rms[self.start_axis] >= min_rms_start:
                    self.is_running = True
                else:
                    self.is_running = False
            # print(sensor, self.is_running)
        except Exception as e:
            # print(e)
            self.is_running = None

    @staticmethod
    def get_single_sensor_result(sensor, single_sensor_rms=None, alarm=None, err=False):
        """
        返回单个sensor计算rms的结果，如果sensor计算rms报错则返回各个结果为None的字典
        :param alarm:
        :param sensor: 传感器名称
        :param single_sensor_rms: 正常情况下，没有计算报错得到的sensor_rms数据
        :param err: 有没有发生计算故障 False为未发生故障
        :return: 单个sensor的result
        """
        sensor_result = {
            "sensor": sensor,
            "alarm": [],
            "rms_mean": None,
            "rms_x": None,
            "rms_y": None,
            "rms_z": None,
        }
        if not err:
            sensor_result.update(single_sensor_rms)
            sensor_result.update(alarm)
        return sensor_result


def all_ball_mills_rms():
    # # 输入参数
    # config_path, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    shot = '1109201'
    config = functions.read_config(config_path)
    # 得到bail_mill中的bail_name
    bail_names = [mill['name'] for mill in config['ball_mills']]
    for name in bail_names:
        Rms(name, config, shot)


def test_shots_calculate():
    # # 输入参数
    # config_path, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    config = functions.read_config(config_path)
    # 得到bail_mill中的bail_name
    bail_names = [mill['name'] for mill in config['ball_mills']]

    shots = np.arange(1109200, 1109300)
    for shot in shots:
        shot = str(shot)
        for name in bail_names:
            Rms(name, config, shot)


def main():
    # sensor_rms_start(1108200, 1110400)
    test_shots_calculate()
    # all_ball_mills_rms()


if __name__ == '__main__':
    main()
