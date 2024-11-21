from sklearn.cluster import KMeans
import numpy as np
from general_functions import functions
from matplotlib import pyplot as plt


def calculate_rms(data):
    # 计算rms
    rms = np.sqrt(np.mean(data ** 2))
    return rms


def return_web_is_running(single_summary_dict):
    """
    给网页发送is_running更新信息
    :param single_summary_dict:
    :return:
    """
    try:
        functions.update_web_data(single_summary_dict)
    except Exception as e:
        pass


class Rms:
    def __init__(self, name, config_path, shot, model_name='rms'):
        self.date_time, self.start_axis, self.start_sensor, self.is_running, self.temp = None, None, None, None, None
        self.name, self.shot, self.model_name, self.config_path = name, shot, model_name, config_path
        self.config, self.ruamel_yaml = functions.load_yaml(self.config_path)
        self.data_source, self.output_path, self.sensors = (self.config['data_source'], self.config['Inference_path'],
                                                            self.config['sensors'])
        for sensor in self.sensors:
            print(self.calculate_single_sensor_rms(sensor)[0])

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
        channel_num, r_rms, z_rms, temp = 1, 0, 0, 0
        for channel, value in data.items():
            # axis = functions.channel_to_axis(self.config['channels'], channel)
            axis = 'axis' + '_' + str(channel_num) + '_' + self.model_name  # axis_1_rms
            rms = calculate_rms(value[:-1])  # 计算rms
            temp += value[-1]
            single_sensor_rms[axis] = rms
            if self.config['channels'][channel_num - 1]['channel' + str(channel_num)] == 'r':
                r_rms += rms
            elif self.config['channels'][channel_num - 1]['channel' + str(channel_num)] == 'z':
                z_rms = rms
            channel_num += 1
        single_sensor_rms['r_' + self.model_name] = r_rms / 2
        single_sensor_rms['z_' + self.model_name] = z_rms
        single_sensor_rms['temp'] = temp / 3
        # self.single_sensor_rms = single_sensor_rms
        return single_sensor_rms, sample_data

    def calculate_single_sensors_rms(self):
        sensors_rms = {}
        for sensor in self.sensors:
            single_sensor_rms, sample_data = self.calculate_single_sensor_rms(sensor)
            sensors_rms[sensor] = single_sensor_rms
        return sensors_rms, sample_data

    def save_single_summary(self):
        """
        将模型单独炮的数据分析结果汇总
        :return:
        """
        results = []
        sensors_rms, _ = self.calculate_single_sensors_rms()
        self.get_is_running(sensors_rms)
        for sensor, single_sensor_rms in sensors_rms.items():
            # 防止出现传感器出现故障，导致数据没有被采集的故障报错导致程序中断运行
            try:
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

    def get_is_running(self, sensors_rms):
        """
        得到该炮是否is_running的结果
        :param sensors_rms:
        :return:
        """
        try:
            min_rms_start = self.get_config_start_rms()
            is_running_results = []
            for index, start_sensor in enumerate(self.start_sensor):
                single_sensor_rms = sensors_rms[start_sensor]
                if single_sensor_rms[self.start_axis] >= float(min_rms_start[index]):
                    is_running_results.append(True)
                else:
                    is_running_results.append(False)
            self.is_running = any(is_running_results)
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
        Rms(name, config_path, shot)


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
            Rms(name, config_path, shot)


def main():
    # sensor_rms_start(1108200, 1110400)
    # test_shots_calculate()
    all_ball_mills_rms()


if __name__ == '__main__':
    main()
