import os.path
from general_functions import functions, plots, database_data
from general_functions.database_data import DatabaseFinder


def split_input_collection(input, separate_identifier='>'):
    """
    将数据库输入进行切割
    :param input: 数据库查询输入 如bm1_rms>sensors.sensor1.r_rms
    :param separate_identifier: 划分input的collection的辨识符 如>
    :return: 集合名称，集合内要取出的数据对应的关键字
    """
    input_collection = functions.split_string(input, separate_identifier)
    collection = input_collection[0]
    input = input_collection[1]
    return collection, input


class Trend:
    def __init__(self, config_path, name, shot, include_collection=True,
                 collection_separate_identifier='>', input_separate_identifier='.',
                 collection=None):
        self.config = functions.read_config(config_path)
        self.name = name
        self.shot = int(shot)
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
        data = DatabaseFinder(data_key, self.shot, shot_num, db, collection_name, self.input_separate_identifier).data
        return data

    def get_x_y(self, input, shot_num, db):
        """
        根据输入的字符串取出趋势图的x或者y数据
        """
        input = functions.replace_ball_mill_name(input, self.name)
        data = self.get_data(input, shot_num, db)
        return data

    def plot_trend(self, x_input, db, y_input, output, save_folder_path):
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
        x_data = self.get_x_y(x_input, point_max, db)
        y_data = self.get_x_y(y_input, point_max, db)
        # print(len(y_data))
        if len(y_data) >= point_min:  # 不小于绘制趋势图最小的需求点数，才绘制趋势图
            plots.plot_trend(x_data, y_data, x_label, y_label, title, save_path, linewidth=line_width, color=line_color,
                             back_color=back_color, is_x_time=True, x_rotation=-10)

    def plot_trends(self):
        """
        绘制所有的趋势图
        """
        client, db = self.connect_mongodb_database()
        save_folder_path = self.create_save_folder()
        self.copy_rename_plot_desc(save_folder_path)
        for i, y_input in enumerate(self.config['inputs']):
            x_input = self.config['x_input']
            output = self.config['outputs'][i]
            self.plot_trend(x_input, db, y_input, output, save_folder_path)
        client.close()

    def create_save_folder(self):
        """
        将配置文件中的保存文件夹地址替换并创建对应文件夹
        :return:
        """
        save_folder_path = self.config['output_base_path']
        save_folder_path = functions.replace_ball_mill_name(save_folder_path, self.name)
        save_folder_path = functions.replace_shot_100(save_folder_path, str(self.shot))
        functions.create_folder(save_folder_path)
        return save_folder_path

    def copy_rename_plot_desc(self, save_path):
        plots.copy_and_rename_file(self.config['gui_config_file'], save_path, self.config['plot_desc_file'])


def main():
    config_path, name, shot = functions.get_input_params('trend')
    # config_path, name, shot = r'./config.yml', 'bm1', 1110400
    Trend(config_path, name, shot)


if __name__ == '__main__':
    main()
