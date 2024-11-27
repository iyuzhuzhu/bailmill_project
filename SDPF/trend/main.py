from general_functions import functions, plots
from alarmSystem.Data.db.collectionDB import CollectionDB
from pymongo import MongoClient


def split_database_input_collection(input, separate_identifier):
    """
    将数据库输入进行切割
    :param input: 数据库查询输入 如bm1_rms>sensors.sensor1.r_rms
    :param separate_identifier: 划分input的collection的辨识符 如>
    :return: 集合名称，集合内要取出的数据
    """
    collection = input.split(separate_identifier)[0]
    input = input.split(separate_identifier)[1]
    return collection, input


class Trend:
    def __init__(self, config_path, name, shot, include_collection=True, separate_identifier='>', collection=None):
        self.config = functions.read_config(config_path)
        self.name = name
        self.shot = int(shot)
        self.separate_identifier = separate_identifier
        self.plot_trends()

    def get_y(self, db, y_input, output):
        y_input = y_input.replace('$bm$', self.name)
        collection, y_input = split_database_input_collection(y_input, self.separate_identifier)
        collection_db = CollectionDB(db, collection)
        y_input = collection_db.find_latest_n_records(output['point_max'], max_shot=self.shot)
        print(y_input)
        return y_input

    def get_x_y(self):
        pass

    def plot_trends(self):
        client = MongoClient(self.config['db']['connection'])
        db = client[self.config['db']['db_name']]
        for i, y_input in enumerate(self.config['inputs']):
            y = self.get_y(db, y_input, self.config['outputs'][i])


def main():
    # config_path, name, shot = functions.get_input_params('trend')
    config_path, name, shot = r'./config.yml', 'bm1', 1109200
    Trend(config_path, name, shot)


if __name__ == '__main__':
    main()
