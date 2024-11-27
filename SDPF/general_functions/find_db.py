from pymongo import MongoClient


def split_db_input_collection(input, separate_identifier):
    """
    将数据库输入进行切割
    :param input: 数据库查询输入 如bm1_rms>sensors.sensor1.r_rms
    :param separate_identifier: 划分input的collection的辨识符 如>
    :return: 集合名称，集合内要取出的数据
    """
    collection = input.split(separate_identifier)[0]
    input = input.split(separate_identifier)[1]
    return collection, input


class DatabaseFinder:
    def __init__(self, database_address, database_name, input, shot, shot_num, include_collection=True,
                 separate_identifier='>', collection=None):
        self.database_address, self.database_name, self.input = database_address, database_name, input
        self.input, self.include_collection, self.collection = input, include_collection, collection
        self.separate_identifier = separate_identifier

    def split_input_collection(self):
        if self.include_collection:
            self.collection, self.input = split_db_input_collection(self.input, self.separate_identifier)

    def find_data(self):
        client = MongoClient(self.database_address)
        # 选择数据库
        db = client[self.database_name]
        # 选择集合
        collection = db[self.collection]

        pass

    def get_collection_data(input, db, include_collection, separate_identifier='>', collection=None):
        pass


def get_collection_data(input, db, include_collection, separate_identifier='>', collection=None):
    pass
