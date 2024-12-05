from alarmSystem.Data.db.collectionDB import CollectionDB
from general_functions.functions import split_string
from pymongo import MongoClient


class DatabaseFinder(CollectionDB):
    def __init__(self, input, shot, shot_num, db, collection_name, separate_identifier='.'):
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
        self.data = self.get_data()
        # print(self.data)

    def find_documentation_list(self):
        documentation_list = super().find_latest_n_records(self.shot_num, max_shot=self.shot)
        # print(documentation_list)
        return documentation_list

    def from_documentation_get_data(self, documentation):
        data = documentation
        for key in self.input_list:
            data = data[key]
        return data

    def get_data(self):
        data_list = []
        documentation_list = self.find_documentation_list()
        for documentation in documentation_list:
            data = self.from_documentation_get_data(documentation)
            data_list.append(data)
        return data_list


if __name__ == '__main__':
    db_config = {
        "connection": "mongodb://localhost:27017/",
        "db_name": "bm",
        "collection": "bm1_rms"
    }
    # 输入
    input_query = "sensors.sensor1.r_rms"
    client = MongoClient(db_config["connection"])
    db = client[db_config["db_name"]]
    database_finder = DatabaseFinder(input_query, 1108215, 5, db, "bm1_rms")
    print(database_finder.data)
