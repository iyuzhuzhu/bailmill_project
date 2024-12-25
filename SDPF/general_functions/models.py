from general_functions import functions
from general_functions.database_data import connect_mongodb_database


class BasicModel:
    def __init__(self, name, config_path, shot, model_name):
        self.date_time, self.is_running = None, None
        self.name, self.shot, self.model_name, self.config_path = name, shot, model_name, config_path
        self.config, self.ruamel_yaml = functions.load_yaml(self.config_path)
        self.data_source, self.output_path, self.sensors = (self.config['data_source'], self.config['Inference_path'],
                                                            self.config['sensors'])
        self.client, self.db = connect_mongodb_database(self.config['db']['connection'], self.config['db']['db_name'])
        self.channels = self.get_sensor_channels()

    def create_output_folder(self):
        output_path = functions.create_output_folder(self.config['Inference_path'], self.shot, self.name)
        return output_path

    def get_sensor_channels(self):
        channels = [list(item.keys())[0] for item in self.config['channels']]
        return channels
    # def get_is_running_raw_data(self):
