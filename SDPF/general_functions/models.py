from general_functions import functions


class BasicModel:
    def __init__(self, name, config_path, shot, model_name):
        self.date_time, self.is_running = None, None
        self.name, self.shot, self.model_name, self.config_path = name, shot, model_name, config_path
        self.config, self.ruamel_yaml = functions.load_yaml(self.config_path)
        self.data_source, self.output_path, self.sensors = (self.config['data_source'], self.config['Inference_path'],
                                                            self.config['sensors'])

    def create_output_folder(self):
        output_path = functions.create_output_folder(self.config['Inference_path'], self.shot, self.name)
        return output_path
