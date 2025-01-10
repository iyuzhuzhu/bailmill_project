import os.path
from ai.ae_model import LstmAutoencoder
from general_functions import functions, database_data
from general_functions.models import BasicModel
import pandas as pd
import torch
from ai.Train import train_model, preprocessing_training_data, plot_history, create_train_model
from ai.Inference import predict
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


class Ai(BasicModel):
    def __init__(self, name, config_path, shot, model_name='ai'):
        super().__init__(name, config_path, shot, model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.train_models()
        self.ai()
        # self.predict_single_axis_loss()

    def ai(self):
        if self.config['is_training']:
            self.train_models()
        sample_data, sensors_data = functions.get_sensors_data(self.data_source, self.shot, self.name, self.sensors)
        self.predict_single_shot(sensors_data)

    def get_training_data(self):
        is_running_collection = functions.replace_ball_mill_name(self.config['db']['is_running_collection'], self.name)
        # print(is_running_collection, self.shot)
        running_shots_data = functions.get_is_running_shots_data(self.db, is_running_collection,
                                                                 self.config['training_shots'],
                                                                 int(self.shot), self.data_source, self.name,
                                                                 self.sensors,
                                                                 self.channels)
        # train_data, val_data, test_data = preprocessing_training_data(shots_data)
        return running_shots_data

    def get_save_model_path(self, sensor, channel):
        """
        得到model的完整保存路径（包括模型的文件名称）
        """
        model_name = functions.replace_string(self.config['model_name'], channel, 'channel')
        model_path = self.config['model_path']
        model_path = functions.replace_ball_mill_name(model_path, self.name)
        model_path = functions.replace_sensor(model_path, sensor)
        # print(model_path)
        functions.create_folder(model_path)
        folder_path = model_path
        model_path = Path(model_path) / model_name
        # print(model_path)
        return model_path, folder_path

    def train_models(self):
        """
        训练针对于球磨机的每个sensor的每个channel分别训练对应模型
        :return:
        """
        running_shots_data = self.get_training_data()
        for sensor in self.sensors:
            for channel in self.channels:
                train_data, val_data, test_data = preprocessing_training_data(running_shots_data[sensor][channel])
                model_path, folder_path = self.get_save_model_path(sensor, channel)
                if self.config['is_training']:
                    create_train_model(train_data, val_data, model_path, folder_path)
                # predictions, losses = self.predict_single_axis_loss(test_data, sensor, channel)
                # print(len(predictions[0]), losses[0])
                # print(test_losses)

    def load_model(self, sensor, channel):
        """
        加载对应sensor的channel的模型
        """
        model_path, folder_path = self.get_save_model_path(sensor, channel)
        return torch.load(model_path)

    def predict_single_axis_loss(self, data, sensor, channel, drop_last=True):
        model = self.load_model(sensor, channel)
        predictions, losses = predict(model, data, self.device)
        # print(predictions, losses)
        return predictions, losses

    def predict_single_shot(self, sensors_data):
        for sensor in self.sensors:
            for channel in self.channels:
                predictions, losses = self.predict_single_axis_loss(sensors_data[sensor][channel], sensor, channel)
                print(predictions[0])


def main():
    # test_shots_calculate()
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('ai')
    config_path = './config.yml'
    name = 'bm1'
    shot = '1110400'
    Ai(name, config_path, shot)


if __name__ == "__main__":
    main()
