# %%
import subprocess
import argparse
import json
import yaml

CONFIG_PATH = r"task.yml"


def run_command(command_string):
    try:
        # Run the command in the terminal
        # process = subprocess.Popen(command_string, shell=True)
        process = subprocess.Popen(command_string)
        # Wait for the command to finish
        process.wait()
    except Exception as e:
        print(f"Error occurred: {e}")


def run_tasks(tasks, params):
    for i in range(len(tasks)):
        # replace the variables in the task with params from the -p argument
        # do error handling here
        # tasks[i][-1] = replace_placeholders(tasks[i][-1], params)
        for j, task_seq in enumerate(tasks[i]):
            tasks[i][j] = replace_placeholders(tasks[i][j], params)
        # print(params[params_keys[i]])
        # slave_params = ""
        # param_dict = json.loads(tasks[i][-1])
        # param_values = param_dict.values()
        # for index, param_value in enumerate(param_values):
        #     if index < len(param_values) - 1:
        #         slave_params += param_value + '\t'
        #     else:
        #         slave_params += param_value
        # tasks[i][-1] = slave_params
        cmd = tasks[i]
        # print(cmd)
        # run the task
        run_command(cmd)


# read tasks.yaml
def read_tasks(tasks_config_path):
    with open(tasks_config_path, 'r') as f:
        tasks = []
        task_data = yaml.safe_load(f)
        # print(task_data['Tasks'])
        # for task in task_data['Tasks']:
        for task_seq in task_data['Tasks']:
            task = list(task_seq.values())
            tasks.append(task)
        return tasks


def replace_placeholders(string, replacement_params):
    for key, value in replacement_params.items():
        placeholder_key = f"${key}$"
        string = string.replace(placeholder_key, value)
    return string


# %%


def main():
    # setup argument
    parser = argparse.ArgumentParser(description='A simple CLI program')
    parser.add_argument('config_path', type=str,
                        help='config path', default=CONFIG_PATH)
    parser.add_argument('--params', '-p', type=str,
                        help='json format string of a dictionary', default="")
    args = parser.parse_args()
    # read the tasks yaml file
    # get tasks
    tasks = read_tasks(args.config_path)
    # print(args.params)
    # get params
    params = json.loads(args.params)
    # for each task in tasks
    run_tasks(tasks, params)


if __name__ == "__main__":
    main()

# %%
