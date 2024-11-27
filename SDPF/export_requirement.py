import subprocess
import os


def generate_requirements(project_path):
    # 确保项目路径存在
    if not os.path.isdir(project_path):
        raise ValueError(f"The directory {project_path} does not exist")

    # 调用 pipreqs 命令生成 requirements.txt
    try:
        result = subprocess.run(['pipreqs', project_path], check=True, capture_output=True, text=True)
        print("Generated requirements.txt successfully")
        print("Output:", result.stdout)
        print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate requirements.txt: {e}")
        print("Output:", e.stdout)
        print("Errors:", e.stderr)
    except FileNotFoundError:
        print("pipreqs is not installed or not found in the system path")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# 示例用法
project_path = r'E:\ROS\Bailmill_project\Bailmill2\sdpf\code\SDPF'
generate_requirements(project_path)
