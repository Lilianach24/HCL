import time
import os

""" prepare logdir for tensorboard and logging output"""
def set_log(output_dir, dataset_name):
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(output_dir, dataset_name + "-" + t)
    logs = {}
    for temp in ['model', 'logs']:
        temp_dir = os.path.join(log_dir, temp)
        os.makedirs(temp_dir, exist_ok=True)
        logs[temp] = temp_dir
    return logs