import datetime
import hashlib
import os
import pickle
import logging
import random
import shutil
import time
import traceback
import numpy as np
import psutil
import pytz
from pathlib import Path


class IndexedDict:
    def __init__(self):
        self.ind = 0
        self.dict = dict()

    def get_index(self, key, show_add=False):
        add_key = False
        if key not in self.dict:
            self.dict[key] = self.ind
            self.ind += 1
            add_key = True
        if show_add:
            return self.dict[key], add_key
        else:
            return self.dict[key]


class Counter:
    def __init__(self):
        self.index = -1

    def get_index(self):
        self.index += 1
        return self.index


class Logger:
    @staticmethod
    def log(msg, color=None, style=None, new_line=True, title=True, write_file=None):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if len(logger.handlers) == 0:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        elif not isinstance(logger.handlers[0], logging.StreamHandler) or len(logger.handlers) > 1:
            while len(logger.handlers) > 0:
                logger.removeHandler(logger.handlers[0])  # empty exist handler
            handler = logging.StreamHandler()
            logger.addHandler(handler)

        handler = logger.handlers[0]
        if not new_line:
            handler.terminator = ""
        formatter = logging.Formatter(
            datefmt="",
            fmt="%(message)s"
        )
        handler.setFormatter(formatter)

        normal = "\033[0m"
        black = "\033[30m"
        red = "\033[31m"
        green = "\033[32m"
        yellow = "\033[33m"
        blue = "\033[34m"
        purple = "\033[35m"
        cyan = "\033[36m"
        grey = "\033[37m"

        bold = "\033[1m"
        italic = "\033[3m"
        underline = "\033[4m"
        invert = "\033[7m"
        remove = "\033[9m"

        msg = str(msg)

        if color == "blue":
            msg = blue + msg + normal
        elif color == "yellow":
            msg = yellow + msg + normal
        elif color == "black":
            msg = black + msg + normal
        elif color == "cyan":
            msg = cyan + msg + normal
        elif color == "grey":
            msg = grey + msg + normal
        elif color == "red":
            msg = red + msg + normal
        elif color == "green":
            msg = green + msg + normal
        elif color == "purple":
            msg = purple + msg + normal
        else:
            pass

        if style == "bold":
            msg = bold + msg + normal
        elif style == "underline":
            msg = underline + msg + normal
        elif style == "italic":
            msg = italic + msg + normal
        elif style == "invert":
            msg = invert + msg + normal
        elif style == "remove":
            msg = remove + msg + normal
        else:
            pass

        if not title:
            title = ""
        else:
            title_str = Logger.make_msg_title()
            title = cyan + title_str[0][0: 6] + yellow + title_str[0][6:] + blue + title_str[1] + green + title_str[2] + normal + " "
        logger.info(title + msg)
        if write_file is not None:
            Logger.write_log(msg=title + msg, path=write_file)

    @staticmethod
    def make_msg_title():
        """
        year+month+day+hour+minute+second+CPU+Memory+GPU+GMem
        :return:
        """
        cpu_utilization, mem_used, gpu_info_list, all_gpu_utilization, all_gpu_mem_used = \
            Logger.get_hardware_info()
        date = datetime.datetime.fromtimestamp(
            int(time.time()),
            pytz.timezone("Asia/Shanghai")
        ).strftime("%y%m%d%H%M%S")
        title = "{}|{:>2}{:>2}|{:>2}{:>2}".format(
            date, int(cpu_utilization), int(mem_used), int(all_gpu_utilization), int(all_gpu_mem_used)
        )

        return title.split("|")

    @staticmethod
    def get_hardware_info():
        # output system info
        try:
            with open("/proc/meminfo") as mem_file:
                total = int(mem_file.readline().split()[1])  # total memory
                free = int(mem_file.readline().split()[1])  # free memory
            mem_used = ((total - free) / total) * 100
            cpu_utilization = psutil.cpu_percent(0)
        except FileNotFoundError:
            cpu_utilization, mem_used = -1, -1

        try:
            import pynvml
            pynvml.nvmlInit()

            all_gpu_utilization = 0
            all_gpu_mem_used = 0
            gpu_info_list = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_utilization_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = (gpu_mem_info.used / gpu_mem_info.total) * 100
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                gpu_fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                gpu_power_max = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000  # milli-watts / 1000
                gpu_power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000

                gpu_info_list.append([i, gpu_utilization_rate.gpu, gpu_mem_used, gpu_temp,
                                      gpu_fan, gpu_power_usage, gpu_power_max])
                all_gpu_utilization += gpu_utilization_rate.gpu
                all_gpu_mem_used += gpu_mem_used

            all_gpu_utilization = all_gpu_utilization / pynvml.nvmlDeviceGetCount()
            all_gpu_mem_used = all_gpu_mem_used / pynvml.nvmlDeviceGetCount()

            return cpu_utilization, mem_used, gpu_info_list, all_gpu_utilization, all_gpu_mem_used
        except Exception:
            return cpu_utilization, mem_used, [], -1, -1

    @staticmethod
    def write_log(msg, path):
        file_name = str(path).split("/")[-1]
        directory = str(path)[0:(len(path) - len(file_name))]
        IO.make_dir(directory)
        with open(path, "a+") as log_file:
            log_file.write(msg + "\n")


class Funcs:
    @staticmethod
    def one_hot(indices, max_value):
        if indices >= max_value:
            return Funcs.one_hot(max_value - 1, max_value)
        one_hot = np.zeros(max_value)
        one_hot[indices] = 1
        return one_hot

    @staticmethod
    def rand_prob():
        return random.random()

    @staticmethod
    def print_obj(obj):
        s = "["
        for name in dir(obj):
            if name.startswith("_"):
                continue
            s += "{}:{}, ".format(name, getattr(obj, name))
        s = s[:-2] + "]"
        Logger.log(s)

    @staticmethod
    def trace_exception(write_file=None):
        exception_str = traceback.format_exc()
        Logger.log("Error msg: " + str(exception_str), write_file=write_file)

    @staticmethod
    def matrix_hashing(obs):
        """
        hashing function for matrix without collision
        :param obs: the sparse matrix
        :return: the hashed str
        """
        return hashlib.sha256(np.array(obs)).hexdigest()


class IO:
    @staticmethod
    def write_disk_dump(path, target_object):
        file_name = str(path).split("/")[-1]
        directory = str(path)[0:(len(path) - len(file_name))]
        IO.make_dir(directory)
        with open(path, "wb") as object_persistent:
            pickle.dump(target_object, object_persistent)

    @staticmethod
    def read_disk_dump(path):
        with open(path, "rb") as object_persistent:
            restore = pickle.load(object_persistent)
        return restore

    @staticmethod
    def make_dir(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy(src, dst):
        shutil.copytree(src, dst)

    @staticmethod
    def delete_dir(directory):
        shutil.rmtree(directory, True)

    @staticmethod
    def renew_dir(directory):
        IO.delete_dir(directory)
        IO.make_dir(directory)

    @staticmethod
    def delete_file(path):
        os.remove(path)

    @staticmethod
    def list_dir(directory):
        files_and_dirs = os.listdir(directory)
        return files_and_dirs

    @staticmethod
    def cascaded_list_dir(directory, ext_filter=None):
        files = []
        dirs = []
        # r = root, d = directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                if ext_filter is not None:
                    if ext_filter in file:
                        files.append(os.path.join(r, file))
                else:
                    files.append(os.path.join(r, file))
            for di in d:
                dirs.append(di)
        return files, dirs
