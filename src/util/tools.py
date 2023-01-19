import datetime
import os
import pickle
import logging
import shutil
import time
import traceback
from src.util.imports.numpy import np
import hashlib
import psutil
import pytz
from pathlib import Path
import subprocess
import signal 
# import rich


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


class LinkedListElement:
    def __init__(self, data):
        self.prev = None
        self.next = None
        self.data = data

    def get_entire_list(self):
        head = self
        backtraced = set()
        while True:
            backtraced.add(head.data)
            if head.prev is not None:
                if head.prev.data in backtraced:  # loop
                    break
                else:
                    head = head.prev
            else:
                break
        data_list = []
        while True:
            data_list.append(head.data)
            if head.next is not None:
                if head.next.data in data_list:  # loop
                    break
                else:
                    head = head.next
            else:
                break
            
        return data_list


class Counter:
    def __init__(self):
        self.index = -1

    def get_index(self):
        self.index += 1
        return self.index


class Logger:
    path = None

    @staticmethod
    def error(msg):
        Logger.log(msg, color="cyan")

    @staticmethod
    def log(raw_msg, color=None, style=None, new_line=True, make_title=True):
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
        else:
            handler.terminator = "\n"
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

        msg = str(raw_msg)

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
        
        if make_title:
            title_str = Logger.make_msg_title()
            title = cyan + title_str[0][0: 6] + yellow + title_str[0][6:] + blue + title_str[1] + green + title_str[2] + normal + " "
            raw_title = title_str[0][0: 6] + title_str[0][6:] + title_str[1] + title_str[2] + " "
        else:
            title_str = ""
            title = ""
            raw_title = ""

        logger.info(title + msg)
        if Logger.path is not None:
            Logger.write_log(msg=f"{raw_title} {raw_msg}", path=Logger.path)

    @staticmethod
    def make_msg_title():
        """
        year+month+day+hour+minute+second+CPU+Memory+GPU+GMem
        :return:
        """
        cpu_utilization, mem_used, gpu_info_list, all_gpu_utilization, all_gpu_mem_used = \
            Logger.get_hardware_info()
        date = Logger.get_date()
        title = "{}|{:>3}{:>2}|{:>3}{:>2}".format(
            date, int(cpu_utilization), int(mem_used), int(all_gpu_utilization), int(all_gpu_mem_used)
        )

        return title.split("|")

    @staticmethod
    def get_date():
        return datetime.datetime.fromtimestamp(
            int(time.time()),
            pytz.timezone("Asia/Shanghai")
        ).strftime("%y%m%d%H%M%S")

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
        return np.random.random()

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
    def trace_exception():
        exception_str = traceback.format_exc()
        Logger.error("Error msg: " + str(exception_str))

    @staticmethod
    def matrix_hashing(obs, type="sha256"):
        """
        hashing function for matrix without collision
        :param obs: the sparse matrix
        :return: the hashed str
        """
        if type == "sha256":
            return hashlib.sha256(np.array(obs)).hexdigest()
        if type == "shake_256":
            return hashlib.shake_256(np.array(obs)).hexdigest(length=32)
        if type == "md5":
            return hashlib.md5(np.array(obs)).hexdigest()
        if type == "sha384":
            return hashlib.sha384(np.array(obs)).hexdigest()
    
    @staticmethod
    def run_cmd(command, timeout=60):
        cmd = command.split(" ") 
        start = datetime.datetime.now() 
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        while process.poll() is None: 
            time.sleep(0.2) 
            now = datetime.datetime.now() 
            if (now - start).seconds > timeout: 
                os.kill(process.pid, signal.SIGKILL) 
                os.waitpid(-1, os.WNOHANG) 
                return None 
        return process.stdout.readlines() 



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
    def file_exist(path):
        return Path(path).exists()

    @staticmethod
    def read_file(path):
        content = []
        with open(path, "r") as file:
            for line in file:
                content.append(line.replace('\n', ''))
        return content

    @staticmethod
    def stick_read_disk_dump(path):
        while True:
            try:
                return IO.read_disk_dump(path)
            except Exception:
                time.sleep(1)

    @staticmethod
    def make_dir(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy(src, dst):
        shutil.copytree(src, dst)

    @staticmethod
    def delete_dir(directory):
        shutil.rmtree(path=directory, ignore_errors=True)

    @staticmethod
    def renew_dir(directory):
        IO.delete_dir(directory)
        IO.make_dir(directory)

    @staticmethod
    def move_file(src, dst):
        shutil.move(src, dst)

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
