import pandas as pd


def get_data_3c():
    filter_frame = pd.DataFrame()
    nofilter_frame = pd.DataFrame()
    try:
        with open("mp2_log_small.txt", "r") as f:
            filter_method_time = {}
            nofilter_method_time = {}
            lines = f.readlines()
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if words[0] != "Time":
                    continue
                words[2] = words[2][:-1]
                print(words)
                if words[1] == "filter":
                    filter_method_time[words[2]] = float(words[3])
                else:
                    nofilter_method_time[words[2]] = float(words[3])
        filter_frame = pd.DataFrame(filter_method_time, index=[0])
        nofilter_frame = pd.DataFrame(nofilter_method_time, index=[0])
        with open("mp2_log_medium.txt", "r") as f:
            filter_method_time = {}
            nofilter_method_time = {}
            lines = f.readlines()
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if words[0] != "Time":
                    continue
                words[2] = words[2][:-1]
                print(words)
                if words[1] == "filter":
                    filter_method_time[words[2]] = float(words[3])
                else:
                    nofilter_method_time[words[2]] = float(words[3])
        filter_frame = pd.concat(
            [filter_frame, pd.DataFrame(filter_method_time, index=[1])])
        nofilter_frame = pd.concat(
            [nofilter_frame, pd.DataFrame(nofilter_method_time, index=[1])])

        with open("mp2_log_large.txt", "r") as f:
            filter_method_time = {}
            nofilter_method_time = {}
            lines = f.readlines()
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if words[0] != "Time":
                    continue
                words[2] = words[2][:-1]
                print(words)
                if words[1] == "filter":
                    filter_method_time[words[2]] = float(words[3])
                else:
                    nofilter_method_time[words[2]] = float(words[3])
        filter_frame = pd.concat(
            [filter_frame, pd.DataFrame(filter_method_time, index=[2])])
        nofilter_frame = pd.concat(
            [nofilter_frame, pd.DataFrame(nofilter_method_time, index=[2])])
    except FileNotFoundError:
        pass
    return filter_frame, nofilter_frame


def get_data_4c():
    data_frame = pd.DataFrame()
    try:
        with open("4c_log_small.txt", "r") as f:
            filter_method_time = {}
            lines = f.readlines()
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if words[0] != "Time":
                    continue
                print(words)
                filter_method_time[words[1]] = float(words[5])
        data_frame = pd.DataFrame(filter_method_time, index=[0])
        with open("4c_log_medium.txt", "r") as f:
            filter_method_time = {}
            lines = f.readlines()
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if words[0] != "Time":
                    continue
                print(words)
                filter_method_time[words[1]] = float(words[5])
        data_frame = pd.concat(
            [data_frame, pd.DataFrame(filter_method_time, index=[1])])

        with open("4c_log_large.txt", "r") as f:
            filter_method_time = {}
            lines = f.readlines()
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if words[0] != "Time":
                    continue
                words[2] = words[2][:-1]
                print(words)
                filter_method_time[words[1]] = float(words[5])
        data_frame = pd.concat(
            [data_frame, pd.DataFrame(filter_method_time, index=[2])])
    except FileNotFoundError:
        pass
    return data_frame


def get_data_mttkrp():
    # maps each tensor name to a pandas frame. Each frame has the time taken by each method for each mode. Order is mode1, mode2, mode3
    tensor_frame_dict = {}
    for tname in ["nell-1", "nell-2", "flickr-3d", "vast-2015-mc1-3d"]:
        tensor_frame_dict[tname] = pd.DataFrame()

    try:
        with open("mttkrp_mode1.txt", "r") as f:
            lines = f.readlines()
            tensor_name = None
            running_frame = None
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if "Running" in words:
                    if running_frame is not None:
                        tensor_frame_dict[tensor_name] = pd.concat(
                            [tensor_frame_dict[tensor_name], running_frame])
                    try:
                        tensor_name = words[2].split(".")[0]
                        running_frame = pd.DataFrame({"const": -1}, index=[0])
                    except IndexError:
                        continue
                    continue
                if words[0] != "Time":
                    continue
                method_name = words[2][:-1]
                running_frame[method_name] = float(words[3])
            if running_frame is not None:
                tensor_frame_dict[tensor_name] = pd.concat(
                    [tensor_frame_dict[tensor_name], running_frame])

        with open("mttkrp_mode2.txt", "r") as f:
            lines = f.readlines()
            tensor_name = None
            running_frame = None
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if "Running" in words:
                    if running_frame is not None:
                        tensor_frame_dict[tensor_name] = pd.concat(
                            [tensor_frame_dict[tensor_name], running_frame])
                    try:
                        tensor_name = words[2].split(".")[0]
                        running_frame = pd.DataFrame({"const": -1}, index=[1])
                    except IndexError:
                        continue
                    continue
                if words[0] != "Time":
                    continue
                method_name = words[2][:-1]
                running_frame[method_name] = float(words[3])
            if running_frame is not None:
                tensor_frame_dict[tensor_name] = pd.concat(
                    [tensor_frame_dict[tensor_name], running_frame])

        with open("mttkrp_mode3.txt", "r") as f:
            lines = f.readlines()
            tensor_name = None
            running_frame = None
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if "Running" in words:
                    if running_frame is not None:
                        tensor_frame_dict[tensor_name] = pd.concat(
                            [tensor_frame_dict[tensor_name], running_frame])
                    try:
                        tensor_name = words[2].split(".")[0]
                        running_frame = pd.DataFrame({"const": -1}, index=[2])
                    except IndexError:
                        continue
                    continue
                if words[0] != "Time":
                    continue
                method_name = words[2][:-1]
                running_frame[method_name] = float(words[3])
            if running_frame is not None:
                tensor_frame_dict[tensor_name] = pd.concat(
                    [tensor_frame_dict[tensor_name], running_frame])

    except FileNotFoundError:
        pass
    return tensor_frame_dict


def get_data_ttmc():
    # maps each tensor name to a pandas frame. Each frame has the time taken by each method for each mode. Order is mode1, mode2, mode3
    tensor_frame_dict = {}
    for tname in ["nell-1", "nell-2", "flickr-3d", "vast-2015-mc1-3d"]:
        tensor_frame_dict[tname] = pd.DataFrame()

    try:
        with open("ttmc_mode1.txt", "r") as f:
            lines = f.readlines()
            tensor_name = None
            running_frame = None
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if "Running" in words:
                    if running_frame is not None:
                        tensor_frame_dict[tensor_name] = pd.concat(
                            [tensor_frame_dict[tensor_name], running_frame])
                    try:
                        tensor_name = words[2].split(".")[0]
                        running_frame = pd.DataFrame({"const": -1}, index=[0])
                    except IndexError:
                        continue
                    continue
                if words[0] != "Time":
                    continue
                method_name = words[2][:-1]
                running_frame[method_name] = float(words[3])
            if running_frame is not None:
                tensor_frame_dict[tensor_name] = pd.concat(
                    [tensor_frame_dict[tensor_name], running_frame])

        with open("ttmc_mode2.txt", "r") as f:
            lines = f.readlines()
            tensor_name = None
            running_frame = None
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if "Running" in words:
                    if running_frame is not None:
                        tensor_frame_dict[tensor_name] = pd.concat(
                            [tensor_frame_dict[tensor_name], running_frame])
                    try:
                        tensor_name = words[2].split(".")[0]
                        running_frame = pd.DataFrame({"const": -1}, index=[1])
                    except IndexError:
                        continue
                    continue
                if words[0] != "Time":
                    continue
                method_name = words[2][:-1]
                running_frame[method_name] = float(words[3])
            if running_frame is not None:
                tensor_frame_dict[tensor_name] = pd.concat(
                    [tensor_frame_dict[tensor_name], running_frame])

        with open("ttmc_mode3.txt", "r") as f:
            lines = f.readlines()
            tensor_name = None
            running_frame = None
            for line in lines:
                line = line.replace("_", " ")
                words = line.split()
                if "Running" in words:
                    if running_frame is not None:
                        tensor_frame_dict[tensor_name] = pd.concat(
                            [tensor_frame_dict[tensor_name], running_frame])
                    try:
                        tensor_name = words[2].split(".")[0]
                        running_frame = pd.DataFrame({"const": -1}, index=[2])
                    except IndexError:
                        continue
                    continue
                if words[0] != "Time":
                    continue
                method_name = words[2][:-1]
                running_frame[method_name] = float(words[3])
            if running_frame is not None:
                tensor_frame_dict[tensor_name] = pd.concat(
                    [tensor_frame_dict[tensor_name], running_frame])

    except FileNotFoundError:
        pass
    return tensor_frame_dict
