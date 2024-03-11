import pandas as pd
def get_data():
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
        filter_frame = pd.concat([filter_frame, pd.DataFrame(filter_method_time, index=[1])])
        nofilter_frame = pd.concat([nofilter_frame, pd.DataFrame(nofilter_method_time, index=[1])])
    
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
        filter_frame = filter_frame.concat(pd.DataFrame(filter_method_time, index=[2]))
        nofilter_frame = nofilter_frame.concat(pd.DataFrame(nofilter_method_time, index=[2]))
    except FileNotFoundError:
        pass
    return filter_frame, nofilter_frame
get_data()
