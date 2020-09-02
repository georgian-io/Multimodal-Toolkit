from os import makedirs
from os.path import dirname, abspath, join, exists
import re


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return join(get_root_path(), 'datasets')


def get_log_path():
    return join(get_log_path(), 'logs')


def create_dir_if_not_exists(folder):
    if not exists(folder):
        makedirs(folder)


def get_args_info_as_str(config_flags):
    rtn = []
    d = vars(config_flags)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        if type(v) is dict:
            for k2, v2 in v.items():
                s = '{0:26} : {1}'.format(k + '_' + k2, v2)
                rtn.append(s)
        else:
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
    return '\n'.join(rtn)


def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn