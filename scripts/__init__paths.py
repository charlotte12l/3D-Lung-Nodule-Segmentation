""" setup python path """
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path("/cluster/home/it_stu95/lxylib/")
print("add code root path (with `lxylib`).")
