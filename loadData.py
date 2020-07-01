import sys
import os
import numpy as np
import time

def load_data(path):
    path_dir = path # CHANGE HERE!
    file_list = os.listdir(path_dir)
    file_list.sort()
    return file_list


# files means 00001, 00002, ``````


if __name__ == "__main__":
    print("Error.. Why loadData Module execute")





'''
###############concatenate method######################
array_1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
prev = array_1[0:3]
for i in range(1,int(len(array_1)/4)):
    prev = np.vstack([prev,array_1[4*i:4*i+3]])
print(prev)
'''