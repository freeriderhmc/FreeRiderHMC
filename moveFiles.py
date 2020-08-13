import os
import shutil


src = r'D:/test/test_lidar/'
src_files = os.listdir(src)
#print(src_files)
folder = 0
cnt = 0
#directory = r"D:/test/test_lidar/"
    
path = r'D:/test/{}/lidar'.format(folder)
#os.mkdir(path)    
    

for file_name in src_files:    
    cnt +=1
    print(directory+file_name)
    shutil.move(path+file_name, directory+file_name)
    
    if cnt == 126:
        cnt = 0
        folder +=1
        path = r'D:/test/{}/lidar'.format(folder)
        #os.mkdir(path)
