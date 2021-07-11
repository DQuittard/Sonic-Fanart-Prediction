# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 15:56:17 2021

@author: Kabey
"""

import os
import platform
import pandas as pd
#from deviantart_dl import download_img
from data.deviantart_dl import download_img
import zipfile
import time

def zip_generator(df,main_dir,inf,sup):

    if platform.system()=="Windows":
        img_path = main_dir + 'img\\'
    elif platform.system()=="Linux":
        img_path = main_dir + 'img/'    


    files = os.listdir(img_path)
    
    file_number = len(files)
    
    for i in range(0,file_number):
        os.remove(os.path.join(img_path,files[i]))
        #print(files[i] + ' has been deleted.')

    download_img(df, main_dir, inf, sup)
    
    files = os.listdir(img_path)
    
    file_number = len(files)
    
    my_zipfile = zipfile.ZipFile(img_path + "sonic_images.zip", mode='w', compression=zipfile.ZIP_DEFLATED)
    for i in range(0,file_number):
        my_zipfile.write(img_path+files[i],os.path.basename(img_path+files[i]))
    
    
    my_zipfile.close()

'''
if platform.system()=="Windows":
    main_path = os.getcwd()+'\\'
elif platform.system()=="Linux":
    main_path = os.getcwd()+'/'
    
main_dir = main_path

index = main_dir.rfind('data')

main_dir = main_dir[0:index]

df = pd.read_csv(main_path+"dash_table.csv")

inf = 878
sup = 880

zip_generator(df,main_dir,inf,sup)
'''