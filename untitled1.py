# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:58:18 2020

@author: suparji1969
"""
import pandas as pd
import numpy as np

my_file = open("caltech-lanes\lists\cordova1-list.txt", "r")
content = my_file.read()

content_list = content.split("\n")
my_file.close()
print(content_list)

list_file = np.asmatrix(pd.read_csv('caltech-lanes\lists\cordova1-list.txt',sep = "\n",header = None))