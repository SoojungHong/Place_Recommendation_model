# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:15:12 2018

@author: shong
"""

#====================
# import libraries
#====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from dateutil.relativedelta import relativedelta


#====================
# data and header
#====================
CHICAGO_DATA = 'C:/Users/shong/Documents/data/chicago1year_train_clean_newclients_dedupe_chain_center_click_restaurants_15112018.tsv'
CHICAGO_HEADER = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info', 'unknown_col1', 'unknown_col2']


#=============
# functions 
#=============
def readData(dataset, header):
    file_path = dataset
    headers = header
    data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)
    return data 


def initDataWithHeader(dataset, header): 
    data = readData(dataset, header)
    data['rating'] = 1
    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]
    
    return data


def convertToReadableDate(unix_time):
    result_ms = pd.to_datetime(unix_time, unit='ms')
    str(result_ms)
    return result_ms


def measureTimeDiff(cookie_idx, sorted_user_time_dataframe): 
    time_in_day = 0.0
    time_in_hour = 0.0 
    cookie_time = sorted_user_time_dataframe[sorted_user_time_dataframe['cookie_int'] == cookie_idx].time 
    first_click_time = cookie_time.min()
    last_click_time = cookie_time.max()
    diff = relativedelta(last_click_time, first_click_time)
    #print "The difference is %d year %d month %d days %d hours %d minutes" % (diff.years, diff.months, diff.days, diff.hours, diff.minutes)
    
    if(diff.months > 0) :
        print "The difference is %d year %d month %d days %d hours %d minutes" % (diff.years, diff.months, diff.days, diff.hours, diff.minutes)
        total_days = diff.days + (diff.months * 30)
        time_in_day = total_days
    elif(diff.months < 1) and (diff.days > 0):
        time_in_day = diff.days 
    elif(diff.days < 1) and (diff.hours > 0):
        time_in_day = np.float(diff.hours)/24 
    elif(diff.hours < 1) and (diff.minutes > 0): 
        time_in_day = np.float(diff.minutes)/1440
    time_in_hour = np.float(time_in_day) * 24    
    return time_in_day, time_in_hour 


def plotTimeDiffInDays(num_cookie): 
     objects = range(0, num_cookie)
     y_pos = np.arange(len(objects))
     performance = time_in_day_array[:num_cookie]

     plt.figure(figsize=(20,8))
     plt.bar(y_pos, performance, alpha=0.5)
     plt.xlabel('cookie_int')
     plt.ylabel('diff in days')
     plt.title('time difference in days')
 
     plt.show()
  

def plotTimeDiffInDaysHistogram(cookie_timediff, num_cookie): 
    print cookie_timediff
    plt.figure(figsize=(24, 20))
    x = cookie_timediff
    time_buckets = range(0,150,5)
    #plt.hist(x, normed=False, bins= time_buckets)
    plt.hist(x, normed=False, bins=[0,1,5,10,15,20,25,30,60,90,120,150])
    #plt.title("Histogram with time difference between first and last user click")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('number of users', fontsize=30)
    plt.xlabel('time diff in days', fontsize=30)
    plt.show()

   
#===============
# experiments 
#===============
data = initDataWithHeader(CHICAGO_DATA, CHICAGO_HEADER)
data['result_provider']
#data.describe()
#data['click_time']

cookie_time = ['cookie_int', 'click_time']
user_time_dataframe = data[cookie_time]
user_time_dataframe.sort_values(by=['cookie_int'])
user_time_dataframe['time'] = user_time_dataframe.apply(lambda row : convertToReadableDate(row.click_time), axis=1)
sorted_user_time_dataframe = user_time_dataframe.sort_values(by=['cookie_int', 'click_time', 'time']) 

time_in_day_array = []
time_in_hour_array = []
for i in range(5818): # num of unique cookie : 5818
    #time_in_day_array.append(measureTimeDiff(i, sorted_user_time_dataframe))
    day_tspan, hour_tspan = measureTimeDiff(i, sorted_user_time_dataframe)
    time_in_day_array.append(day_tspan)
    time_in_hour_array.append(hour_tspan)


len(time_in_day_array) #list 
time_in_hour_array
np.mean(time_in_day_array)
np.median(time_in_day_array)

np.mean(time_in_hour_array)
np.median(time_in_hour_array)


# bar chart
plotTimeDiffInDays(200) # bars with first 200 cookies 

# histogram
plotTimeDiffInDaysHistogram(time_in_day_array, 5818)

#---------------
# PBAPI 
#---------------

    
