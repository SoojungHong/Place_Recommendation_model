#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import logging
import pandas as pd
import os
import requests
import xmltodict
import re
import math
from io import open
import simplejson as json
from os import listdir
from os.path import isfile, join
from read_category_file import *
from multiprocessing import Pool
import time


record_limit = -1 ## <1 means no limit
file_limit = -1   ## <1 means no limit
filter_for_food_type = True
category_dict = {}
food_dict = {}
food_type_to_int_dict = {}
debug_flag = False
ppid_missing_name_list = []
n_empty_query_string = 0
duplicate_cookie_dict = {}

### Return the longest name, max_name_, from a list of names, name_list.
def max_name(name_list):
    max_name_ = name_list[0]
    max_len = len(name_list[0])
    for each_name in name_list:
        each_len = len(each_name)
        if each_len > max_len:
            max_len = each_len
            max_name_ = each_name
    max_name_ = max_name_.replace('&amp;','&')
    return max_name_


### Convert the xml string containing the category information to a list of category tuples.
### Each tuple has the form (category_system, category_id, category_name).  Some tuples
### may have the form (category_system, category_id, category_name, category_integer),
### where the category_integer can be used for training a model.
def xml_to_categories(xml_string_of_categories):
    categories = []
    contains_restaurant_or_food_types_p = False
    if xml_string_of_categories=="": return categories, contains_restaurant_or_food_types_p
        
    xml_list = re.findall('categorySystem=\"[a-zA-Z]+[-[a-zA-Z]+]*\"><CategoryId>[\d]+[-[\d]+]*</CategoryId>',
                              xml_string_of_categories)
    for system_category_str in xml_list:
        category_system = re.findall("System=\"[a-zA-Z]+[-[a-zA-Z]+]*", system_category_str)
        category_system[0] = category_system[0][8:]
        category_id = re.findall("[\d]+[-[\d]+]*", system_category_str)
        if category_system[0] == "food-types":
            category_inst = food_dict.get(category_id[0], None)
        else:
            category_inst = category_dict.get(category_id[0], None)
        if category_inst == None:
            print("\nNo category instance for id, "+ str(category_id[0]))
            category_name="Missing Data"
        else:
            category_name = category_inst.get_name()
        if (category_system[0] == "food-types"):
            contains_restaurant_or_food_types_p = True
            #print("\ncategory_id[0] = "+str(category_id[0]) + " " +str(food_type_to_int_dict))
            category_int = food_type_to_int_dict.get(category_id[0], None)
            #print(" category_int  = "+str(category_int))
            cat_tuple = (category_system[0], category_id[0], category_name, category_int)
        else:
            cat_tuple = (category_system[0], category_id[0], category_name)
        categories.append(cat_tuple)
        if (category_id[0][0:4] == "100-"):
            #print ("\n Found restaurant category: "+str(category_id[0]))
            contains_restaurant_or_food_types_p = True
    return categories, contains_restaurant_or_food_types_p


### Returns a list of place names from the xml returned by the curl call in ppid_to_categories.
### example = '<BaseText default=\"true\" languageCode=\"de\" type=\"OFFICIAL\">Olympiastadion</BaseText>'

def xml_to_names(xml_string_of_names):
    names = []
    if xml_string_of_names=="": return names
    xml_list = re.findall('<BaseText[\sa-zA-Z\"=]*>[^<^>]+</BaseText>', xml_string_of_names)
    ##print(str(xml_list))
    for result in xml_list:
        begin_name = result.find(">")
        end_name = len(result) - 11 ## len("</BaseText>")
        if begin_name > 0 and (end_name > (begin_name + 1)):
            each_name = result[begin_name+1:end_name]
            names.append(each_name)
        else: print("\nxml_to_names: unable to extract name from, " + str(result))
    return names
    
    
def extract_category(result):
    begin_cat = result.find("<Category")
    end_cat = result.rfind("</Category")
    if begin_cat>0 and end_cat>begin_cat:
        cat_result = result[begin_cat:end_cat]
    else: cat_result=""
    return cat_result


def extract_name(result):
    begin_name = result.find("<NameList")
    end_name = result.rfind("</NameList>")
    if begin_name>0 and end_name>begin_name:
        name_result = result[begin_name:end_name]
    else: name_result=""
    return name_result


### Given a ppid, return a list of category tuples assigned to that ppid.
### Each category tuple (usually) has the form (category_system, category_id, category_name).
### Some tuples have a fourth field, which is a category_integer used for training a model.
def ppid_to_categories(ppid):
    header = {'X-Sdc-Access-Token': 'cd6f6508cde80088347563296cfc8d91'}
    url = 'http://jenkins-dt-nightly-frontends.dt.gate5.us/place/' + ppid
    result = requests.get(url, headers=header)
    response = str(result.content)
    cat_result = extract_category(response)
    categories, contains_food_types_p = xml_to_categories(cat_result)
    if (contains_food_types_p == True):
        name_result = extract_name(response)
        names = xml_to_names(name_result)
        if debug_flag == True: print("\nNames = " + str(names) + " ppid = " + str(ppid))
        if len(names) == 0: ppid_missing_name_list.append(ppid)
        return categories, contains_food_types_p, names
    return categories, contains_food_types_p, ['']


def almost_equal_ut (x_ut, y_ut): return (x_ut/100000) == (y_ut/100000)

### Reads the jsonfile that was generated by a pig script, and
### a. filters out empty query_string's
### b. filters out duplicate selections
###    where duplicate means same cookie, same ppid, same query_string and same timestamp


def clicks_to_training(jsonfile, outfile):
    global n_empty_query_string

    print("\nLoading data from " + jsonfile)
    df = pd.read_json(jsonfile, lines=True)
    global n_empty_query_string
    df = df.sort_values(by=['cookie', 'result_name', 'click_time', 'query_string'], ascending=[True, True, True, True])
    n_rows = len(df.index)

    df = df[df['cookie'] !=  None]
    df = df[df['cookie'] != "null"]

    #include only POI or RECO
    df = df[(df['result_provider'] == "POI") | (df['result_provider'] == "RECO")]

    #filter out empty strings
    rows_before = len(df.index)
    df = df[df['query_string'] != ""]
    n_empty_query_string += rows_before - len(df.index)

    #filter out duplicates (exact time)
    n_dupes_before = len(duplicate_cookie_dict.keys())
    mask = df.duplicated(subset=['cookie', 'result_name', 'query_string', 'click_time'], keep='first')
    duplicated = df[mask]
    df = df[~mask]

    duplicate_list = duplicated['cookie'].unique()
    if len(duplicate_list) > 0:
        for cookie in duplicate_list:
            x = duplicate_cookie_dict.get(cookie)
            if x == None:
                duplicate_cookie_dict[cookie] = 1
            else:
                duplicate_cookie_dict[cookie] = x + 1
    n_dupes_new = len(duplicate_cookie_dict.keys()) - n_dupes_before

    ppids = df['result_name'].unique()

    pool = Pool(20)
    mapped_results = pool.map(ppid_to_categories, ppids)
    pool.close()
    pool.join()

    listed_results = list(map(list, zip(*mapped_results)))

    categories = listed_results[0]
    is_food_type = listed_results[1]
    places = listed_results[2]

    places = [str(max_name(place)) for place in places]

    ppid_category = dict(zip(ppids, categories))
    df['category'] = df['result_name'].map(ppid_category)
    ppid_food_type = dict(zip(ppids, is_food_type))
    df['food_type'] = df['result_name'].map(ppid_food_type)
    ppid_places = dict(zip(ppids, places))
    df['places'] = df['result_name'].map(ppid_places)

    if filter_for_food_type:
        df = df[df['food_type'] == True]

    df[['cookie', 'result_name', 'result_provider',
        'click_time', 'query_string', 'query_language', 'category', 'places', 'result_lat',
        'result_lon', 'click_rank']].to_csv(outfile, sep='\t', header=False, index=False)

    print("\nn_dupes_new = " +str(n_dupes_new))
    print("\nn_rows = " + str(n_rows))
    print("\nDone. df.shape = " + str(df.shape))
    return n_rows


def read_click_files(click_directory, target_directory):
    click_files = [each_file for each_file in listdir(click_directory) if isfile(join(click_directory, each_file))]
    file_count = 0
    for each_file in click_files:
        if file_limit > 0 and file_count > file_limit: break
        print ("\nreading clicks from: "+str(click_directory)+str(each_file))
        outfile = str(each_file)+"_train.tsv"
        if target_directory == None: target_directory = click_directory
        clicks_to_training(str(click_directory)+str(each_file), str(target_directory)+str(outfile))
        file_count = file_count + 1

        
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Create collaborative filter training data from JSON entries from NULF.')
    parser.add_argument('--infile', default = "data/part-m-00382", type=str, help="JSON queries and clicks from NULF")
    parser.add_argument('--outfile', default = "some_training_data.tsv", type=str, help="intermediate form of training data")
    parser.add_argument('--indir', default = None, type=str, help="directory containing NULF data files in JSON format.")
    parser.add_argument('--outdir', default = None, type=str, help="directory to put click/category data files in TSV.")
    
    args = parser.parse_args()
    jsonfile = args.infile
    outfile = args.outfile
    jsonfile_dir = args.indir
    tsvfile_dir = args.outdir

    category_dict, food_dict = load_category_data()
    food_type_to_int_dict, food_data_dict = load_food_type_data()
    
    if (jsonfile_dir == None):
        clicks_to_training(jsonfile, outfile)
    else:
        ## go through the directory and process all the files.
        read_click_files(jsonfile_dir, tsvfile_dir)

    print("\n Failed to parse names of, "+str(len(ppid_missing_name_list)) +", ppids.")
    for each_ppid in ppid_missing_name_list:
        print("\n Failed to parse BaseText of ppid, "+str(each_ppid))
    print("\n n_empty_query_string = " +str(n_empty_query_string))
    print("\n n duplicate cookies found = " +str(len(duplicate_cookie_dict.keys())))
    for each_key in duplicate_cookie_dict.keys():
        print("\n  cookie: " + str(each_key) + " has " + str(duplicate_cookie_dict[each_key]) + " dupes.")