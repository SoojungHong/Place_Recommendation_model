#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""read_category_file.py                                                                                                              
Reads PDS category data in CSV files. The format/schema of the file
is 
<categoryId),<categoryName>,<Description> .
Each row is one category record. 

"""

from numpy import *
import sys
import csv

debug_flag = False

pds_category_file = "./PDS-categories.csv"
pds_category_system = "navteq-lcms"

pds_food_types_file = "./PDS-food-types.csv"
pds_food_types = "food_types"

food_types_with_ints = "./food-types-only.csv"


def load_category_data():
    category_dict = read_category_file(pds_category_file, pds_category_system)
    food_dict = read_category_file(pds_food_types_file, pds_food_types)
    return category_dict, food_dict


class category:
    id = None
    name  = ''
    description = ''
    numeric_id = 0
    type = "navteq-lcms" ## or "food-types"
    def __init__(self, id_, name_, description_, numeric_id_):
        self.id = id_
        self.name = name_
        self.description = description_
        self.numeric_id = numeric_id_
    def get_name(self):
        return self.name
    def show(self):
        print("\nid: "+str(self.id))
        print("\nname: "+str(self.name))
        print("\ndesc: "+str(self.description))
        print("\nnumeric: "+str(self.numeric_id))

def read_category_file (category_file, category_type="navteq-lcms", n_records=500):
    category_dict = {}
    count = 0
    row_list=[]
    with open(category_file,'rU') as csvin:
        csvin = csv.reader(csvin, delimiter=',') ##'\t')
        for row in csvin:
            if (debug_flag==True): print ("\n Row # "+str(count))
            row.append(count) # introduce a unique integer for each category.
            row_list.append(row)
            count = count + 1
            if count >= n_records: break
                
    for row in row_list[1:]: ## Skip the first row containing schema. 
        #print("\n"+str(row))
        if (debug_flag == True): print("\n"+str(row[0]+"  "+str(row[1])))
        category_id   = row[0]
        category_name = row[1]
        category_description = row[2]
        category_int  = row[3]
        this_category = category(category_id, category_name, category_description,category_int)
        this_category.type = category_type
        
        category_dict[category_id] = this_category
    if (debug_flag == True): print("\ncount = " + str(count))
    return category_dict


################################################################################################
### Reading food-types-only.csv for converting between category-id's and integers
################################################################################################
#
# Creates a dictionary that maps food-type id's to unique integers.
# Reads the food-types-only.csv file with schema:
#
#  Unique Int,Food Type Code,Food Type,Definition<ret>
#
#
# Reads this out of the Places Datastore Categories  20180312.xlsx file.
# And assign integers during the read.
# food_types_list =
# [(100,"101-000", "American", "description"),
#  (1,"101-001", "American-Californian", "description"),
#  (1,"101-002", "American-Southwestern", "description"),
#  (1,"101-003", "American-Barbecue/Southern", "description"),
#  ...]

def load_food_type_data():
    food_type_dict, food_data_dict = read_food_type_file(food_types_with_ints)
    return food_type_dict, food_data_dict


def read_food_type_file (food_type_file):
    food_type_dict = {}
    food_data_dict = {}
    count = 0
    row_list=[]
    with open(food_type_file,'rU') as csvin:
        csvin = csv.reader(csvin, delimiter=',') ##'\t')                                                                                                                                   
        for row in csvin:
            if (debug_flag==True): print ("\n Row # "+str(count))
            #row.append(count) # introduce a unique integer for each category.                                                                                                              
            row_list.append(row)
            count = count + 1


    for row in row_list[1:]: ## Skip the first row containing schema.                                                                                                                      
        #print("\n"+str(row))                                                                                                                                                              
        if (debug_flag == True): print("\n"+str(row[0]+"  "+str(row[1])))
        food_type_int = row[0]
        food_type_id  = row[1]
        food_type_name = row[2]
        food_type_description = row[3]

        food_type_dict[food_type_id] = int(food_type_int)
        food_data_dict[food_type_id] = (food_type_name, food_type_int, food_type_description)
    if (debug_flag == True): print("\ncount = " + str(count))
    return food_type_dict, food_data_dict


### Test reading in the food-types.

def experiment2():
    food_type_dict, food_data_dict = read_food_type_file("./food-types-only.csv")
    print("\n len(food_type_dict): "+ str(len(food_type_dict)))
    print("\n 101-004: " + str(food_data_dict.get("101-004")))



### Test reading in the category dictionary.

def experiment1():

    category_dict = read_category_file("./PDS-categories.csv", "navteq-lcms")
    food_dict = read_category_file("./PDS-food-types.csv", "food-types")

    print("\n len(category_dict): "+ str(len(category_dict)))
    print("\n len(food_dict): "+ str(len(food_dict)))

    category_dict.get("100-1000-0001").show()

    
if __name__ == "__main__":
    #experiment1()
    experiment2()
