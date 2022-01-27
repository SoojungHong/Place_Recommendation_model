#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""read_training_file.py                                                                                                              
Reads a training TSV files. The format/schema of the file is 

cookie<tab>place_id<tab>result_provider<tab>click_time<tab>query_string<tab>query_language<tab>categories<tab>place_name<ret>.

Also generates 2 histograms: one based on users (i.e. cookies) and a second based on items (i.e. restaurants).
"""

from numpy import *
import sys
import csv
import io
debug_flag = False


def load_training_data(training_file="/data/passera/emr/chicago/train/chicago_train_7_16_18.tsv"):
    user_dict, item_dict  = read_training_file(training_file)
    return user_dict, item_dict



class training_record:
    cookie = None
    result_name  = None
    result_provider = None
    click_time = 0
    query_string = ''
    query_language = None
    categories = []
    place_name = ''

    def __init__(self, cookie_, result_name_, result_provider_, click_time_, query_string_, query_language_, categories_, place_name_):
        self.cookie = cookie_
        self.result_name = result_name_
        self.result_provider = result_provider_
        self.click_time = click_time_
        self.query_string = query_string_
        self.query_language = query_language_
        self.categories = categories_
        self.place_name = place_name_
    def get_ppid(self):
        return self.result_name
    def get_cookie(self):
        return self.cookie
    def get_place_name(self):
        return self.place_name
    def get_query_string(self):
        return self.query_string
    def get_categories(self):
        return self.categories
    def show(self):
        print("\ncookie: "+str(self.cookie))
        print("\nppid: "+str(self.result_name))
        print("\ncategories: "+str(self.categories))
        print("\nplace_name: "+str(self.place_name))


print_count = 0

### Reads the training file and creates user_dict and item_dict.

def read_training_file (training_file, n_records=-1):
    count = 0
    row_list=[]
    user_dict={}
    item_dict={}
    with open(training_file,'rU') as csvin:
        tsvin = csv.reader(csvin, delimiter='\t')
        for row in tsvin:
            #if (debug_flag==True): print ("\n Row # "+str(count))
            row_list.append(row)
            count = count + 1
            if n_records > 0  and count >= n_records: break
    if (debug_flag==True): print ("\n Row # "+str(count))
    count = 0
    for row in row_list:  #[1:]: ## Skip the first row containing schema. 
        #if check_row(row)==True: print("\n"+str(row)) : break
        if (len(row)<8):
            print("\nread_training_file: encountered short row: "+str(row)+" at position "+str(count))
            break
        if (debug_flag == True): print("\n"+str(count)+" "+str(row[0]+"  "+str(row[1])))
        this_training_record = training_record(row[0], row[1], row[2],row[3],row[4],row[5],row[6],row[7])
        this_user = row[0]
        this_item = row[1]
        
        if this_user not in user_dict:
            user_dict[this_user] = [this_training_record]
        else:
            training_list = user_dict.get(this_user)
            training_list.append(this_training_record)

        if this_item not in item_dict:
            item_dict[this_item] = [this_training_record]
        else:
            training_list = item_dict.get(this_item)
            training_list.append(this_training_record)
        count = count +1
    global print_count
    if (print_count == 0):
        print("\n Total rows of data read = " + str(count))
        print("\n Total number of distinct users, before  filtering: "+str(len(user_dict.keys())))
        print("\n Total number of distinct items, before  filtering: "+str(len(item_dict.keys())))
        print_count = print_count + 1
    return user_dict, item_dict



def update_histogram ( n, hist_dict ):
    count = hist_dict.get(n)
    if count == None:
        hist_dict[n] = 1
    else:
        hist_dict[n] = hist_dict[n] + 1
    return hist_dict


def histogram_of_value_counts(user_dict, hist_dict):
    # map over all queries
    user_keys = user_dict.keys()
    # retrieve each query_response object
    for each_key in user_keys:
        value_list = user_dict.get(each_key)
        update_histogram(len(value_list), hist_dict)


def show_histogram(hist_dict, print_data_p = True):
    hist_keys = hist_dict.keys()
    hist_keys.sort()
    results = []
    if print_data_p == True: print("\n")
    for each_key in hist_keys:
        val = hist_dict.get(each_key)
        results.append((each_key, val))
        if print_data_p == True: print (" (" + str(each_key) + ", " + str(val)+ ")")
    return results


### Returns a list of tuples (n_clicks, n_users) or (n_clicks, n_items) depending
### on the contents of the user_or_item_dict.

def make_and_show_histogram(user_or_item_dict, show_p = False):
    hist_dict= {}
    histogram_of_value_counts(user_or_item_dict, hist_dict)
    results = show_histogram(hist_dict, show_p)
    return results

### Load the 'training' data and return the data as two dictionaries, user_dict and item_dict.
### The user_dict has cookies as keys. The item_dict has ppids (aka result_name) as keys.
### The value of each hash table is a list of record objects, where each object is a copy
### of one row of the data in the source *.tsv file(s).
### Also, dumps the histogram data to the console.

def experiment1(data_file_with_categories):
    user_dict, item_dict = load_training_data(data_file_with_categories)

    print("\n Cookie histogram:")
    print("\nn_cookie_keys = "+str(len(user_dict)))
    print("\n (n_clicks, n_cookies)")
    make_and_show_histogram(user_dict, True)

    print("\n Restaurant histogram:")
    print("\nn_restaurant_keys = "+str(len(item_dict)))
    print("\n (n_clicks, n_places)")
    results = make_and_show_histogram(item_dict, True)

    # import matplotlib.pyplot as plt
    # _ = plt.hist(results, bins=20)
    # _ = plt.xlabel('number of cookies')
    # _ = plt.ylabel('number of clicks')
    # plt.show()

    return user_dict, item_dict


def get_cookie_ppid_combinations(train_file, n=5):
    user_dict, ppid_dict = load_training_data(train_file)
    cookies = user_dict.keys()
    ppids = ppid_dict.keys()
    cookiesXppids = []
    ppidsXcookies = []

    for each_cookie in cookies:
        obj_list = user_dict.get(each_cookie)
        if len(obj_list) >= n:
            for each_obj in obj_list:
                each_ppid = each_obj.get_ppid()
                cookiesXppids.append((each_cookie, each_ppid))

    for each_ppid in ppids:
        obj_list = ppid_dict.get(each_ppid)
        if len(obj_list) >= n:
            for each_obj in obj_list:
                each_cookie = each_obj.get_cookie()
                ppidsXcookies.append((each_ppid, each_cookie))

    return cookiesXppids, ppidsXcookies


def experiment2(train_file, n=5):
    cXp, pXc = get_cookie_ppid_combinations(train_file, n)

    print("\n cookies X ppids: ")
    print("\n len(cookies X ppids): " +str(len(cXp)))
    for each_pair in cXp:
        print("\n "+ str(each_pair))

    print("\n ppids X cookies: ")
    print("\n len(ppids X cookies): " +str(len(pXc)))
    for each_pair in pXc:
        print("\n "+ str(each_pair))            

### Returns (clicks_on_place, place_name, from_n_cookies)

def analyze_ppid_clicks1(data_file_with_categories, click_threshold=5):
    user_dict, item_dict = load_training_data(data_file_with_categories)
    results = []
    ###
    item_keys = item_dict.keys()
    for each_key in item_keys:
        each_value_list = item_dict.get(each_key)
        n = len(each_value_list)
        from_distinct_cookies = []
        if n >= click_threshold:
            for each_value in each_value_list:
                if not each_value.get_cookie() in from_distinct_cookies:
                    from_distinct_cookies.append(each_value.get_cookie())
            results.append((n, each_value_list[0].get_place_name(),len(from_distinct_cookies )))

    results = sorted(results, key=lambda x:x[0])
    return results


def cookie_has_n_ppids(cookie, user_dict, click_threshold=5):
    value_list = user_dict.get(cookie)
    if len(value_list) >= click_threshold:
        return True
    return False


def ppid_has_n_cookies(ppid, item_dict, click_threshold=5):
    value_list = item_dict.get(ppid)
    if len(value_list) >= click_threshold:
        return True
    return False

    
def analyze_ppid_clicks2(data_file_with_categories, ppid_click_threshold=5, cookie_click_threshold=5, verbose=False):
    user_dict, item_dict = load_training_data(data_file_with_categories)
    results = []
    n_positions_filled = 0
    remove_duplicate_cookies_p = False
    ### how many users are above threshold
    user_keys = user_dict.keys()
    n_users = 0
    for each_key in user_keys:
        each_value_list = user_dict.get(each_key)
        n = len(each_value_list)
        if n >= cookie_click_threshold:
            n_users = n_users + 1
    ### how many items are above threshold
    item_keys = item_dict.keys()
    n_items = 0
    for each_key in item_keys:
        each_value_list = item_dict.get(each_key)
        n = len(each_value_list)
        from_distinct_cookies = []
        if n >= ppid_click_threshold:
            n_items = n_items + 1
            for each_value in each_value_list:
                if cookie_has_n_ppids(each_value.get_cookie(), user_dict, cookie_click_threshold) == True:
                    from_distinct_cookies.append(each_value.get_cookie())
                    n_positions_filled = n_positions_filled +1
            if verbose == True: results.append((n, each_value_list[0].get_ppid(),from_distinct_cookies))
            else: results.append((n, each_value_list[0].get_place_name(),len(from_distinct_cookies)))

    results = sorted(results, key=lambda x:x[0])
    return results, n_users, n_items, n_positions_filled


    
def experiment3(data_file_with_categories, click_threshold=5):
    results = analyze_ppid_clicks1(data_file_with_categories, click_threshold)
    print("\n (clicks_to_place, place_name, from_cookies)")
    for each_result in results:
        print("\n "+str(each_result))
    return results


    
def experiment4(data_file_with_categories, ppid_click_threshold=5, cookie_click_threshold=5, verbose=False):
    results = analyze_ppid_clicks2(data_file_with_categories, ppid_click_threshold, cookie_click_threshold, verbose)
    print("\n (clicks_to_place, ppid, from_cookies)")
    for each_result in results:
        print("\n "+str(each_result))
    return results

        
### Given a list of users, and a list of items, and the user_dict and item_dict,
### determine which users clicked on which items. For each click add a 1 to the
### running total. This will tell us how many positions in a hypothetical user X item
### matrix will have data.

def compute_matrix_density(user_list, item_list, user_dict, item_dict):
    n_user = len(user_list)
    n_item = len(item_list)
    n_filled = 0.0
    ### which user clicked on which item?
    for each_user in user_list:
        for each_item in item_list:
            value_list = item_dict.get(each_item)
            for each_obj in value_list:
                if each_user == each_obj.get_cookie():
                    n_filled = n_filled + 1
                    break
    return n_filled/(n_user * n_item)



class density_result:
    density  = 0.0
    n_users  = 0
    n_items  = 0
    user_click_threshold = 1
    item_click_threshold = 5
    n_positions_filled = 0
    n_slots   = 0
    data_file = ''
    def __init__(self, density_, n_filled_, n_users_, n_items_, user_click_threshold_, item_click_threshold_, data_file_):
        self.density = density_
        self.n_filled = n_filled_
        self.n_users = n_users_
        self.n_items = n_items_
        self.user_click_threshold = user_click_threshold_
        self.item_click_threshold = item_click_threshold_
        self.data_file = data_file_
    def set_n_positions_filled(self, n):
        self.n_positions_filled = n
    def show(self):
        print(" -------------------------------------------")
        print(" density = " + str(self.density))
        print("\n n_filled = " +str(self.n_filled))
        print("\n n_users = " +str(self.n_users))
        print("\n n_items = " +str(self.n_items))
        print("\n ppid_click_threshold = "+str(self.item_click_threshold))
        print("\n cookie_click_threshold = "+str(self.user_click_threshold))
        print("\n data_file = "+str(self.data_file))



### Filter ppid's and cookies by thresholds.  Then collect the ppids and cookies and
### compute the density of the corresponding cookie x ppid matrix. This is a user x item matrix.

def select_training_items_for_density(data_file_with_categories, ppid_click_threshold=5, cookie_click_threshold=5):
    results, n_users, n_items, n_positions_filled = analyze_ppid_clicks2(data_file_with_categories, ppid_click_threshold, cookie_click_threshold, True)
    ## track how many ppid's and cookie's are used given the current thresholds
    ppid_dict = {}
    cookie_dict = {}
    
    ppid_list = []
    cookie_list = []
    n_filled = 0
    for each_result in results:
        each_ppid = each_result[1]
        ppid_list.append(each_ppid)
        mapped_cookies = each_result[2]
        for each_cookie in mapped_cookies:
            cookie_list.append(each_cookie)
            ## this ppid is an item in the user x item matrix
            ppid_dict[each_ppid] = 1
            ## this cookie is a user in the user x item matrix
            cookie_dict[each_cookie] = 1
            ## this location in the matrix is filled
            n_filled = n_filled +1
    ## get the dimensions of the user x item matrix given current thresholds        
    n_users = len(cookie_dict.keys())
    n_items = len(ppid_dict.keys())
    ## compute the density/scarcity of the user x item matrix
    n_slots = n_users * n_items
    density = float(n_filled)/float(n_slots)
    result_summary = density_result(density, n_filled, n_users, n_items, cookie_click_threshold, ppid_click_threshold, data_file_with_categories)
    result_summary.set_n_positions_filled(n_positions_filled)
    return results, result_summary


def experiment5(data_file_with_categories, ppid_click_threshold=5, cookie_click_threshold=1):
    results, summary = select_training_items_for_density(data_file_with_categories, ppid_click_threshold, cookie_click_threshold)
    summary.show()

    
def experiment6(data_file_with_categories):    
    ppid_threshold_list = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    cookie_threshold_list = [1, 2, 5, 10, 15, 20, 25, 30]

    count = 1
    for each_ppid_thresh in ppid_threshold_list:
        for each_cookie_thresh in cookie_threshold_list:
            results, summary = select_training_items_for_density(data_file_with_categories, each_ppid_thresh, each_cookie_thresh)
            print ("\n\n Experiment: "+str(count))
            summary.show()
            count = count + 1


### Experiment 7 filters out everything except restaurant queries containing type 100-1000-0000.
### Also assigns integers to food-types for training models.

from read_category_file import *

food_type_to_int_dict = {}

food_type_to_int_dict, food_data_dict = load_food_type_data()

def add_food_type_ints(category_list_as_string, food_type_to_int_dict = food_type_to_int_dict):
    new_category_list = []
    for each_tuple in eval(category_list_as_string):
        if each_tuple[0] == "food-types" and len(each_tuple) == 3:
            category_int = food_type_to_int_dict.get(each_tuple[1], None)
            #print(" category_int  = "+str(category_int))
            new_tuple = (each_tuple[0], each_tuple[1], each_tuple[2], category_int)
            new_category_list.append(new_tuple)
        else:
            new_category_list.append(each_tuple)
    return str(new_category_list)

### For filter_training_file only.
def write_output_file(outfilestream, cookie, result_name, result_provider, click_time, query_string,
                              query_language, categories, place_name, result_lat, result_lon, click_rank):
    #outfilestream.write(unicode(cookie)+"\t"+unicode(result_name)+"\t"+str(result_provider)+"\t"+str(click_time)+"\t\'"+unicode(query_string, encoding='utf8')+"\'\t"+str(query_language)+"\t"+str(categories)+"\t"+unicode(place_name, encoding='utf8', errors='ignore')+"\t"+str(result_lat)+"\t"+str(result_lon)+"\t"+str(click_rank)+"\n")
    outfilestream.write(unicode(cookie)+"\t"+unicode(result_name)+"\t"+str(result_provider)+"\t"+str(click_time)+"\t"+unicode(query_string, encoding='utf8')+"\t"+str(query_language)+"\t"+str(categories)+"\t"+unicode(place_name, encoding='utf8', errors='ignore')+"\t"+str(result_lat)+"\t"+str(result_lon)+"\t"+str(click_rank)+"\n")

### For filter_training_file only.        
def row_has_restaurant_p(categories):
    if categories.find('100-1000-0000') >= 0:
        return True
    return False

### Filter for records containing category "100-1000-0000".
def filter_training_file (training_file, output_file, n_records=-1):
    count = 0
    row_list=[]
    output_list= []
    user_dict={}
    item_dict={}
    with open(training_file,'rU') as csvin:
        tsvin = csv.reader(csvin, delimiter='\t')
        for row in tsvin:
            if (debug_flag==True): print ("\n Row # "+str(count)+ " : "+str(row))
            row_list.append(row)
            count = count + 1
            if n_records > 0  and count >= n_records: break
    if (debug_flag==True): print ("\n Row # "+str(count))
    count = 0
    with io.open(output_file, 'w', encoding="utf-8") as outfilestream:
        for row in row_list:  #[1:]: ## Skip the first row containing schema. 
            if (len(row)<8):
                print("\nfilter_training_file: encountered short row: "+str(row)+" at position "+str(count))
                break
            if (debug_flag == True): print("\n"+str(count)+" "+str(row[0]+"  "+str(row[1])))
            if row_has_restaurant_p(row[6])== True:
                ## add food_type_integer to food_type tuple
                row_6 = add_food_type_ints(row[6])
                if len(row)>8:
                    write_output_file(outfilestream, row[0],row[1],row[2],row[3],row[4],row[5],row_6,row[7],row[8],row[9],row[10])
                else:
                    write_output_file(outfilestream, row[0],row[1],row[2],row[3],row[4],row[5],row_6_,row[7],"","","")
            count = count +1

### filters for restaurants: 100-1000-0000 and breaks for short records when found.            
def experiment7(data_file_with_categories, filtered_file_with_restaurants):
    filter_training_file(data_file_with_categories, filtered_file_with_restaurants, -1)
    print("\nDone")

            
if __name__ == "__main__":

        #experiment1("/data/passera/emr/chicago1year/train/chicago1year_train.tsv")

        #experiment2("/data/passera/emr/chicago1year/train/chicago1year_train.tsv", 5)
      
        #experiment3("/data/passera/emr/chicago1year/train/chicago1year_train.tsv", 5)

        #experiment4("/data/passera/emr/chicago/train/chicago_train_3month.tsv", 5, 1)
        
        #experiment4("/data/passera/emr/chicago1year/train/chicago1year_train.tsv", 5, 1)

        #experiment4("/data/passera/emr/chicago2year/train/chicago2year_train.tsv", 5, 5)

        #experiment3("/data/passera/emr/chicago1year/train/chicago1year_train.tsv", 5)

        #experiment4("/data/passera/emr/chicago1year/train/chicago1year_train.tsv", 5, 5)

        #experiment5("/data/passera/emr/chicago1year/train/chicago1year_train.tsv", 25, 5)
        
        #experiment5("/data/passera/emr/chicago2year/train/chicago2year_train.tsv", 50, 10)

        #experiment5("/data/passera/emr/chicago2years/train/chicago2yearscombined_train.tsv", 60, 20)
        
        #experiment5("/data/passera/emr/chicago2year/train/chicago2year_train.tsv", 50, 5)

        #experiment6("/data/passera/emr/chicago1year/train/chicago1year_train.tsv")

        ### This gives the density data dumped to the console, which can be directed to a file.
        #experiment6("/data/passera/emr/berlin1year/train/save/berlin1year_train_clean.tsv")

        #experiment1("/data/passera/emr/berlin1year/train/save/berlin1year_train_clean.tsv")

        #experiment6("/data/passera/emr/berlin1year2x/train/save/berlin1year2x_train_clean.tsv")

        ### quick test
        #experiment7("/data/passera/emr/chicago/train/save/chicago_3month", "/data/passera/emr/chicago/train/save/chicago_3month_filtered.tsv")

        #experiment7("/data/passera/emr/berlin1year/train/save/berlin1year_train_clean.tsv", "/data/passera/emr/berlin1year/train/save/berlin1year_train_clean_restaurants.tsv")

        #experiment7("/data/passera/emr/chicago1year/train/save/chicago1year_train.tsv", "/data/passera/emr/chicago1year/train/save/chicago1year_train_restaurants.tsv")
        
        #experiment7("/data/passera/emr/newyork1year/train/save/newyork1year_train_clean.tsv", "/data/passera/emr/newyork1year/train/save/newyork1year_train_clean_restaurants.tsv")

        experiment7("/data/passera/emr/berlin1year2x/train/save/berlin1year2x_train_clean.tsv", "/data/passera/emr/berlin1year2x/train/save/berlin1year2x_train_clean_restaurants.tsv")

        experiment7("/data/passera/emr/newyork1year2x/train/save/newyork1year2x_train_clean.tsv", "/data/passera/emr/newyork1year2x/train/save/newyork1year2x_train_clean_restaurants.tsv")
        
        experiment7("/data/passera/emr/berlin1year2x/train/save/berlin1year2x_train_clean_newclients.tsv", "/data/passera/emr/berlin1year2x/train/save/berlin1year2x_train_clean_newclients_restaurants.tsv")

        #experiment6("/data/passera/emr/berlin1year2x/train/save/berlin1year2x_train_clean_newclients_restaurants.tsv")

        experiment7("/data/passera/emr/newyork1year2x/train/save/newyork1year2x_train_clean_newclients.tsv", "/data/passera/emr/newyork1year2x/train/save/newyork1year2x_train_clean_newclients_restaurants.tsv")
        
