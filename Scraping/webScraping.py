# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 16:34:04 2018

@author: zdiveki

This script connects to the bibendum-wine.co.uk website and collects information about their wines. Like name, producer, abv, description, food matching etc. 
The intention with this data is to clean it later on and use it as an input to a machine learning code that will determine the variety of the grape based on a given description.
"""

import requests
import bs4
import pandas as pd
import re
import numpy as np
import pdb

#searchTerm = 'alpro coconut milk'


#url = 'https://www.sainsburys.co.uk/webapp/wcs/stores/servlet/SearchDisplayView?langId=44&storeId=10151&searchTerm=' + '%20'.join(searchTerm.split(' ')) + '&beginIndex=0'

# initialize data container: name of wine and html
data = {'name': [], 'website': []}

# loop trough all the pages of the website
# there are 1700 items and on a page 30 is showed, so there is 57 pages
for pageNum in range(57):
    url = 'http://www.bibendum-wine.co.uk/shop?limit=30&p=' + str(pageNum+1) + '&product_type=4046'
    page = requests.get(url)
    page.raise_for_status()
    bs = bs4.BeautifulSoup(page.text, 'html.parser')
    # product names are stored in h2 headers
    for link in bs.find_all('h2'):
        # their webpage is within the header under an a href
        data['name'].append(link.text)
        data['website'].append(link.a['href'])



# save down to an excel the results
#df = pd.DataFrame(data)

class WineData:
    def __init__(self, name, website):
        self.name = name
        self.website = website
    
    def getDescription(self, soup):
        '''soup is a beautifulsoup type created from an html reqest'''
        self.description = soup.find_all('p', attrs={'class' : 'short-description std'})[0].text.strip()
        prod = soup.find_all('h3', text=re.compile(r'^Prod'))
        for tag in prod:
            tdesc = tag.findNext().text
            self.description = self.description + ' ' + tdesc
        
        
    def getMainAttributes(self, soup):
        div = soup.find('div', attrs={'class':'product-main-attrbutes'})
        divAttrs = div.select('li > strong')
        divValue = div.select('li > span')
        for ii in range(len(divAttrs)):
            atname = divAttrs[ii].text.lower().strip(':').split()
            atname = '_'.join(atname)
            atvalue = divValue[ii].text
            setattr(self, atname, atvalue)
        # get food match attribute
        food = soup.find('h3', text=re.compile(r'^Food'))
        if food:
            fdesc = food.findNext().text
            setattr(self, 'food_match', fdesc)
        # get producer name
        prod = soup.find('div', attrs={'class':'producer-information'})
        if prod:
            fdesc = prod.h3.span.text
            setattr(self, 'producer', fdesc)


def create_winelist_df(keySet, winelist):
    # initialize dictionary for dataframe usage
    data_dict = dict(zip(list(keySet), [[] for x in range(len(keySet))]))
    # step through the wine list and fill in the data into data_dict
    num = 0
    for w in winelist:
        if (num+1) % subDetail == 0:
            print('Dealing with wine #%s' % str(num+1))
        for key in data_dict.keys():
            #pdb.set_trace()
            try:
                value = getattr(w, key)
            except:
                value = np.nan
            data_dict[key].append(value)
        num += 1
    return pd.DataFrame(data_dict)

# visit each wine's website and collect information
new_data = []
subDetail = 50
for num, wp in enumerate(data['website']):
    if (num+1) % subDetail == 0:
        print('Getting details about wine #%s' % str(num+1))
    wine = WineData(data['name'][num], wp)
    page = requests.get(wp)        
    page.raise_for_status()
    bs = bs4.BeautifulSoup(page.text, 'html.parser')
    wine.getDescription(bs)
    wine.getMainAttributes(bs)
    new_data.append(wine)
    
# determine all possible attributes over all wines. These attributes will serve the column name of the dataframe and excelsheet
keySet = set()
for w in new_data:
    attrs = [x for x in dir(w) if x.startswith('_') == False and x.startswith('get') == False]
    keySet = keySet.union(set(attrs))

## transform data into dataframe
df = create_winelist_df(keySet, new_data)
# save data into excel sheet
save_name = 'raw_bibendum_data.xlsx'
df.to_excel(save_name)

#####################
## in another file we are going to clean the data




