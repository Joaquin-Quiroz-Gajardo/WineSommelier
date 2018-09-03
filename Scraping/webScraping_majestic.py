#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 15:22:52 2018

@author: diveki
Creating classes for webscraping.
"""

import requests
import bs4
import pandas as pd
import re
import numpy as np
import pdb


def get_html(url):
    page = requests.get(url)
    page.raise_for_status()
    html = bs4.BeautifulSoup(page.text, 'html.parser')
    return html


def create_winelist_df(keySet, winelist, subDetail = 20):
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


class MajesticWineData:
    body_definition = {'1': 'dry', '2': 'off-dry', '3': 'medium', '4': 'sweet', 'A': 'light', 'B': 'medium', 'C': 'full'}
    def __init__(self, name, website):
        self.name = name
        self.website = website

    def get_mini_product_info(self, product):
        # get the abv
        info=product.find('div', attrs={'class':'product-info__symbol-label js-tooltip'})
        abv = info.text.replace('ABV', '').strip()
        self.abv = abv
        # get body
        info=product.find('div', attrs={'class':'js-symbols-tooltip product-info__symbol-icon product-info__symbol-icon--leaf'})
        if info != None:
            self.Body = MajesticWineData.body_definition.get(info.text)
        else:
            info=product.find('div', attrs={'class':'js-symbols-tooltip product-info__symbol-icon product-info__symbol-icon--pinot'})
            if info == None:
                self.Body = np.nan
            else:
                self.Body = MajesticWineData.body_definition.get(info.text)
        # get resume
        info = product.find('p')
        if info != None:
            self.resume = info.text.strip()
        else:
            self.resume = np.nan        
        
    def getDescription(self, soup):
        '''#soup is a beautifulsoup type created from an html reqest'''
        block = soup.find('p', attrs={'class':'product-content__description'})
        self.description = block.text.strip()
                
    def get_content_table(self, soup):
        table = soup.find('table', attrs={'class':'content-table'})
        attr = table.find_all('td', attrs={'class': 'content-table__title'})
        value = table.find_all('td', attrs={'class': 'content-table__text'})
        for atr, val in zip(attr, value):
            setattr(self, atr.text.lower(), val.text)



if __name__ == '__main__':
    page_num = 0
    data = []
    base = 'https://www.majestic.co.uk'
    # there are 17 pages as at 05/08/2018
    for page_num in range(17):
        print('Page: %s' % str(page_num+1))
        url = base + '/wine?pageNum=' + str(page_num) + '&pageSize=40'
        html = get_html(url)
        products = html.find_all('div', attrs={'class':"product-details"})
        for product in products:
            block = product.find('h3', attrs={'class':'space-b--none'})
            name = block.a.text
            website = base + block.a['href']
            wine = MajesticWineData(name, website)
            wine.get_mini_product_info(product)
            # visit the wines own page and get info from there
            html_wine = get_html(wine.website)
            wine.getDescription(html_wine)
            wine.get_content_table(html_wine)
            data.append(wine)
    
    # determine all possible attributes over all wines. These attributes will serve the column name of the dataframe and excelsheet
    keySet = set()
    for w in data:
        attrs = [x for x in dir(w) if x.startswith('_') == False and x.startswith('get') == False and x.endswith('_definition') == False]
        keySet = keySet.union(set(attrs))
    ## transform data into dataframe
    df = create_winelist_df(keySet, data)
    # save data into excel sheet
    save_name = 'raw_majestic_data.xlsx'
    df.to_excel(save_name)

    


