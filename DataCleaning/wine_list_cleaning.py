# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:06:22 2018

@author: zdiveki

The purpuse of this code is to load in the excel sheet created from scraping a wine webpage. The excel sheet live in ../Scraping/raw_bibendum_data.xlsx .

The main task is to unify the grape variety names and clean the descriptions from unwanted content (just like the name of the grape itself).

The description will be used to predict the type of the grape.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
import re
import pdb
import unidecode

def get_body_from_description(text):
    body = re.compile(r'(light|medium|full)(-?|\s?)bod*?')
    try:
        result = body.findall(text)
    except:
        return np.nan            
    if result:
        return result[0][0]
    else:
        return np.nan


## read in the excel sheet
path = '../Scraping/raw_bibendum_data.xlsx'
df = pd.read_excel(path)

# investigate the data
df.describe()

# For starters we need abv, colour, country, description, food_match, grape_variety, name, producer, vintage
# TO DO write summary on columns stats
cols = ['abv', 'colour', 'country', 'description', 'food_match', 'grape_variety', 'name', 'producer', 'vintage']
df_sel = df[cols]
df_sel['description'] = df_sel['description'].str.lower()
# create a body column that is extracted from the description
df_sel['Body'] = df_sel.apply(lambda row: get_body_from_description(row['description']), axis=1)


# Our main input is description and output is grape_variety, so we make sure they do not have nan entries
df_sel.dropna(subset=['grape_variety', 'description', 'abv', 'Body'], inplace = True)



# TO DO write summary on columns stats

all_grapes = pd.unique([grape for x in df['grape_variety'].dropna().str.lower().tolist() for grape in x.split(', ')])
## save down to text file all the grape names
with open('all_grape_names.txt', 'w') as f:
    for g in all_grapes:
        g = unidecode.unidecode(g)
        f.write(g + '\n')
    f.close()

# unify the names in grape variety
# we decompose the wines into individual grapes and try to find unified nomenclature, like shiraz and syrah describe the same grape
def correct_grape_names(row):
    regexp = [r'shiraz', r'ugni blanc', r'cinsaut', r'carinyena', r'^ribolla$', r'palomino', r'turbiana', r'verdelho', r'viura', r'pinot bianco|weissburgunder', r'garganega|grecanico', r'moscatel', r'moscato', r'melon de bourgogne', r'trajadura|trincadeira', r'cannonau|garnacha', r'grauburgunder|pinot grigio', r'pinot noir|pinot nero', r'colorino', r'mataro|monastrell', r'mourv(\w+)']
    grapename = ['syrah', 'trebbiano', 'cinsault', 'carignan', 'ribolla gialla', 'palomino','verdicchio', 'verdejo','macabeo', 'pinot blanc', 'garganega', 'muscatel', 'muscat', 'muscadet', 'treixadura', 'grenache', 'pinot gris', 'pinot noir', 'lambrusco', 'mourvedre', 'mourvedre']
    f = row
    for exsearch, gname in zip(regexp, grapename):
        f = re.sub(exsearch, gname, f)
    return f

df_sel['grape_variety'] = df_sel['grape_variety'].str.lower()
df_sel['grape_variety'] = df_sel['grape_variety'].apply(lambda row: correct_grape_names(row))
grapes = df_sel['grape_variety'].tolist()

# in case we do not want to blend grapes
grape = [grape for x in grapes if len(x.split(','))==1 for grape in x.split(', ')]
fdist=nltk.FreqDist(grape)

# get the frequency distribution of the grapes
# print it out and check the names. Find which grapes are referring to the same ones: like shiraz=syrah, ugni blanc = trebbiano etc. There may be mispelling as well like cinsaut is cinsault. Write functions that corrigate these names

# check the distribution of the 10 most common grapes after correcting for names
fdist.plot(10)
# it turns out chardonnay, pinot noir, syrah, sauvignon blanc, cabernet sauvignon, grenache, merlot, pinot meunier are the most frequent ones. 
# in non blended scenario the 4 most frequent grape is chardonnay, pinot noir, sauvignon blanc and syrah

#subset the 4 grape types we want to study and which are the most representative
list_of_grapes = ['chardonnay', 'pinot noir', 'sauvignon blanc', 'syrah', 'sangiovese', 'cabernet sauvignon']
df_small = df_sel.loc[df_sel['grape_variety'].isin(list_of_grapes)]

###########################################
## remove the % from alcohol level and turn them into floats
def clean_abv(row):
    row = row.strip('%')
    return(float(row))

df_small.loc[:, 'abv'] = df_small.loc[:, 'abv'].apply(lambda row: clean_abv(row))

###########################################
## cleaning the description column
def removeHints(a, b, term=''):
    x = a.lower().replace(b,term).strip()
    return x

df_small.dropna(subset=['colour'], inplace = True)

    
df_small['vintagelessName'] = df_small.apply(lambda row: removeHints(row['name'].lower(), row['vintage'], term=''), axis=1)

df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], row['vintagelessName'].lower(), term='wine'), axis=1)

df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], row['grape_variety'], term = 'wine'), axis=1)

df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], '100%', term = ''), axis=1)

df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], row['colour'].lower()+' wine', term = 'wine'), axis=1)

for g in all_grapes:
    df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], g, term = 'wine'), axis=1)

df_small.to_excel('cleaned_wine_list_with_body.xlsx')









