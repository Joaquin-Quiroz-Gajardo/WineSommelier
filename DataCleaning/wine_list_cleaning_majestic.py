# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:06:22 2018

@author: zdiveki

The purpuse of this code is to load in the excel sheet created from scraping a wine webpage. The excel sheet live in ../Scraping/raw_mjestic_data.xlsx .

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

# read in grape names from text
all_grapes = []
with open('all_grape_names.txt', 'r') as f:
    for line in f.readlines():
        all_grapes.append(line.strip())


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


def get_variety_from_name(name, grape, grp_names):
    if type(grape) == str:
        return grape
    else:
        try:
            tmp = [x for x in grp_names if x in name]
            if tmp == []:
                return np.nan
            else:
                return tmp[0]
        except:
            return np.nan
        

def unidecode_text(text):
    try:
        #pdb.set_trace()
        text = unidecode.unidecode(text)
    except:
        pass
    return text

def compare_resume_with_description (descr, res):
    try:
        if res not in descr:
            descr = descr + ' ' + res
    finally:
        return descr
        
            
    
    
## read in the excel sheet
path = '../Scraping/raw_majestic_data.xlsx'
df = pd.read_excel(path)

# investigate the data
df.describe()

# For starters we need abv, colour, country, description, food_match, grape_variety, name, producer, vintage
# TO DO write summary on columns stats
# these will be the column names that we want to have
cols_needed = ['abv', 'colour', 'country', 'description', 'food_match', 'grape_variety', 'name', 'producer', 'vintage']
cols = ['abv', 'type', 'country', 'description', 'resume', 'grape', 'name', 'style', 'Body']
df_sel = df[cols]

# change text into lower case
for col in col_text:
    df_sel[col] = df_sel[col].str.lower()

# remove any unicode accents from text columns
for col in col_text:
    df_sel[col] = df_sel.apply(lambda row: unidecode_text(row[col]), axis=1)


# extract grape varieties from the name and description column
df_sel['grape'] = df_sel.apply(lambda row: get_variety_from_name(row['name'], row['grape'], all_grapes), axis=1)
df_sel['grape'] = df_sel.apply(lambda row: get_variety_from_name(row['description'], row['grape'], all_grapes), axis=1)

# check if resume has something to add to description, if yes, add it to description
df_sel['description'] = df_sel.apply(lambda row: compare_resume_with_description(row['description'], row['resume']), axis=1)

# rename columns of dataframe
cols_needed = {'type': 'colour', 'grape': 'grape_variety'}
df_sel.rename(index=int, columns=cols_needed, inplace=True)

# Our main input is description and output is grape_variety, so we make sure they do not have nan entries
df_sel.dropna(subset=['grape_variety', 'description', 'abv', 'Body'], inplace = True)


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

###########################################
## remove the % from alcohol level and turn them into floats
def clean_abv(row):
    row = row.strip('%')
    return(float(row))

df_small = df_sel.copy()
df_small.loc[:, 'abv'] = df_small.loc[:, 'abv'].apply(lambda row: clean_abv(row))

###########################################
## cleaning the description column
def removeHints(a, b, term=''):
    x = a.lower().replace(b,term).strip()
    return x

def remove_hints_from_description(desc, text):
    d_token = nltk.word_tokenize(desc)
    d = set(d_token)
    t = set(nltk.word_tokenize(text))
    its = list(d.intersection(t))
    tmp = [x for x in d_token if x not in its]
    tmp = ' '.join(tmp)
    return tmp
    
    
    
    
## remove wine from colour
df_small['colour'] = df_small.apply(lambda row: removeHints(row['colour'].lower(), 'wine', term=''), axis=1)

# remove grape names from description
for g in all_grapes:
    df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], g, term = 'wine'), axis=1)

# remove hints that exist in name column from description
df_small['description'] = df_small.apply(lambda row: remove_hints_from_description(row['description'].lower(), row['name'].lower()), axis=1)

df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], '100%', term = ''), axis=1)

df_small['description'] = df_small.apply(lambda row: removeHints(row['description'], row['colour'].lower()+' wine', term = 'wine'), axis=1)



#subset the 4 grape types we want to study and which are the most representative
#list_of_grapes = ['chardonnay', 'pinot noir', 'sauvignon blanc', 'syrah', 'sangiovese', 'cabernet sauvignon']
#df_small = df_sel.loc[df_sel['grape_variety'].isin(list_of_grapes)]


df_small.to_excel('cleaned_majestic_wine_list_with_body.xlsx')









