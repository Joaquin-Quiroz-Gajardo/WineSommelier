# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:21:38 2018

@author: zdiveki
"""


import pandas as pd
import nltk


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import cross_val_score
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
import re
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
from string import punctuation
from functools import reduce
import pdb
import seaborn as sns
import matplotlib.pyplot as plt

filename = '../cleaned_wine_list_with_body.xlsx'
file1 = '../cleaned_majestic_wine_list_with_body.xlsx'

a0=pd.read_excel(filename)
a1 = pd.read_excel(file1)

columns_sel = ['abv', 'colour', 'country', 'description', 'grape_variety', 'name', 'Body']

a_concat = pd.concat([a0[columns_sel], a1[columns_sel]]).reset_index()
grapes = a_concat['grape_variety'].tolist()

# in case we do not want to blend grapes
grape = [grape for x in grapes if len(x.split(','))==1 for grape in x.split(', ')]
fdist=nltk.FreqDist(grape)


result = a_concat['grape_variety']

## removing varieties that have only one member in the database
counts = nltk.Counter(result)
varieties = [key for key in counts if counts[key] > 40]
data_input = a_concat[a_concat['grape_variety'].isin(varieties)].reset_index()


# split the data into train and test
combined_features = ['Body', 'description', 'grape_variety']
target = 'grape_variety'

X_train, X_test, y_train, y_test = train_test_split(data_input[combined_features], data_input[target], test_size=0.33, random_state=42)


# aggregate description by grape type
grouped = X_train[['grape_variety', 'description']].groupby(['grape_variety']).agg({'description': lambda z: reduce(lambda x,y: ''.join(x+y), z)})

# tokenize the combined text
grouped['tokens'] = grouped.loc[:, 'description'].apply(word_tokenize)

def word_count_df(df, src_col, out_col=('grape', 'token', 'count')):
    dfp = pd.DataFrame()
    for ii in df.index:
        dist = nltk.FreqDist(df.loc[ii, src_col])
        dftmp = pd.DataFrame(dict(zip(out_col, ([ii]*len(dist.keys()), list(dist.keys()), list(dist.values())))))
        dfp=pd.concat([dfp, dftmp], ignore_index=True)
    return(dfp)

tkn_count = word_count_df(grouped, 'tokens', out_col=('grape', 'token', 'count'))

# plot the word count of each grape type
fig, axarr = plt.subplots(2, 2, figsize=(10, 6))
plt.subplots_adjust(top=0.99, bottom=0.15, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
ii = 0
for row in axarr:
    for col in row:
        col.set_title(grouped.index[ii])
        sns.barplot(x='token', y='count', 
            data=tkn_count[(tkn_count['grape']==grouped.index[ii]) & (tkn_count['count']>20)].sort_values('count', ascending=False), 
            ax = col)
        ii += 1
        col.set_xticklabels(col.get_xticklabels(), rotation=50)



pl = sns.catplot(x="token", y="count", col="grape", data=tkn_count, kind="bar", height=4, aspect=.7, sharex=False, sharey=False, col_wrap=2)
pl.set_xticklabels(rotation=50)


### Cleaning the whole description section
pattern = re.compile(r'[0-9]+|\b[\w]{2,2}\b|[%.,_`!"?\')({~@;:#}+-]+|\b[\w]{1,1}\b')

X_train['description_cleaned'] = X_train['description'].apply(lambda t: pattern.sub(' ', t))


# get position tags 
def filter_tokens(tok):
    tmp = [x for x,y in tok if y not in ['DT', 'CC', 'IN', 'TO', 'CD', '.', ':']]
    return(tmp)
    
grouped['postag'] = grouped.loc[:, 'tokens'].apply(nltk.pos_tag)
grouped['postag'] = grouped.loc[:, 'postag'].apply(filter_tokens)


col = 'postag'
dfp = pd.DataFrame()
for ii in grouped.index:
    dist = nltk.FreqDist(grouped.loc[ii, col])
    dftmp = pd.DataFrame(dict(zip(('grape', 'token', 'count'), ([ii]*len(dist.keys()), list(dist.keys()), list(dist.values())))))
    dfp=pd.concat([dfp, dftmp], ignore_index=True)

# plot the findings
con1 = dfp['count']>10
con2 = dfp['count']<100
con3 = dfp['grape'] == 'pinot noir'
sns.barplot(x='token', y='count', data=dfp[con1 & con2 & con3])
plt.xticks(rotation=50)

con1 = dfp['count']>25
con2 = dfp['count']<30
pl = sns.catplot(x="token", y="count", col="grape", data=dfp[con1 & con2], kind="bar", height=4, aspect=.7, sharex=False, sharey=False, col_wrap=2)
pl.set_xticklabels(rotation=50)

txt_tok = word_tokenize(txt_sub)
lem = LemmaTokenizer()
txt_filt = lem(txt_sub)
txt_filt = [x for x in txt_filt if len(x) > 2]
tdist = nltk.FreqDist(txt_filt)
