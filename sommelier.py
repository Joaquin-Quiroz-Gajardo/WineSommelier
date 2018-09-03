# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
import pdb

filename = 'winelist.xlsx'
filename = 'cleaned_wine_list.xlsx'
filename = 'cleaned_wine_list_with_body.xlsx'
file1 = 'cleaned_majestic_wine_list_with_body.xlsx'

a0=pd.read_excel(filename)
a1 = pd.read_excel(file1)

columns_sel = ['abv', 'colour', 'country', 'description', 'grape_variety', 'name', 'Body']

a_concat = pd.concat([a0[columns_sel], a1[columns_sel]]).reset_index()
grapes = a_concat['grape_variety'].tolist()

# in case we do not want to blend grapes
grape = [grape for x in grapes if len(x.split(','))==1 for grape in x.split(', ')]
fdist=nltk.FreqDist(grape)

list_of_grapes = ['chardonnay', 'pinot noir', 'sauvignon blanc', 'syrah', 'sangiovese', 'cabernet sauvignon']
list_of_grapes = [key for key, value in fdist.items() if value > 20]
a = a_concat.loc[a_concat['grape_variety'].isin(list_of_grapes)]
a.loc[:, 'colour'] = a.loc[:, 'colour'].str.lower()
#data=a['Description']
#data = a['Alcohol']

#result = a['Variety']

#data=a['description']
#data = a['abv']

result = a['grape_variety']


##### dealing with variety
def shiraz_filter(ss):
    if ss == 'shiraz':
        return 'syrah'
    else:
        return ss


variety = {}
for ii in range(len(result)):
    tmp = result.iloc[ii].lower()
    tmp = tmp.split(',')
    tmp = [re.sub(r'^ ', '', x) for x in tmp]
    tmp = [re.sub(r' $', '', x) for x in tmp]
    tmp = [shiraz_filter(samp) for samp in tmp]
    tmp = str(set(tmp)).replace("'", '').replace('{', '').replace('}','')
    variety[ii] = tmp
    


result = pd.Series(variety)
#a['Variety']=result

## removing varieties that have only one member in the database
counts = nltk.Counter(result)
varieties = [key for key in counts if counts[key] > 30]

#data_input = a[a['Variety'].isin(varieties)]
data_input = a[a['grape_variety'].isin(varieties)].reset_index()
############################################





#defTags = ['CC', '.', ',', 'IN', ';', 'PRP', 'DT', 'MD', 'PDT', 'POS', 'TO', 'WDT', 'WP', 'WRB', 'NNP', 'RP']
#defTags = ['CC', '.', ',', 'IN', ';', 'DT', 'TO', 'CD']
defTags = ['NNS', 'NN', 'JJ', 'JJS', 'JJR']#, 'RB', 'RBS', 'RBR']#, 'VBD', 'VBZ']
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+'

def clean_function(ll):
    list_to_remove = ["'", 's', '-']
    list_to_split = ["'", "-", ".", '/', ',']
    tmp = [t for t in ll if t not in list_to_remove]
    for sremove in list_to_split:
        tmp = [" ".join(x.split(sremove)) for x in tmp]
        tmp = " ".join(tmp)
        tmp = tmp.split(" ")
    return tmp

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stemmer = nltk.stem.PorterStemmer()
    def __call__(self, doc):
        pattern = re.compile(r'[0-9]+|\b[\w]{2,2}\b|[%.,&_`!"?\')({~@;:#}+-]+|\b[\w]{1,1}\b')
        doc = pattern.sub('', doc)
        doc_tagged = nltk.pos_tag(word_tokenize(doc))
        doc = [t for t in doc_tagged if t[1] in defTags]
        doc = [(t[0], penn_to_wn(t[1])) for t in doc]
        doc = [self.wnl.lemmatize(t[0], t[1]) for t in doc]
        doc = clean_function(doc)
        doc = [x for x in doc if x != '']
        #doc = [self.stemmer.stem(x) for x in doc]
        return doc

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return nltk.stem.wordnet.wordnet.ADJ
    elif is_noun(tag):
        return nltk.stem.wordnet.wordnet.NOUN
    elif is_adverb(tag):
        return nltk.stem.wordnet.wordnet.ADV
    elif is_verb(tag):
        return nltk.stem.wordnet.wordnet.VERB
    return nltk.stem.wordnet.wordnet.NOUN
    

###### feature selector ##############
from sklearn.base import BaseEstimator, TransformerMixin


class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None, *parg, **kwarg):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    
    def transform(self, x, y=0):
        return self.encoder.transform(x)

##### train on alcohol level  ########
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer

clf = KNeighborsClassifier()
clf = DecisionTreeClassifier()
clf = RandomForestClassifier()



### testing selectors
body_dummies = pd.get_dummies(data_input['Body'])
colour_dummies = pd.get_dummies(data_input['colour'])
data_input = data_input.merge(body_dummies, left_index=True, right_index=True)
data_input = data_input.merge(colour_dummies, left_index=True, right_index=True)


#features= ['Body','Description']
#numeric_features= ['Alcohol', 'Year']
#combined_features = ['Alcohol', 'Full', 'Medium', 'Light', 'Description']
#target = 'Variety'
#combined_features = ['Body', 'description', 'full', 'light', 'medium', 'dry']
combined_features = ['Body', 'description', 'full', 'light', 'medium', 'dry', 'red', 'rose', 'white']
target = 'grape_variety'

## treating the Body feature

X_train, X_test, y_train, y_test = train_test_split(data_input[combined_features], data_input[target], test_size=0.33, random_state=42)

full = Pipeline([
                ('selector', NumberSelector(key='full')),
                ])
medium = Pipeline([
                ('selector', NumberSelector(key='medium')),
                ])
light = Pipeline([
                ('selector', NumberSelector(key='light')),
                ])
dry = Pipeline([
                ('selector', NumberSelector(key='dry')),
                ])
red = Pipeline([
                ('selector', NumberSelector(key='red')),
                ])
rose = Pipeline([
                ('selector', NumberSelector(key='rose')),
                ])
white = Pipeline([
                ('selector', NumberSelector(key='white')),
                ])
## description related training
stop_words = stopwords.words('english')
STOPwords = ['ha', 'wine', 'eg', "", '“', '”', '‘', '’', '``', '``', 'º f', 'º', '–', '+', 'wa', 'km', '°c', 'cth', 'tim', 'yet']#, 'fruit']
stop_words = stop_words + STOPwords + list(punctuation)

#vectorizer = CountVectorizer(ngram_range=(1,3), stop_words=STOPwords, analyzer='word', tokenizer=LemmaTokenizer(), token_pattern = TOKENS_ALPHANUMERIC)
 
#TfidfVectorizer
text = Pipeline([
                ('selector', TextSelector(key='description')),
                ('vectorizer', TfidfVectorizer(ngram_range=(1,1), stop_words=stop_words, analyzer='word', norm='l2', token_pattern = TOKENS_ALPHANUMERIC, tokenizer=LemmaTokenizer()))
                ])
    

## alcohol feature

alc = Pipeline([
                ('selector', NumberSelector(key='abv')),
                ('standard', StandardScaler())
                ])

## Feature union
feats = FeatureUnion([('full', full),
                      ('medium', medium),
                      ('light', light),
                      ('dry', dry),
                      ('description', text)
                      ])
feats = FeatureUnion([#('alcohol', alc),
                      ('description', text),
                      ('red', red),
                      ('rose', rose),
                      ('white', white)
                      ])
feats = FeatureUnion([('full', full),
                      ('medium', medium),
                      ('light', light),
                      ('dry', dry),
                      ('description', text),
                      ('red', red),
                      ('rose', rose),
                      ('white', white)
                      ])

    
pipe = Pipeline([('feats', feats),
                 ('clf',RandomForestClassifier(random_state=42))
                 ])
    
pipe.fit(X_train, y_train)

#train stats
preds = pipe.predict(X_train)
print(metrics.accuracy_score(y_train, preds))
print(metrics.classification_report(y_train, preds))
print(metrics.confusion_matrix(y_train, preds))
print(nltk.Counter(y_train))

# test stats
preds = pipe.predict(X_test)
print(metrics.accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds))
print(metrics.confusion_matrix(y_test, preds))
print(nltk.Counter(y_test))


from sklearn.model_selection import GridSearchCV

hyperparameters = { 'feats__description__vectorizer__ngram_range': [(1,1), (1,2), (1,3)],
                    'feats__description__vectorizer__strip_accents': [None, 'ascii', 'unicode'],
                    'feats__description__vectorizer__min_df': [0,0.5,1],
                    'feats__description__vectorizer__sublinear_tf': [False,True]#,
            #       'clf__max_depth': [50, 70],
            #        'clf__min_samples_leaf': [1,2],
            #        'clf__criterion': ['gini', 'entropy']
                  }

clf = GridSearchCV(pipe, hyperparameters, cv=5)
clf.fit(X_train, y_train)
clf.refit
preds = clf.predict(X_test)
#probs = clf.predict_proba(X_test)

np.mean(preds == y_test)
print(metrics.accuracy_score(y_test, preds))
print(metrics.classification_report(y_test, preds))
print(metrics.confusion_matrix(y_test, preds))
print(nltk.Counter(y_test))


### stratified training
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
sc_mean=[]
for train, test in skf.split(data_input[combined_features], data_input[target]):
    pipe.fit(data_input.loc[train,combined_features], data_input.loc[train, target])
    preds = pipe.predict(data_input.loc[test,combined_features])
    sc_mean.append(metrics.accuracy_score(data_input.loc[test, target], preds))
    
    print(metrics.accuracy_score(data_input.loc[test, target], preds))
    print(metrics.classification_report(data_input.loc[test, target], preds))
    print(metrics.confusion_matrix(data_input.loc[test, target], preds))
    print(nltk.Counter(data_input.loc[test, target]))
print('Mean: %s' % str(sum(sc_mean)/len(sc_mean)))
    




