# WineSommelier
A project on a blind taster algorithm that intends to predict the grape variety based on a semi-professional description of wines.

## Organization of the repository
### Blind taster predictor
The predictor resides in the main folder in *sommelier.py*. The code comprises a pipeline that takes numerical and text based data and a classifier, trains the classifier on a train set and prints out the accuracy, confusion matrix and classification report for both train and test predictions. Furthermore, there is a hyperparameter optimization using GridSeach cross validation and at the end cross validation test results for 3 folds. 

### DataBase folder
The input data for the *sommelier.py* code resides in the *DataBase* folder in an excel file.

### Scraping folder
The input database was obtained by webscraping [Bibendum](https://www.bibendum-wine.co.uk/) and [Majestic Wine](https://www.majestic.co.uk/). These scripts result in a much larger database then the one showed in the *DataBase* folder. To make sure I am not exploiting the hard work of those two companies to put together their database I only share a small portion of their data in the *DataBase* folder.

### DataCleaning folder
The scripts in this folder take the raw data from the scraping codes and clean and filter them. Particularly, they perform lower case transformation, extracting grape variety from wine names, removing grape types from the description column etc. These codes make sure that the input data for the *sommelier.py* code is not introducing bias to the machine learning algorithm. The cleaned and truncated data is transferred into the *DataBase* folder.

### DataAnalysis and Publish folders
These two folders contain basically the same information (to some extent), but one was written in a text editor while the other in jupyter notebook. The files especially in the *Publish* folder are giving a detailed explanation of how can an algorithm make predictions on grape variety based on a description of a wine. They include visualization of the data and try to dig deep to understand the characteristic features of the grapes in the database. 


Zsolt Diveki
2018
[My GitHub Page](https://diveki.github.io)
[My GitHub](https://github.com/diveki)
