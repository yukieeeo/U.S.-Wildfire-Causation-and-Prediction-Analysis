#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:31:30 2019

@author: YukiZ.
"""
# Import Packages

from bs4 import BeautifulSoup as soup
from urllib.request import urlopen as uReq  
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns
wildfire = pd.read_csv('Fires.csv', header = None, low_memory = False)

wildfire.columns = ['Fire', 'FOD_ID', 'FPA_ID', 'Source_system_type', 'source_system',
                    'nwcg_reporting_agency', 'nwcg_reporting_unit_id', 'nwcg_reporting_unit_name',
                    'source_reporting_unit', 'source_reporting_unit_name',
                    'local_fire_report_id', 'local_incident_id', 'fire_code', 'fire_name', 
                    'ics_209_incident_number', 'ics_209_name', 'mtbs_id', 'mtbs_fire_name', 'complex',
                    'fire_year', 'discovery_date', 'discovery_doy', 'discovery_time', 'stat_cause_code',
                    'stat_cause_descr', 'cont_date', 'cont_doy', 'cont_time', 'fire_size', 
                    'fire_size_class', 'lat', 'long', 'owner_code', 'owner_descr', 'state', 'county', 
                    'fips_code', 'fips_name', 'code']

wildfire_abridged = wildfire.loc[0:200, :]
wf = wildfire.loc[:]


###############################################################

################## Dropping unneeded columns ##################

###############################################################
wf = wildfire.drop('Fire', axis=1)
wf = wf.drop('FPA_ID', axis=1)
#We drop these columns because they don't provide relevant information for our project 

#wildfire = wildfire.drop('FOD_ID', axis=1)

wf = wf.drop('Source_system_type', axis=1)
wf = wf.drop('source_system', axis=1)
wf = wf.drop('nwcg_reporting_agency', axis=1)
#We drop these columns because we don't think these variables will have much 
#impact on the occurence of wildfires

wf = wf.drop('nwcg_reporting_unit_id', axis=1)
wf = wf.drop('source_reporting_unit', axis=1)
wf = wf.drop('source_reporting_unit_name', axis=1)
wf = wf.drop('local_fire_report_id', axis=1)
wf = wf.drop('local_incident_id', axis=1)
wf = wf.drop('fire_code', axis=1)
wf = wf.drop('ics_209_incident_number', axis=1)
#We drop these columns because they contain abstract elements the audience
#would have a difficult time deciphering


wf = wf.drop('fire_name', axis=1)
#We drop this column because the name assigned to the wildfire won't have any 
#effect on a future occurence of wildfires

wf = wf.drop('ics_209_name', axis=1)
wf = wf.drop('mtbs_id', axis=1)
wf = wf.drop('mtbs_fire_name', axis=1)
wf = wf.drop('complex', axis=1)
#We drop these columns because they contain numerous missing values that could
#skew the data tremendously

#wf = wf.drop('stat_cause_code', axis=1)

wf = wf.drop('owner_code', axis=1)
wf = wf.drop('county', axis=1)
wf = wf.drop('code', axis=1)
#We drop these columns because they don't provide relevant information for our project

###############################################################

################### Handling Missing Values ###################

###############################################################

nn = wf.notnull()
wf_abridged = wf.loc[0:len(wf), :]
length = len(wf_abridged) #To aid with later code, length of the dataframe is collected before dropping 
#the missing values are dropped.  
ctc = wf_abridged.state.isin(['CA', 'TX', 'CO'])
wf_abridged = wf_abridged[wf_abridged.state.isin(['CA', 'TX', 'CO'])]

#Rows to be used can be adjusted

wf_abridged = wf_abridged.dropna(how='any')

###############################################################

###################### Formatting issues ######################

###############################################################

#First we'll create a better representation of the dates
def conversion(n, year): #Function to convert the day (1-366) into a numbered month and day
    date_conversion = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365, 400]   
    month = 1
    day = 0
    placeholder = 0
    for i in date_conversion:
        if n <= i:
            day = n - placeholder
            break
        else:
            month = month + 1
        placeholder = i
  
    #Case of leap-year
    if month > 2 and year % 4 == 0:
        day = day - 1
        if day == 0:
            month = month - 1
            day = date_conversion[month - 1] - date_conversion[month - 2]
    
    return(month, day)
    
#Convert elements in 'discovery_doy' into a day and month
dmonth = []
dday = []
for i in wf_abridged.index: #'length' is defined above in "Handling Missing Values"
    #if all(nn.loc[i,:]): #'nn' is defined above in "Handling Missing Values"
     #   if ctc.loc[i]:
      
    dmonth.append(conversion(wf_abridged.loc[i,'discovery_doy'], wf_abridged.loc[i,'fire_year'])[0])
    dday.append(conversion(wf_abridged.loc[i,'discovery_doy'], wf_abridged.loc[i,'fire_year'])[1])

#Add these converted days and months to the dataframe
wf_abridged.loc[:,'discovery_month'] = dmonth
wf_abridged.loc[:,'discovery_day'] = dday

#Convert elements in 'cont_doy' into a day and month
cmonth = []
cday = []
for i in wf_abridged.index:  
    cmonth.append(conversion(wf_abridged.loc[i,'cont_doy'], wf_abridged.loc[i,'fire_year'])[0])
    cday.append(conversion(wf_abridged.loc[i,'cont_doy'], wf_abridged.loc[i,'fire_year'])[1])

#Add these converted days and months to the dataframe
wf_abridged.loc[:,'cont_month'] = cmonth
wf_abridged.loc[:,'cont_day'] = cday

#Now that we have better representations of the dates, we can drop a few columns
wf_abridged = wf_abridged.drop('discovery_date', axis=1)
#wf_abridged = wf_abridged.drop('discovery_doy', axis=1)
wf_abridged = wf_abridged.drop('cont_date', axis=1)
wf_abridged = wf_abridged.drop('cont_doy', axis=1)

wf_abridged.to_csv('abridged.csv')



##################################################################

###################### Getting Weather Data ######################

##################################################################

os.chdir("/Users/YukiZ./Desktop")
#import the table 
final = pd.read_csv("new1.csv")

#Drop the untitled variables
final = final.drop("Unnamed: 0", axis = 1)
final = final.drop("Unnamed: 0.1", axis = 1)

#Function that gets the webpage based on the zipcode, year, month, and day
def getUrl(zipcode, year, month, day):
    return("https://www.almanac.com/weather/history/zipcode/" + zipcode + "/" + year + "-" + month + "-" + day)

#Function web scrapes temperature and wind
def getWeather(someString):
    url = someString
    uClient = uReq(url)    
    page_html = uClient.read() 
    uClient.close()
    page_soup = soup(page_html, "lxml")
    tabular = page_soup.findAll("table")
    
    #Locates the table which the data is stored
    split = tabular[1].findAll("tr")
    
    #Web scrapes the temperature and wind speed
    temp_text = split[2].text
    temperature = temp_text[16:len(temp_text) - 3]
    wind_text = split[len(split) - 3].text
    wind_speed = wind_text[15:len(wind_text) - 4]
    
    #Changes text to ascii
    temperature = temperature.encode("ascii")
    wind_speed = wind_speed.encode("ascii")
    
    #Returns values as tuple
    return(temperature, wind_speed)
        
  

    
def main():
    #print(getWeather("81327", "2005", "09", "29"))
    data = {'index' : "",'temperature': "", 'wind_speed': ""}
    exc = pd.DataFrame([data])
    exc.to_csv('exc.csv', mode = 'a+', header = False)
    
    for i in range(0, 8000):
        temp_info = []
        wind_info = []
        print(i)
        #j = i 
        #Stores information from row i in table into variables
        city = str(final.loc[i, "cities"])
        zipcode = city[len(city) - 5:len(city)]
        year = str(final.loc[i, "fire_year"])
        month = str(final.loc[i, "discovery_month"])
        
        #Formats month and day
        if (len(month) == 1):
            month = "0" + month
        day = str(final.loc[i, "discovery_day"])
        if (len(day) == 1):
            day = "0" + day
            
        #Web scrapes specific information
        inf = getUrl(zipcode, year, month, day)
        information = getWeather(inf)
        
        #Gets information from tuple
        temp_info.append(information[0])
        wind_info.append(information[1])
        
        #Stores information in excel
        data = {'index' : i,'temperature': temp_info, 'wind_speed': wind_info}
        exc = pd.DataFrame(data)
        exc.to_csv('fab.csv', mode = 'a+', header = False)
    #print(temp_info)
    #print(wind_info)
    
main()  


fire = pd.read_csv('fab.csv')
fire.columns
# Drop redundant columns and rename columns
fire = fire.drop(['Unnamed: 0','FOD_ID','stat_cause_code','discovery_doy'], axis=1)
fire = fire.rename(columns = {'nwcg_reporting_unit_name': 'reporting_unit_name'})
# convert reporting_unit_name to numerical 
le = preprocessing.LabelEncoder()
le.fit(fire.reporting_unit_name)
le.classes_
reporting_unit_name = le.transform(fire.reporting_unit_name)

# convert owner_descr to numerical 
le.fit(fire.owner_descr)
le.classes_
owner_descr = le.transform(fire.owner_descr)

# convert state to numerical 
le.fit(fire.state)
le.classes_
state = le.transform(fire.state)

# convert fips_name to numerical 
le.fit(fire.fips_name)
le.classes_
fips_name  = le.transform(fire.fips_name)

# convert cities to numerical 
le.fit(fire.cities )
le.classes_
cities  = le.transform(fire.cities)

# convert fire_size_class to numerical 
le.fit(fire.fire_size_class)
le.classes_
fire_size_class  = le.transform(fire.fire_size_class)

# replace columns with numerical values
fire['reporting_unit_name']=reporting_unit_name
fire['owner_descr']=owner_descr
fire['state']=state
fire['fips_name']=fips_name
fire['cities']=cities
fire['fire_size_class']=fire_size_class
fire.head(10)

# Split into train and test
Y = fire['stat_cause_descr']
X = fire.drop(['stat_cause_descr'],axis = 1)

# training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # 20% test data
X_train.head(10)



########## Create Decision Tree Classifier #########

def tree_accuracy_max_depth(n):
    clf = DecisionTreeClassifier(criterion="gini", max_depth = n)
    # Trian Decision Tree Classifier
    clf = clf.fit(X_train,Y_train)
    # predict y_test
    Y_pred = clf.predict(X_test)
    # Model Accuracy
    print("Accuracy when max_depth = ", n,":",accuracy_score(Y_test,Y_pred))
for i in np.arange(1,15):
    tree_accuracy_max_depth(i)



# Therefore we choose max_depth = 11, we will evaluate overfitting later using cross validation
clf = DecisionTreeClassifier(criterion="gini", max_depth = 11)
# Trian Decision Tree Classifier
clf = clf.fit(X_train,Y_train)
# predict y_test
Y_pred = clf.predict(X_test)
# Model Accuracy
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))


# Cross Validation- avergae score

scores = []
for k in range(5):
    # We use 'list' to copy, in order to 'pop' later on
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    scores.append(clf.fit(X_train, Y_train).score(X_test, Y_test))
np.mean(scores)



# Confusion Matrix
y_true = Y_test
y_pred = Y_pred
y_actu = pd.Series(Y_test, name='Actual')
y_pred = pd.Series(Y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion

# Confusion Matrix - vis
y_true = Y_test
y_pred = Y_pred
y_actu = pd.Series(Y_test, name='Actual')
y_pred = pd.Series(Y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) ################
df_confusion

plt.figure(figsize = (10,7))
sns.heatmap(df_confusion, annot=True,fmt="d",cmap="Greys")
plt.title('Decision Tree Classifier\n Accuracy:0.533')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



# Graph - tree example with max_depth = 2
clf_2 = DecisionTreeClassifier(criterion="gini", max_depth = 2)
# Trian Decision Tree Classifier
clf_2 = clf_2.fit(X_train,Y_train)
features = X.columns
dot_data = StringIO()
export_graphviz(clf_2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

###################################
############ ROC ##################
###################################


def matrix_elements(array, n):
    tp = array[n,n]
    fn = sum(array[n,:]) - array[n,n]
    fp = sum(array[:,n]) - array[n,n]
    tn = sum(sum(array[:,:])) - tp - fn - fp
    return(tp, tn, fp, fn)
    

def tpr(array, n):
    ret = matrix_elements(array, n)[0] /(matrix_elements(array, n)[0] + matrix_elements(array, n)[3])
    return(ret)
def fpr(array, n):
    ret = matrix_elements(array, n)[2] /(matrix_elements(array, n)[2] + matrix_elements(array, n)[1])
    return(ret)
    
#We'll go for 100 data points
elements = [(0,0),(1,1)]

for i in range(0,100):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    #print(Y_test.head())
    clf = DecisionTreeClassifier(criterion="gini", max_depth = 1, random_state = i) #random_state can be any non-constant number
    clf = clf.fit(X_train,Y_train) 
    Y_pred = clf.predict(X_test)
    mat = confusion_matrix(Y_test, Y_pred)
    elements.append((tpr(mat, 6), fpr(mat,6)))
size = lambda elements: elements[1]   
elements.sort(key=size)
tprArray = []
fprArray = []
for i in range(0, len(elements)):
    tprArray.append(elements[i][0])
    fprArray.append(elements[i][1])

plt.xlabel("FPR (1 - specificity)")
plt.ylabel("TPR (sensitivity)")
plt.title("ROC curve")
plt.plot(fprArray, tprArray)
plt.plot([0,1],[0,1], 'r--')
#plt.xticks(np.arange(len(tprArray)), tprArray)
plt.show()


####### XGB_TREE: fit model- training data #######
xgb_tree = xgb.XGBClassifier()
xbg_tree = xgb_tree.fit(X_train, Y_train)
# predict y_test
Y_pred = xgb_tree.predict(X_test)
Y_pred
# Model Accuracy
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))



# Confusion Matrix
y_true = Y_test
y_pred = Y_pred
y_actu = pd.Series(Y_test, name='Actual')
y_pred = pd.Series(Y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion

# Confusion Matrix
y_true = Y_test
y_pred = Y_pred
y_actu = pd.Series(Y_test, name='Actual')
y_pred = pd.Series(Y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion
plt.figure(figsize = (10,7))
sns.heatmap(df_confusion, annot=True,fmt="d",cmap="Greens")
plt.title('XGB_TREE\nAccuracy:0.539')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

xgb.plot_tree(xbg_tree,num_trees=0)
plt.show()



###### Random Forest ######
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf = clf_rf.fit(X_train, Y_train)
# predict y_test
Y_pred = clf_rf.predict(X_test)



# Model Accuracy
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))



# Cross Validation- avergae score

scores = []
for k in range(5):
    # We use 'list' to copy, in order to 'pop' later on
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    scores.append(clf_rf.fit(X_train, Y_train).score(X_test, Y_test))
np.mean(scores)



# Confusion Matrix
y_true = Y_test
y_pred = Y_pred
y_actu = pd.Series(Y_test, name='Actual')
y_pred = pd.Series(Y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion

# Confusion Matrix - vis
plt.figure(figsize = (10,7))
sns.heatmap(df_confusion, annot=True,fmt="d",cmap="Oranges")
plt.title('Random Forest\nAccuracy:0.584')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Get numerical feature importances and Graph
features = X.columns
importances = list(clf_rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]



# Fire Fighter: delete cont_month, cont_time, cont_day, state, fire_size_class  
# U.S. Forest Services and Insurers: delete state, fire_size_class to avoid over-fitting

# U.S. Forest Services and Insurers example
X = X.drop(['state','fire_size_class'],axis = 1)
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # 20% test data
# Random Forest
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf = clf_rf.fit(X_train, Y_train)
# predict y_test
Y_pred = clf_rf.predict(X_test)
# Model Accuracy
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
# Confusion Matrix
y_true = Y_test
y_pred = Y_pred
y_actu = pd.Series(Y_test, name='Actual')
y_pred = pd.Series(Y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion
# Get numerical feature importances and Graph
features = X.columns
importances = list(clf_rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# Fire Fighters example:
X = X.drop(['cont_time','cont_day','cont_month'],axis = 1)
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # 20% test data
# Random Forest
clf_rf = RandomForestClassifier(n_estimators=100)
clf_rf = clf_rf.fit(X_train, Y_train)
# predict y_test
Y_pred = clf_rf.predict(X_test)
# Model Accuracy
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
# Confusion Matrix
y_true = Y_test
y_pred = Y_pred
y_actu = pd.Series(Y_test, name='Actual')
y_pred = pd.Series(Y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion
# Get numerical feature importances and Graph
features = X.columns
importances = list(clf_rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]



height = importances
bars = features
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height, color='orange')
 
# Create names on the x-axis
plt.xticks(y_pos, bars, rotation='vertical')
 
# Show graphic
plt.show()
X_test




###### Conclusion : We will use random forest method to split trees. It has a accuracy of 0.58 on average.

