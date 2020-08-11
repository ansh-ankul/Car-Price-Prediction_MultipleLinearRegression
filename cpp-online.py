#------------------------------------------Data Preprocessing--------------------------------------------------
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing the dataset
dataset = pd.read_csv('datasets_175168_395113_CarPrice_Assignment.csv')

#---------Data Cleaning----------
#Splitting company name from CarName column
CompanyName = dataset['CarName'].apply(lambda x : x.split(' ')[0])
dataset.insert(3,"CompanyName",CompanyName)
dataset.drop(['CarName'],axis=1,inplace=True)
dataset.head()

#Correcting misspells
dataset.CompanyName = dataset.CompanyName.str.lower()

def replace_name(a,b):
    dataset.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

#Creating new features based on plotting on jupyter
dataset['fueleconomy'] = (0.60 * dataset['citympg']) + (0.40 * dataset['highwaympg'])
dataset = dataset.drop(["highwaympg"], axis = 1)
dataset = dataset.drop(["citympg"], axis = 1)
dataset = dataset.drop(["car_ID"], axis = 1)
dataset = dataset.drop(["doornumber"], axis = 1)
dataset = dataset.drop(["cylindernumber"], axis = 1)
dataset = dataset.drop(["enginelocation"], axis = 1)

dataset["brand_category"] = dataset['price'].apply(lambda x : "Budget" if x < 10000 
                                                     else ("Mid_Range" if 10000 <= x < 20000
                                                           else ("Luxury")))


attributes = dataset[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginetype'
       , 'enginesize',  'boreratio', 'horsepower', 'price', 'brand_category', 'fueleconomy']]


#Handling Categorical Data
# Defining the map function
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

dataset = dummies('CompanyName',dataset)
dataset = dummies('fueltype',dataset)
dataset = dummies('fuelsystem',dataset)
dataset = dummies('aspiration',dataset)
dataset = dummies('carbody',dataset)
dataset = dummies('drivewheel',dataset)
dataset = dummies('enginetype',dataset)
dataset = dummies('brand_category',dataset)

#Splitting into training and test set
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(dataset, train_size = 0.8, test_size = 0.2, random_state = 100)

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

#Correlation using heatmap
plt.figure(figsize = (30, 25))
sns.heatmap(df_train.corr(), annot = True, cmap="RYlGnBu")
plt.show()


#Dividing data into X and y variables
y_train = df_train.pop('price')
X_train = df_train

#----------------------------------------Model Building----------------------------------------------------------

#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

lm = LinearRegression()
lm.fit(X_train,y_train)
rfe = RFE(lm, 10)
rfe = rfe.fit(X_train, y_train)

X_train_rfe = X_train[X_train.columns[rfe.support_]]

def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)

X_train_new = build_model(X_train_rfe,y_train)
X_train_new = X_train_rfe.drop(["hardtop"], axis = 1)

X_train_new = build_model(X_train_new,y_train)



lm = sm.OLS(y_train,X_train_new).fit()
y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)   
#Calculating the Variance Inflation Factor
checkVIF(X_train_new)


#Scaling the test set
num_vars = ['symboling','wheelbase', 'curbweight', 'enginesize', 'boreratio','stroke','compressionratio','peakrpm', 'horsepower','fueleconomy','carlength','carwidth','carheight','price']
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

#Dividing into X and y
y_test = df_test.pop('price')
X_test = df_test

# Now let's use our model to make predictions.
X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)

# Making predictions
y_pred = lm.predict(X_test_new)

from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)

#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)


