import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')



train=pd.read_csv("loanpred/train_loan predict.csv",encoding="latin1")
test=pd.read_csv("loanpred/test_loan.csv",encoding="latin1")
print("train data")
print(train.head())
print("test data")
print(test.head())
train_orginal=train.copy()
test_orginal=test.copy()

print(" train_data column name: \n",train.columns)
print("test data column name:\n",test.columns)
print(train.dtypes)
print("train_data_shap: ",train.shape)
print("test_data_shap: ",test.shape)

print(train['Loan_Status'].value_counts())
print("Normalize the data")
print(train['Loan_Status'].value_counts(normalize=True))
print(train['Loan_Status'].value_counts().plot.bar())
plt.xlabel('Loan_Status')
plt.ylabel('count')
plt.title("loan statue plot")
plt.show()

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='self_employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='credit_history')
#plt.show()


plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='dependents')
plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Education')
plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='property_area')
#plt.show()'''

plt.figure(2)
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
#plt.show()

train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")
#plt.show()

plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
#plt.show()

plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(train['LoanAmount'])
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
#plt.show()

Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
#plt.show()

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
self_Emoloyed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
##plt.show()
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
#plt.show()
Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
#plt.show()
self_Emoloyed.div(self_Emoloyed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
#plt.show()


Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
#plt.show()
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
#plt.show()

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
#plt.show()

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High','Very high']
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('ApplicantIncome')
p=plt.ylabel('Percentage')
#plt.show()

bins=[0,1000,3000,42000]
group=['Low','Average',"High"]
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('CoapplicantIncome')
p=plt.ylabel('Percentage')
#plt.show()

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High','Very High']
train['Total_incom_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_income_bin=pd.crosstab(train['Total_incom_bin'],train['Loan_Status']) 

Total_income_bin.div(Total_income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
plt.xlabel('Total_income')
p=plt.ylabel('Percentage')
#plt.show()
Total_income_bin.div(Total_income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)
Total_income_bin.div(Total_income_bin.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

train=train.drop(['Income_bin','Coapplicant_Income_bin','LoanAmount_bin','Total_income_bin','Total_Income'],axis=1,errors="ignore")

train['Loan_Status'] = train['Loan_Status'].replace({'Y': 1, 'N': 0})

train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)

#test['Loan_Status'].replace('Y',1,inplace=True)

matrix = train.select_dtypes(include=[np.number]).corr()
f, ax=plt.subplots(figsize=(9,6))
sns.heatmap(matrix,vmax=.8,square=True,cmap="BuPu")
#plt.show()
#HANDILING OUTLIER AND MISSING VALUE in train data set
print(train.isnull().sum())
# HANDILING CATOGARICAL VALUE IN 'MODE' AND NUMERICAL VALUE IN USING MEAN AND MEDIAN
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train["Dependents"].fillna(train['Dependents'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)

train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)


train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)

print(train.isnull().sum())
#HANDILING OUTLIER AND MISSING VALUE in test data set
test['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
test['Married'].fillna(train['Married'].mode()[0],inplace=True)
test["Dependents"].fillna(train['Dependents'].mode()[0],inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)

test['Loan_Amount_Term'].value_counts()
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)


test['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)

#handiling   OUTLIER 

train['LoanAmount_log']=np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log']=np.log(test['LoanAmount'])

#MODEL BUILDING (LOGISTIC REGRESSION)

# Drop Loan_ID
train = train.drop('Loan_ID', axis=1)
test = test.drop('Loan_ID', axis=1)

# Separate features and target
X = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']

# Convert categorical to dummy variables
X = pd.get_dummies(X)
test = pd.get_dummies(test)

# Align test dataset columns with train dataset
test = test.reindex(columns=X.columns, fill_value=0)

# Train-Test Split (with stratify)
x_train, x_cv, y_train, y_cv = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Logistic Regression Model
model = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    random_state=42
)

model.fit(x_train, y_train)

# Cross Validation Prediction
pred_cv = model.predict(x_cv)

print("Prediction Accuracy:", accuracy_score(y_cv, pred_cv))

# Predict on Test Data
pred_test = model.predict(test)

# Create submission file
submission = pd.DataFrame({
    'Loan_ID': test_orginal['Loan_ID'],
    'Loan_Status': pred_test
})

# Convert 0/1 back to N/Y
submission['Loan_Status'] = submission['Loan_Status'].replace({0: 'N', 1: 'Y'})

submission.to_csv('logistic.csv', index=False)

print("Submission file created successfully!")


# Create submission DataFrame
submission = pd.DataFrame({
    'Loan_ID': test_orginal['Loan_ID'],
    'Loan_Status': pred_test
})

# Convert numeric predictions back to categorical labels
submission['Loan_Status'].replace({0: 'N', 1: 'Y'}, inplace=True)

# Save to CSV
submission.to_csv('logistics.csv', index=False)

from sklearn.metrics import classification_report
print(classification_report(y_cv, pred_cv))