# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:15:01 2018
@author: agentimis1
"""
# This dataset have no catogorical vairables, and the algorithm doesn't use grid research and feature selection
# including feature importance plot
#%% Loading appropriate libraries ===============================================
import pandas as pd #Pandas 是一个强大的数据处理库，常用于加载、清洗和操作表格数据
import numpy as np  #NumPy 是一个用于数值计算的库，特别适用于数组和矩阵操作
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics, linear_model
# Unused packages 
#import tkinter as tk
#from tkinter import filedialog
#root = tk.Tk()
#root.withdraw()

#%% Loading data ===========================================================
#main_dir = filedialog.askdirectory(parent=root,initialdir="//",title='Pick a directory for the project')
main_dir=r'C:\Users\YaohuiLiu\Desktop\PrecisionAG\PrecisionHomework\Homework5'
#main_dir='G:/My Drive/Collaborations/01_Current/Digital_Ag/Digital_Ag_Class/01_General/Notes/04_MachineLearning/Random_Forest/Python'
Ex1 = pd.read_csv(main_dir+'/Data/Rice.csv')

#%%# 查看数据类型
print(Ex1.dtypes)

#%% Ex1.info() 还能提供非空值数量、内存占用等信息。
Ex1.info()

#%%
# 将数据类型转换为 DataFrame
dtypes_df = Ex1.dtypes.reset_index()
dtypes_df.columns = ['Column Name', 'Data Type']

# 保存为 CSV
dtypes_df.to_csv(main_dir + '/Results/Rice_dtypes.csv', index=False)

#%% Removing entries with missing values ==============================
#Ex1.dropna(subset=['YLD'],inplace=True)
Ex1=Ex1.dropna() #删除 DataFrame 中包含缺失值的行

#%% Description of the dataset and the columns, re-arranging so that Yield is first ========================
cols = list(Ex1) #Ex1 的列名（即 DataFrame 的所有列标签）转换为一个列表
cols.insert(0, cols.pop(cols.index('Yield')))
Ex1=Ex1.loc[:,cols]
df1=Ex1.describe()

#%% Converting character to numeric dummy variables ===============================================
Ex1feat = pd.get_dummies(Ex1)   #machine learning (random forest) can't run with catogrical variable
print('The shape of the Dataset with Dummy Variables is :',Ex1feat.shape)

#%% Training and test separations ===============================================
X=Ex1feat.iloc[:,1:len(Ex1feat.columns)].values #x needs to be matrix, .values：将 DataFrame 转换为 NumPy 数组，因为 RandomForestRegressor 需要 NumPy 矩阵作为输入。
y=Ex1feat.iloc[:,0].values.flatten()  #y needs to be array, .flatten()：确保 y 是 一维数组，而不是 (n,1) 形状的二维数组
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

#%% Normalization of the training and test =====================================
sc = StandardScaler()  #convert big number into small number between -1 to 1
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 

#%% Setting up the random forest and Fiting the dataset ===================
regressor = RandomForestRegressor(n_estimators=1000, random_state=0)  
regressor.fit(X_train, y_train)  

#%% Prediction and computation of metrics of errors =============================
y_pred_rf = regressor.predict(X_test)  
print('Mean Absolute Error for Random Forests:', metrics.mean_absolute_error(y_test, y_pred_rf))  #MAE（平均绝对误差）
print('Mean Squared Error for Random Forests:', metrics.mean_squared_error(y_test, y_pred_rf))  #MSE（均方误差）
print('Root Mean Squared Error for Random Forests:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))  #use the input(yield) mean to estimate the feeling of error

#%%#%% Show relative variable importance in RF (Sorted) =====================================
import matplotlib.pyplot as plt

# 获取特征重要性
importances = regressor.feature_importances_
feature_names = np.array(Ex1feat.columns[1:])  # 排除掉 'Yield'，只保留自变量的名称

# 对特征重要性进行排序
sorted_indices = np.argsort(importances)  # 按升序排序
sorted_importances = importances[sorted_indices]
sorted_features = feature_names[sorted_indices]

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.show()


#%% Create linear regression object =============================================
regr = linear_model.LinearRegression() #while use regression, strongly suggest using random forest to test 
regr.fit(X_train, y_train)

#%% Computing metrics for Linear regression ====================================
y_pred_lr=regr.predict(X_test)
print('Mean Absolute Error for Linear Regression:', metrics.mean_absolute_error(y_test, y_pred_lr))  
print('Mean Squared Error for Linear Regression:', metrics.mean_squared_error(y_test, y_pred_lr))  
print('Root Mean Squared Error for Linear Regression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))  

#didn't set random state in regression, results might change
#Python run random forest faster than R, nenural network runs well than random forest while bigger dataset