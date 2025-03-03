# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:49:21 2025

@author: YaohuiLiu
"""
#%% 代码调整点 （用feature selection and grid search来优化代码的超参数）
# 1. 避免重复训练模型: 以前 RandomForestRegressor 直接训练数据，现在先训练一个 基础模型 (rf_base) 用于特征选择，再用 优化后的模型 (best_rf) 进行最终预测。
# 2. 特征选择后训练最优模型:SelectFromModel() 自动挑选最相关的特征，避免不必要的变量降低模型性能。
# 3. 超参数优化,用 GridSearchCV() 选择最优参数，提高模型准确度。
# 4. 特征显著性作图

#%% import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel

#%% 读取数据
main_dir = r'C:\Users\YaohuiLiu\Desktop\PrecisionAG\PrecisionHomework\Homework5'
Ex1 = pd.read_csv(main_dir + '/Data/Rice.csv')

#%% 处理缺失值
Ex1 = Ex1.dropna()

#%% 调整列顺序，使 'Yield' 变为第一列
cols = list(Ex1)
cols.insert(0, cols.pop(cols.index('Yield')))
Ex1 = Ex1.loc[:, cols]

#%% 训练集 & 测试集划分
X = Ex1.iloc[:, 1:].values  # 自变量 (independent variables)
y = Ex1.iloc[:, 0].values.flatten()  # 因变量 (Yield)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% 归一化 (Standardization)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% 进行特征选择 (Feature Selection) - 可以减少变量，提高训练效率。
rf_base = RandomForestRegressor(n_estimators=100, random_state=0)
rf_base.fit(X_train, y_train)
selector = SelectFromModel(rf_base, threshold="median", prefit=True)  # 选择比中位数重要的变量
X_train_selected = selector.transform(X_train) #transform() 生成新的特征集
X_test_selected = selector.transform(X_test)

#%%
print(f"Selected features shape: {X_train_selected.shape}")  # 查看筛选后特征数量

#%% 网格搜索进行超参数优化 (Grid Search) -找到最优的超参数，提高模型精度
# 定义超参数搜索范围
param_grid = {
    'n_estimators': [100, 500, 1000],   # 决策树数量
    'max_depth': [None, 10, 20],        # 树的最大深度
    'min_samples_split': [2, 5, 10],    # 节点最小分裂样本数
    'min_samples_leaf': [1, 2, 4]       # 叶子节点最小样本数
}

#%% 使用 GridSearchCV 进行超参数优化
grid_search = GridSearchCV(RandomForestRegressor(random_state=0),
                           param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

#%%获取最佳模型
best_rf = grid_search.best_estimator_  # 获取最优模型
print("Best Parameters:", grid_search.best_params_)

#%% 预测 & 计算误差
y_pred_rf = best_rf.predict(X_test_selected) # 用最佳模型预测
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_rf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_rf))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))

#%% 查看被选中的变量
selected_features = Ex1.columns[1:][selector.get_support()] #selector.get_support() 返回一个布尔数组，True 表示该特征被选中，结合 Ex1.columns 获取具体的列名。
print("Selected Features:", list(selected_features))

#%% 查看变量的重要性 (所有变量)
feature_importances = rf_base.feature_importances_
feature_names = Ex1.columns[1:]  # Exclude 'Yield'
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)  # 按重要性排序
print(importance_df)

#%%或者只显示被选中的变量的重要性。
selected_importance = importance_df[importance_df['Feature'].isin(selected_features)]
print(selected_importance)

#%%画图可视化特征重要性
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.gca().invert_yaxis()  # 让最重要的特征排在最上方
plt.show()
