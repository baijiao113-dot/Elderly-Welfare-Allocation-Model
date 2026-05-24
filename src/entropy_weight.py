# -*- coding: utf-8 -*-
"""
熵权
"""
import pandas as pd
import numpy as np

#  1. 读取数据 
file_path = "D:\math model\数学建模1.xlsx"   # 修改为你的实际路径

# 读取人员数据
df = pd.read_excel(file_path, sheet_name="数学建模")

# 读取指标对应关系（可选，用于参考）
ahp = pd.read_excel(file_path, sheet_name="AHP指标构建")
print("指标对应关系：")
print(ahp)

#  2. 选取你需要的指标
# 根据图片和AHP表，你选择的二级指标对应列名如下：
indicators = {
    '年龄': '年龄',
    '身体状况': '身体状况',
    '每月养老金（元）': '每月养老金（元）',
    '房产数目': '房产数目',
    '是否为低保户': '是否为低保户',
    '子女个数': '子女个数',
    '人才层级':'是否为国家、省、市级人才',
    '配偶情况': '当前是否是配偶配偶是否健在'
}

# 提取数据矩阵
data = df[list(indicators.values())].copy()

# 重命名列为中文方便阅读
data.columns = list(indicators.keys())
print("\n数据预览：")
print(data.head())

# 3. 数据预处理 
# 处理缺失值（如果有，用0或均值填充，根据实际意义决定）
data = data.fillna(0)


#  4. 熵权法核心函数 
def entropy_weight(df):
    # 转为numpy数组
    X = df.values.astype(float)
    m, n = X.shape  # m=样本数, n=指标数
    
    # 1. 数据标准化（正向化）
    # 对于所有指标，我们统一做“越大越好”的标准化（极差法）
    X_std = np.zeros_like(X)
    for j in range(n):
        col = X[:, j]
        min_val = col.min()
        max_val = col.max()
        if max_val == min_val:  # 防止全相同
            X_std[:, j] = 1
        else:
            # 正向指标：越大越好
            X_std[:, j] = (col - min_val) / (max_val - min_val)
    
    # 2. 计算比例 p_ij
    X_std = X_std + 1e-10  # 避免log(0)
    p = X_std / X_std.sum(axis=0)
    
    # 3. 计算熵值 e_j
    k = 1 / np.log(m)
    e = -k * np.sum(p * np.log(p), axis=0)
    
    # 4. 计算差异系数 d_j
    d = 1 - e
    
    # 5. 计算权重 w_j
    w = d / d.sum()
    
    # 结果
    result = pd.DataFrame({
        '指标': df.columns,
        '熵值': e,
        '差异系数': d,
        '熵权': w
    })
    return result.round(4)

#  5. 计算并输出
weights = entropy_weight(data)
print("\n=== 熵权计算结果 ===")
print(weights)

# 保存结果
weights.to_excel("熵权结果.xlsx", index=False)
print("\n结果已保存为：熵权结果.xlsx")

