# -*- coding: utf-8 -*-
"""
数学建模 - 养老金补贴分配系统
融合 MPI分类 + AHP + 熵权 + 敏感性分析 + 最大余额法整数规划
"""

import os
import pandas as pd
import numpy as np

# 1. 参数设置 
TOTAL_BUDGET = 200000      # 总预算 20万元
HIGH_DEPRIVED_AMOUNT = 1500  # 高剥夺群体固定补贴标准
LAMBDAS = [0.5, 0.6, 0.7]  # 敏感性分析权重调节参数

# 2. 读取数据
# 使用原始字符串 r'' 防止 Windows 路径中的斜杠被误判为转义字符
data_path = r'D:\math model\数学建模1.xlsx'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"未找到指定的Excel文件，请检查路径是否正确: {data_path}")

df = pd.read_excel(data_path, sheet_name='数学建模')

# 列名简化与规范化
df = df.rename(columns={
    '每月养老金（元）': '养老金',
    '当前是否是配偶配偶是否健在': '配偶情况',
    '是否为低保户': '低保户',
    '是否为国家、省、市级人才': '人才层级'
})

print(f"数据读取成功！总样本人数: {len(df)} 人")

# 3. MPI 剥夺分数计算
def compute_deprivation_score(row):
    """
    计算MPI剥夺分数（多维贫困指数思路）
    """
    score = 0
    if row['养老金'] <= 1000:                    # 经济困难
        score += 1
    if row['子女个数'] == 0:                      # 无子女
        score += 1
    if row['房产数目'] == 0:                      # 无房产
        score += 1
    if row['低保户'] == 1:                        # 低保户
        score += 1
    if row['身体状况'] != 1:                      # 非健康
        score += 1
    if row['配偶情况'] == 0:                      # 配偶不在
        score += 1
    if row['年龄'] >= 80:                         # 高龄
        score += 1
    return score

df['MPI_剥夺分数'] = df.apply(compute_deprivation_score, axis=1)

# 划分高剥夺群体（MPI > 4）与普通群体
high_deprived = df[df['MPI_剥夺分数'] > 4].copy()
remaining = df[df['MPI_剥夺分数'] <= 4].copy()

# 高剥夺群体固定发放金额
high_deprived['补贴'] = HIGH_DEPRIVED_AMOUNT
high_cost = len(high_deprived) * HIGH_DEPRIVED_AMOUNT
remain_budget = TOTAL_BUDGET - high_cost

print("初始筛查完成：")
print(f"   - 特别困难高剥夺老人 (MPI > 4): {len(high_deprived)} 人，共支出 {high_cost} 元")
print(f"   - 纳入精细化分配的普通老人: {len(remaining)} 人")
print(f"   - 普通老人可用剩余预算: {remain_budget} 元")

if remain_budget < 0:
    print("警告：高剥夺群体的固定补贴总额已超过20万总预算，请重新调整预算或补贴标准！")

# 4. 指标与权重设定
indicators = ['年龄', '身体状况', '养老金', '房产数目', '低保户', '人才层级', '子女个数', '配偶情况']
w_ahp = [0.1349, 0.4047, 0.0557, 0.0186, 0.1299, 0.0928, 0.0409, 0.1225]
w_entropy = [0.0302, 0.1081, 0.0342, 0.0306, 0.3507, 0.31, 0.0382, 0.0980]
pos_neg = [1, 1, 0, 0, 1, 1, 0, 0]  # 1代表正向指标，0代表负向指标

# 5. 数据归一化
def min_max_normalize(series, positive=True):
    """标准Min-Max归一化，加入极小值防止分母为0"""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return np.zeros(len(series))
    if positive:
        return (series - min_val) / (max_val - min_val + 1e-8)
    else:
        return (max_val - series) / (max_val - min_val + 1e-8)

binary_cols = {'低保户', '配偶情况'}  # 二值变量保持0/1业务含义，不参与动态极差缩放
norm_data = pd.DataFrame(index=remaining.index)

for i, col in enumerate(indicators):
    if col in binary_cols:
        norm_data[col] = remaining[col].astype(float)
    else:
        norm_data[col] = min_max_normalize(remaining[col], positive=pos_neg[i])

print("数据归一化完成（二值变量已做保留处理）。")

# 6. 最大余额法分配算法 
def distribute_budget(scores, budget, multiple=100):
    """
    采用最大余额法（Largest Remainder Method）变体：
    1. 确保每个人分配金额是 multiple (100) 的整倍数
    2. 确保最终总金额严格等于 budget，误差为0
    3. 优先补偿因取整损失精度的个体，兼顾数学公平性
    """
    if len(scores) == 0 or budget <= 0:
        return np.zeros(len(scores), dtype=int)
    
    total_score = scores.sum()
    if total_score == 0:
        # 如果所有人综合得分均为0，则均分预算
        base = (budget // len(scores)) // multiple * multiple
        alloc = np.full(len(scores), base)
        remainder = budget - alloc.sum()
        num_to_adjust = int(remainder // multiple)
        alloc[:num_to_adjust] += multiple
        return alloc

    # Step 1: 计算按得分比例划分的精确理论份额
    exact_shares = (scores / total_score) * budget
    
    # Step 2: 向下取整到 100 的倍数（进行基础安全分配，确保绝不超支）
    alloc = (exact_shares // multiple) * multiple
    alloc = alloc.astype(int)
    
    # Step 3: 计算由于取整而被“亏欠”的残差（余数）
    remainders = exact_shares - alloc
    
    # Step 4: 计算当前的预算缺口
    current_sum = alloc.sum()
    diff = budget - current_sum
    
    # 计算需要补齐的 100 元面额的个数
    num_slots_to_fill = int(round(diff / multiple))
    
    # Step 5: 按照残差从大到小排序，将缺口依次补齐
    if num_slots_to_fill > 0:
        sorted_indices = np.argsort(-remainders)  # 获取残差降序排列的索引
        chosen_indices = sorted_indices[:num_slots_to_fill]  # 挑出前 N 个亏欠最多的人
        alloc[chosen_indices] += multiple  # 每个人补偿 100 元

    return alloc

#7. 敏感性分析与融合权重计算
for lam in LAMBDAS:
    scores = np.zeros(len(norm_data))
    # 线性组合 AHP 权重与熵权
    w_fused = np.array(w_ahp) * lam + np.array(w_entropy) * (1 - lam)
    
    # 计算每位普通老人的综合发展得分
    for i, col in enumerate(indicators):
        scores += norm_data[col].values * w_fused[i]
    
    remaining[f'融合得分_λ{lam}'] = scores
    
    # 调用最大余额法进行金额整算分配
    alloc = distribute_budget(scores, remain_budget, multiple=100)
    remaining[f'补贴_λ{lam}'] = alloc
    
    # 为高剥夺群体同步创建对应的敏感性列，保持合并结构一致
    high_deprived[f'融合得分_λ{lam}'] = 1.0  # 高剥夺群体不参与评分，设为常数项
    high_deprived[f'补贴_λ{lam}'] = HIGH_DEPRIVED_AMOUNT

# 8. 数据合并与排序 
final_df = pd.concat([high_deprived, remaining], axis=0, sort=False)
final_df = final_df.sort_values(by='人员序号').reset_index(drop=True)

#  9. 最终预算闭环检查与保存
print("\n" + "="*70)
print(" 最终预算执行与敏感性分析检查")
print("="*70)
print(f"总预算总体控制目标: {TOTAL_BUDGET} 元")

for lam in LAMBDAS:
    total_alloc = final_df[f'补贴_λ{lam}'].sum()
    error = total_alloc - TOTAL_BUDGET
    print(f"方案 [ λ = {lam} ] 实际总发放: {total_alloc} 元 | 统计误差: {error} 元")

# 检查是否每个人都是100的倍数
for lam in LAMBDAS:
    non_100_count = np.sum(final_df[f'补贴_λ{lam}'] % 100 != 0)
    if non_100_count == 0:
        print(f"方案 [ λ = {lam} ] 约束合规性: 所有人补贴金额均满足100的倍数要求。")
    else:
        print(f"方案 [ λ = {lam} ] 约束合规性: 存在 {non_100_count} 条数据不满足100的倍数！")

# 保存结果至 Excel
output_path = '养老金补贴分配最终结果.xlsx'
with pd.ExcelWriter(output_path) as writer:
    final_df.to_excel(writer, sheet_name='最终分配结果', index=False)
    high_deprived.to_excel(writer, sheet_name='高剥夺群体明细', index=False)
    remaining.to_excel(writer, sheet_name='普通群体精细分配', index=False)

print("="*70)
print(f"建模数据处理完毕！结果已成功写回: {output_path}")
print("运行前10条样本预览：")
show_cols = ['人员序号', 'MPI_剥夺分', '融合得分_λ0.5', '补贴_λ0.5', '补贴_λ0.6', '补贴_λ0.7']
# 兼容列名映射输出
preview_df = final_df.copy()
if 'MPI_剥夺分数' in preview_df.columns:
    preview_df = preview_df.rename(columns={'MPI_剥夺分数': 'MPI_剥夺分'})
print(preview_df[show_cols].head(10).to_string(index=False))

