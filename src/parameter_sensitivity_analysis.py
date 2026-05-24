# -*- coding: utf-8 -*-
"""
lamda数敏感性分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 解决 matplotlib 中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 读取数据
file_path = "养老金补贴分配最终结果.xlsx"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"未找到文件：{file_path}，请确保该 CSV 文件与本脚本在同一文件夹下。")
    raise

# 定义分析的列名
lambda_cols = ['补贴_λ0.5', '补贴_λ0.6', '补贴_λ0.7']

# =============================================================================
# 一、 数据计算与分析
# =============================================================================

# 1. 宏观描述性统计
desc_stats = df[lambda_cols].describe()
# 动态添加总财政预算行
total_budget = df[lambda_cols].sum()
desc_stats.loc['总财政预算需求(元)'] = total_budget

# 2. Spearman 秩相关系数矩阵
corr_matrix, _ = spearmanr(df[lambda_cols])
corr_df = pd.DataFrame(corr_matrix, 
                       index=['λ=0.5 情景', 'λ=0.6 情景', 'λ=0.7 情景'], 
                       columns=['λ=0.5 情景', 'λ=0.6 情景', 'λ=0.7 情景'])

# 3. 群体响应分析（根据 MPI_剥夺分数 划分高剥夺与普通群体）
# 根据前文数据结构，剥夺分数 >= 4 为高剥夺群体
df['群体分类'] = np.where(df['MPI_剥夺分数'] >= 4, '高剥夺群体', '普通群体')
group_analysis = df.groupby('群体分类')[lambda_cols].mean().reset_index()

# =============================================================================
# 二、 将结果直接导出至 Excel 表格
# =============================================================================
output_excel = "养老金模型敏感性分析结果.xlsx"
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    desc_stats.to_excel(writer, sheet_name='1.宏观描述性统计')
    corr_df.to_excel(writer, sheet_name='2.排名相关性矩阵')
    group_analysis.to_excel(writer, sheet_name='3.群体响应分析', index=False)

print(f" 成功：分析数据已保存至 Excel 文件：'{output_excel}'")

# =============================================================================
# 三、 学术图表绘制与本地保存
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 图1：补贴分布的核密度估计图（KDE）
for col in lambda_cols:
    label_text = col.replace('补贴_', '')
    sns.kdeplot(df[col], label=label_text, ax=axes[0], fill=True, alpha=0.15)
axes[0].set_title('图1：补贴资金社会分布密度演变', fontsize=12, fontweight='bold')
axes[0].set_xlabel('补贴金额 (元)', fontsize=10)
axes[0].set_ylabel('密度', fontsize=10)
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# 图2：目标群体补贴的政策响应轨迹（折线图）
melted_df = df.melt(id_vars=['群体分类'], value_vars=lambda_cols, var_name='Lambda', value_name='Subsidy')
melted_df['Lambda'] = melted_df['Lambda'].str.extract(r'(\d+\.\d+)').astype(float)
sns.lineplot(data=melted_df, x='Lambda', y='Subsidy', hue='群体分类', marker='o', linewidth=2.5, ax=axes[1], errorbar=None)
axes[1].set_title('图2：目标群体补贴对 λ 的政策响应轨迹', fontsize=12, fontweight='bold')
axes[1].set_xlabel('政策偏好系数 (λ)', fontsize=10)
axes[1].set_ylabel('平均补贴金额 (元)', fontsize=10)
axes[1].set_xticks([0.5, 0.6, 0.7])
axes[1].grid(True, linestyle='--', alpha=0.5)

# 图3：秩相关性热力图
sns.heatmap(corr_df, annot=True, cmap='Blues', vmin=0.95, vmax=1.0, fmt=".4f", ax=axes[2], cbar=True)
axes[2].set_title('图3：不同 λ 配置下的分配排名相关性', fontsize=12, fontweight='bold')

plt.tight_layout()

# 显式保存图片到本地
output_image = "lambda_sensitivity_analysis.png"
plt.savefig(output_image, dpi=300, bbox_inches='tight')
print(f" 成功：敏感性分析学术图表已保存为：'{output_image}'")

# 强制在 IDE 中弹出窗口展示图片
plt.show()
