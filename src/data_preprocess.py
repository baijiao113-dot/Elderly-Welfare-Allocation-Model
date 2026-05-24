# -*- coding: utf-8 -*-
"""
第一问: 决策树 
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

#  1. 读取数据
file_path = r"D:\math model\副本附件1.xlsx"

df_info = pd.read_excel(file_path, sheet_name="老年人员信息")
df_plan = pd.read_excel(file_path, sheet_name="三种分配方案")

df = df_info.copy()
df["方案一金额"] = df_plan.iloc[:259, 1].values
df["方案二金额"] = df_plan.iloc[:259, 4].values
df["方案三金额"] = df_plan.iloc[:259, 7].values

#  2. 特征工程 
edu_map = {"小学": 1, "初中": 2, "高中": 3, "本科": 4, "研究生": 5}
health_map = {"健康": 0, "慢性病": 1, "重大疾病": 2}
talent_map = {"否": 0, "市级": 1, "省级": 2, "国家级": 3}

df["学历数值"] = df["学历"].map(edu_map)
df["户籍数值"] = (df["户籍"] == "A市").astype(int)
df["配偶健在"] = (df["当前是否是配偶/配偶是否健在"] == "是").astype(int)
df["身体状况数值"] = df["身体状况"].map(health_map)
df["人才数值"] = df["是否为国家、省、市级人才"].map(talent_map)
df["低保数值"] = (df["是否为低保户"] == "是").astype(int)

features = ["每月养老金（元）", "年龄", "子女个数", "房产数目", 
            "学历数值", "户籍数值", "配偶健在", 
            "身体状况数值", "人才数值", "低保数值"]

X = df[features].fillna(df[features].median(numeric_only=True))

#  3. 生成纯净黑白决策树 
schemes = [
    ("方案一", "方案一金额", "decision_tree_plan1_clean.png"),
    ("方案二", "方案二金额", "decision_tree_plan2_clean.png"),
    ("方案三", "方案三金额", "decision_tree_plan3_clean.png")
]

# 设置中文字体与负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

for name, amount_col, save_name in schemes:
    y = df[amount_col]
    
    # 构建决策树模型
    tree = DecisionTreeRegressor(max_depth=4, min_samples_split=5, 
                                 min_samples_leaf=2, random_state=42)
    tree.fit(X, y)
    
    # 创建画布：纯白背景
    fig, ax = plt.subplots(figsize=(24, 14), facecolor='white')
    ax.set_facecolor('white')
    
    # 核心改动：使用 plot_tree 并自定义 bbox
    # filled=False 确保无颜色填充
    # impurity=False, precision=0 用于尽可能减少其他文本干扰（后续通过文本对象彻底清理）
    annotations = plot_tree(
        tree,
        feature_names=features,
        filled=False, 
        impurity=False,
        node_ids=False,
        proportion=False,
        precision=0,
        ax=ax,
        fontsize=12
    )
    
    # 遍历每个节点，彻底过滤文本
    for annotation in annotations:
        text = annotation.get_text()
        
        # 使用换行符分割，只提取第一行（即特征名称与判定条件，如 "年龄 <= 75"）
        # 如果是叶子节点（没有特征分裂），则保留 "value = xxx" 以明确分配结果，
        lines = text.split('\n')
        if lines:
            first_line = lines[0]
            # 如果是叶子节点，Sklearn默认第一行是 "value = xxx"
            if "value" in first_line:
                annotation.set_text(first_line)  # 保留方案最终分配的金额值
            else:
                annotation.set_text(first_line)  # 保留特征判定条件（如：年龄 <= 65）
        
        # 强制设置节点的视觉样式：纯白背景、纯黑边框、方形黑线框
        annotation.set_bbox(dict(
            boxstyle="square,pad=0.5",
            fc="white",          # FaceColor: 纯白
            ec="black",          # EdgeColor: 纯黑
            lw=1.5               # LineWidth: 边框粗细
        ))
    
    # 设置图表标题
    plt.title(f"{name} 决策树分配规则", fontsize=22, pad=30, weight='bold', color='black')
    
    # 隐藏坐标轴边框，使页面整体呈纯白色
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # 保存图片：确保分辨率与白底设置
    plt.tight_layout()
    plt.savefig(save_name, dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ {name} 纯净黑白版决策树已生成：{save_name}")

print("\n 全部完成！")
