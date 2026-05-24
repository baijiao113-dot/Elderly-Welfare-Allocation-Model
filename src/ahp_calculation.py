"""
AHP指标权重构建
"""
import numpy as np
import pandas as pd

def ahp_weight(priority_scores, criteria_names=None):
    """AHP局部权重计算 + 一致性检验"""
    n = len(priority_scores)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = priority_scores[i] / priority_scores[j]
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eig_idx = np.argmax(np.real(eigenvalues))
    lambda_max = np.real(eigenvalues[max_eig_idx])
    weight = np.real(eigenvectors[:, max_eig_idx])
    weight = weight / np.sum(weight)
    
    CI = (lambda_max - n) / (n - 1)
    RI_dict = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
    RI = RI_dict.get(n, 1.45)
    CR = CI / RI if RI > 0 else 0
    
    print(f"   λ_max = {lambda_max:.4f} | CI = {CI:.4f} | CR = {CR:.4f} {'✅ 通过' if CR < 0.1 else '❌ 需要调整'}")
    
    if criteria_names:
        for name, w in zip(criteria_names, weight):
            print(f"     └─ {name:14} : {w:.4f} ({w*100:5.2f}%)")
    return weight


#1. 定义层次结构
hierarchy = {
    "A健康需求": {
        "weight": 0.5396,
        "sub_indicators": ["年龄", "身体状况"],
        "priorities": [1, 3]
    },
    "B经济需求": {
        "weight": 0.2970,
        "sub_indicators": ["每月养老金", "房产数目", "是否低保户", "人才层级"],
        "priorities": [3, 1, 7, 5]
    },
    "C家庭支持": {
        "weight": 0.1634,
        "sub_indicators": ["子女个数", "配偶情况"],
        "priorities": [1, 3]
    }
}

#2. 读取人员数据
file_path = "D:\math model\数学建模1.xlsx"

df_data = pd.read_excel(file_path, sheet_name="数学建模")
print(f"✅ 成功读取人员数据，共 {len(df_data)} 条记录")

#  3. 计算权重 
print("\n" + "="*70)
print("AHP 权重计算中...\n")

global_weights = {}

for level1_name, data in hierarchy.items():
    print(f"【{level1_name}】 (一级权重: {data['weight']:.4f})")
    
    local_w = ahp_weight(data["priorities"], data["sub_indicators"])
    data["local_weights"] = local_w
    
    # 计算全局权重
    for sub_name, lw in zip(data["sub_indicators"], local_w):
        gw = lw * data["weight"]
        global_weights[sub_name] = gw
        print(f"     → 全局权重: {gw:.5f} ({gw*100:6.2f}%)")
    print("-" * 60)

#4. 全局权重排序
print("\n🎯 最终全局权重排序（从高到低）:")
sorted_global = sorted(global_weights.items(), key=lambda x: x[1], reverse=True)
for name, w in sorted_global:
    print(f"{name:14} : {w:.5f} ({w*100:6.2f}%)")

#  5. 指标列名映射 
col_map = {
    "年龄": "年龄",
    "身体状况": "身体状况",
    "每月养老金": "每月养老金（元）",
    "房产数目": "房产数目",
    "是否低保户": "是否为低保户",
    "人才层级": "是否为国家、省、市级人才",   # 注意：Excel中是这个列
    "子女个数": "子女个数",
    "配偶情况": "当前是否是配偶配偶是否健在"
}



# 6. 输出与保存 
print("\n✅ 前10名结果：")
print(df_data[["人员序号", "综合得分", "排名"]].head(10).to_string(index=False))

# 保存结果
output_file = "数学建模_最终评分结果.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df_data.to_excel(writer, sheet_name="评分结果", index=False)
    
    # 保存权重表
    weight_df = pd.DataFrame(sorted_global, columns=["指标名称", "全局权重"])
    weight_df.to_excel(writer, sheet_name="全局权重", index=False)

print(f"\n所有结果已保存至：{output_file}")

