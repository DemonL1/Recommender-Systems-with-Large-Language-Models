import pandas as pd
import numpy as np

# ===================== 配置文件路径 =====================
# 你的主数据集（需要添加简介的数据集）
MAIN_DATASET_PATH = "coze_input_full_unique_v3.csv"  # 包含rec_id字段的主数据
# 包含简介的数据集（需匹配的数据源）
OVERVIEW_DATASET_PATH = "movies_details_en_multi.csv"  # 包含movieId和overview_en字段
# 输出结果路径（添加简介后的数据集）
OUTPUT_PATH = "dataset_with_overview.csv"


# ===================== 核心逻辑 =====================
def add_recommendation_overview():
    # 1. 读取主数据集（包含推荐电影的rec_id）
    print(f"读取主数据集：{MAIN_DATASET_PATH}")
    main_df = pd.read_csv(MAIN_DATASET_PATH, dtype=str, encoding='gbk')  # 按字符串读取，避免ID格式问题，使用gbk编码
    
    # 检查主数据集是否包含必要字段
    if "rec_id" not in main_df.columns:
        print("错误：主数据集缺少'rec_id'字段，请检查文件！")
        return
    
    # 2. 读取简介数据集（包含movieId和overview_en）
    print(f"读取简介数据集：{OVERVIEW_DATASET_PATH}")
    # 尝试多种编码方式
    encodings = ['utf-8', 'latin-1', 'gbk', 'gb2312', 'cp1252']
    overview_df = None
    for enc in encodings:
        try:
            overview_df = pd.read_csv(OVERVIEW_DATASET_PATH, dtype=str, encoding=enc)
            print(f"成功使用 {enc} 编码读取简介数据集")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if overview_df is None:
        print("错误：无法使用常见编码读取简介数据集，请检查文件编码")
        return
    
    # 检查简介数据集是否包含必要字段
    required_overview_fields = ["movieId", "overview_en"]
    if not all(field in overview_df.columns for field in required_overview_fields):
        print(f"错误：简介数据集缺少必要字段，请确保包含{required_overview_fields}")
        return
    
    # 3. 清理数据（去除空格，避免匹配失败）
    main_df["rec_id"] = main_df["rec_id"].str.strip()  # 去除rec_id前后空格
    overview_df["movieId"] = overview_df["movieId"].str.strip()  # 去除movieId前后空格
    
    # 4. 构建movieId到overview_en的映射字典（加速匹配）
    overview_map = dict(zip(overview_df["movieId"], overview_df["overview_en"]))
    print(f"成功构建简介映射表，共包含{len(overview_map)}部电影的简介")
    
    # 5. 为每一行匹配简介（根据rec_id查找对应的overview_en）
    print("开始匹配推荐电影简介...")
    main_df["rec_overview_en"] = main_df["rec_id"].apply(
        lambda x: overview_map.get(x, np.nan)  # 若匹配不到，用NaN表示
    )
    
    # 6. 统计匹配结果
    matched_count = main_df["rec_overview_en"].notna().sum()
    total_count = len(main_df)
    print(f"匹配完成：共{total_count}条数据，成功匹配{matched_count}条简介，未匹配{total_count - matched_count}条")
    
    # 7. 保存结果
    main_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"结果已保存至：{OUTPUT_PATH}（新增字段'rec_overview_en'为推荐电影简介）")


if __name__ == "__main__":
    add_recommendation_overview()