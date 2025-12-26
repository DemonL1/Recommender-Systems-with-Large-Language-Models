import pandas as pd
import json
import os
from tqdm import tqdm

# ===================== 配置参数 =====================
INPUT_CSV = "fine_tune_dataset_doubao_no_score.csv"  # 输入CSV文件路径
OUTPUT_JSONL = "fine_tune_dataset_instruction.jsonl"  # 输出JSONL文件路径（每行一个JSON对象）
OUTPUT_JSON = "fine_tune_dataset_instruction.json"  # 输出标准JSON文件路径（可选）

# 输出格式选择
# "openai" - OpenAI微调格式: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
# "simple" - 简单格式: {"prompt": "...", "completion": "..."}
# "instruction" - 指令格式: {"instruction": "...", "output": "..."}
OUTPUT_FORMAT = "instruction"  # 可选: "openai", "simple", "instruction"

# 是否过滤空数据
FILTER_EMPTY = True

# 是否清理response中的"###"标记
CLEAN_RESPONSE = True


def clean_response(text):
    """清理response文本，移除开头的###标记、参考编号和多余的空行"""
    import re
    
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # 按行处理，同时移除###标记和参考编号
    lines = text.split("\n")
    cleaned_lines = []
    found_reference = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # 如果已经遇到参考编号，跳过所有后续行
        if found_reference:
            continue
        
        # 检查是否是参考编号标记
        if re.match(r'\[参考编号', line_stripped) or re.match(r'\[参考\d+\]', line_stripped):
            found_reference = True
            continue
        
        # 处理###标记
        if line_stripped.startswith("###"):
            if len(line_stripped) > 3:
                # 如果###后面有内容，保留内容部分
                cleaned_lines.append(line_stripped[3:].strip())
            # 否则跳过这一行
            continue
        
        # 保留其他行
        if line_stripped:
            cleaned_lines.append(line_stripped)
    
    text = "\n".join(cleaned_lines)
    
    # 额外清理：移除行内参考编号（如果还有残留）
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[参考\d+\]', '', text)
    text = re.sub(r'\[参考[^\]]+\]', '', text)
    text = re.sub(r'\[参考编号\].*', '', text, flags=re.DOTALL)
    
    # 清理多余的空行
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # 合并行内多个空格为一个（但保留换行）
    lines = [re.sub(r' +', ' ', line) for line in lines]
    text = "\n".join(lines)
    # 多个连续换行合并为两个
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()


def convert_to_openai_format(prompt, response):
    """转换为OpenAI微调格式"""
    return {
        "messages": [
            {"role": "user", "content": str(prompt).strip()},
            {"role": "assistant", "content": str(response).strip()}
        ]
    }


def convert_to_simple_format(prompt, response):
    """转换为简单格式"""
    return {
        "prompt": str(prompt).strip(),
        "completion": str(response).strip()
    }


def convert_to_instruction_format(prompt, response):
    """转换为指令格式"""
    return {
        "instruction": "请根据input中的用户喜好生成英文电影推荐回答",
        "input": str(prompt).strip(),
        "output": str(response).strip()
    }


def main():
    # 检查输入文件
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 未找到输入文件：{INPUT_CSV}")
        return
    
    # 读取CSV文件
    print(f"📖 正在读取CSV文件：{INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, dtype=str, encoding="utf-8-sig")
        print(f"✅ 成功读取 {len(df)} 条记录")
    except Exception as e:
        print(f"❌ 读取CSV文件失败：{str(e)}")
        return
    
    # 检查必要字段
    required_fields = ["prompt", "response"]
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        print(f"❌ CSV文件缺少必要字段：{missing_fields}")
        return
    
    # 选择转换函数
    if OUTPUT_FORMAT == "openai":
        convert_func = convert_to_openai_format
        print("📝 使用OpenAI微调格式")
    elif OUTPUT_FORMAT == "simple":
        convert_func = convert_to_simple_format
        print("📝 使用简单格式")
    elif OUTPUT_FORMAT == "instruction":
        convert_func = convert_to_instruction_format
        print("📝 使用指令格式")
    else:
        print(f"❌ 不支持的输出格式：{OUTPUT_FORMAT}")
        return
    
    # 转换数据
    print(f"🔄 正在转换数据...")
    json_data = []
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="转换进度"):
        prompt = row.get("prompt", "")
        response = row.get("response", "")
        
        # 检查数据有效性
        if FILTER_EMPTY:
            if pd.isna(prompt) or str(prompt).strip() == "":
                skipped += 1
                continue
            if pd.isna(response) or str(response).strip() == "":
                skipped += 1
                continue
            if str(prompt).strip() == "生成失败":
                skipped += 1
                continue
        
        # 清理response
        if CLEAN_RESPONSE:
            response = clean_response(response)
            if not response:
                skipped += 1
                continue
        
        # 转换为JSON格式
        try:
            json_obj = convert_func(prompt, response)
            json_data.append(json_obj)
        except Exception as e:
            print(f"⚠️ 转换第{idx+1}行时出错：{str(e)}")
            skipped += 1
            continue
    
    print(f"✅ 转换完成：成功 {len(json_data)} 条，跳过 {skipped} 条")
    
    # 保存为JSONL格式（推荐用于微调）
    print(f"💾 正在保存JSONL文件：{OUTPUT_JSONL}")
    try:
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for item in json_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ JSONL文件保存成功：{OUTPUT_JSONL}")
        print(f"   文件大小：{os.path.getsize(OUTPUT_JSONL) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ 保存JSONL文件失败：{str(e)}")
        return
    
    # 可选：保存为标准JSON格式
    if OUTPUT_JSON:
        print(f"💾 正在保存JSON文件：{OUTPUT_JSON}")
        try:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"✅ JSON文件保存成功：{OUTPUT_JSON}")
            print(f"   文件大小：{os.path.getsize(OUTPUT_JSON) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"⚠️ 保存JSON文件失败：{str(e)}（不影响JSONL文件）")
    
    # 显示示例
    if json_data:
        print(f"\n📋 数据示例（前3条）：")
        for i, item in enumerate(json_data[:3], 1):
            print(f"\n--- 示例 {i} ---")
            print(json.dumps(item, ensure_ascii=False, indent=2))
    
    print(f"\n🎉 转换完成！")
    print(f"📊 统计：")
    print(f"   - 总记录数：{len(df)}")
    print(f"   - 成功转换：{len(json_data)}")
    print(f"   - 跳过记录：{skipped}")
    print(f"\n💡 提示：")
    print(f"   - JSONL格式（{OUTPUT_JSONL}）通常用于微调训练")
    print(f"   - 如需更改输出格式，请修改脚本中的 OUTPUT_FORMAT 参数")


if __name__ == "__main__":
    main()

