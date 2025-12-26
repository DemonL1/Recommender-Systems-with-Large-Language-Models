import os
import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ===================== 配置参数 =====================
INPUT_CSV = "test1.csv"  # 输入数据集路径
OUTPUT_CSV = "fine_tune_dataset_doubao_no_score.csv"  # 输出结果路径

# 火山引擎ARK配置
ARK_API_KEY = os.environ.get("ARK_API_KEY") or "879880f6-67c7-424d-a3fd-35fe0db260c6"  # 你的API Key
BOT_ID = "bot-20251116182710-n2cvp"  # 你的智能体ID
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/bots"

# 调用控制参数
BATCH_SIZE = 5  # 增加批次大小，提升速度
DELAY = 1  # 减少延迟，提升速度
MAX_RETRIES = 5  # 增加重试次数
RATE_LIMIT_WAIT = 60  # 遇到429限流错误时等待时间（秒）
MAX_WORKERS = 3  # 并发线程数，提升速度


# ===================== 初始化客户端 =====================
client = OpenAI(
    base_url=BASE_URL,
    api_key=ARK_API_KEY
)


# ===================== 核心函数：生成推荐语录 =====================
def generate_movie_quote(row):
    """生成不含相似性得分的推荐语录，保留联网查询增强功能"""
    try:
        # 检查必要字段是否存在（不再需要rec_overview_en）
        required_fields = ['liked_1_name', 'liked_2_name', 'liked_3_name', 'rec_name']
        missing_fields = [f for f in required_fields if f not in row.index or pd.isna(row.get(f)) or str(row.get(f, '')).strip() == '']
        if missing_fields:
            return None, f"缺少必要字段或字段为空：{missing_fields}"
        
        # 整理电影信息（隐藏相似性得分，使用get方法避免KeyError）
        liked_movies = [
            {
                "name": row.get('liked_1_name', ''),
                "genres": row.get('liked_1_genres', ''),
                "keywords": row.get('liked_1_keywords', ''),
                "director": row.get('liked_1_director', '')
            },
            {
                "name": row.get('liked_2_name', ''),
                "genres": row.get('liked_2_genres', ''),
                "keywords": row.get('liked_2_keywords', ''),
                "director": row.get('liked_2_director', '')
            },
            {
                "name": row.get('liked_3_name', ''),
                "genres": row.get('liked_3_genres', ''),
                "keywords": row.get('liked_3_keywords', ''),
                "director": row.get('liked_3_director', '')
            }
        ]
        rec_movie = {
            "name": row.get('rec_name', ''),
            "genres": row.get('rec_genres', ''),
            "keywords": row.get('rec_keywords', ''),
            "director": row.get('rec_director', '')
            # 移除简介，避免prompt过长导致响应截断
        }
    except Exception as e:
        return None, f"数据整理失败：{type(e).__name__}: {str(e)}"

    # 构建提示词（明确禁止出现分数，移除简介，要求英文输出）
    prompt_content = f"""
    
    User's favorite movies:
    1. 《{liked_movies[0]['name']}》: Genre: {liked_movies[0]['genres']}, Keywords: {liked_movies[0]['keywords']}, Director: {liked_movies[0]['director']}
    2. 《{liked_movies[1]['name']}》: Genre: {liked_movies[1]['genres']}, Keywords: {liked_movies[1]['keywords']}, Director: {liked_movies[1]['director']}
    3. 《{liked_movies[2]['name']}》: Genre: {liked_movies[2]['genres']}, Keywords: {liked_movies[2]['keywords']}, Director: {liked_movies[2]['director']}

    Recommended movie:
    《{rec_movie['name']}》: Genre: {rec_movie['genres']}, Keywords: {rec_movie['keywords']}, Director: {rec_movie['director']}

    """

    # 调用模型（带重试）
    last_error = None
    is_rate_limit_error = False
    
    for retry in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=BOT_ID,
                messages=[
                    {"role": "system", "content": "You are a professional movie recommendation expert. Generate natural and engaging recommendations in English only. Never mention any scores or reference numbers."},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0.85,
                max_tokens=2000,  # 进一步增加token限制，确保Response完整
                stream=False
            )
            
            # 检查响应
            if not response or not hasattr(response, 'choices') or len(response.choices) == 0:
                print(f"\n⚠️ API响应格式异常（重试{retry+1}/{MAX_RETRIES}）：choices为空")
                if retry == MAX_RETRIES - 1:
                    print(f"   完整响应：{response}")
                time.sleep(DELAY * (retry + 1))
                continue
            
            # 获取内容
            message = response.choices[0].message
            if not hasattr(message, 'content') or not message.content:
                print(f"\n⚠️ API返回内容为空（重试{retry+1}/{MAX_RETRIES}）")
                if retry == MAX_RETRIES - 1:
                    print(f"   message对象：{message}")
                time.sleep(DELAY * (retry + 1))
                continue
            
            # 检查finish_reason，如果是length说明被截断
            finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
            if finish_reason == 'length':
                print(f"\n⚠️ API响应被截断（达到token限制），尝试增加max_tokens（重试{retry+1}/{MAX_RETRIES}）")
                # 如果被截断，继续尝试，但会在解析时做容错处理
            
            raw_output = message.content.strip()
            if not raw_output:
                print(f"\n⚠️ API返回内容为空字符串（重试{retry+1}/{MAX_RETRIES}）")
                time.sleep(DELAY * (retry + 1))
                continue
            
            # 成功获取内容（即使被截断也继续，在解析时处理）
            break
            
        except Exception as e:
            last_error = e
            error_type = type(e).__name__
            error_msg = str(e)
            
            # 特殊处理429限流错误
            if "429" in str(e) or "RateLimitError" in error_type or "SetLimitExceeded" in error_msg or "inference limit" in error_msg.lower():
                is_rate_limit_error = True
                wait_time = RATE_LIMIT_WAIT * (retry + 1)  # 60秒、120秒、180秒...
                print(f"\n⚠️ API限流错误（重试{retry+1}/{MAX_RETRIES}）：账户已达到推理限制")
                print(f"   等待 {wait_time} 秒后重试...（建议检查账户限制设置或关闭'安全体验模式'）")
                time.sleep(wait_time)
            else:
                print(f"\n⚠️ API调用失败（重试{retry+1}/{MAX_RETRIES}）：{error_type}: {error_msg[:200]}")
                if retry == MAX_RETRIES - 1:
                    import traceback
                    print(f"   详细错误：{traceback.format_exc()[:500]}")
                time.sleep(DELAY * (retry + 1))
    else:
        # 所有重试都失败
        if is_rate_limit_error:
            return None, "API限流错误：账户已达到推理限制，请检查账户设置或稍后重试"
        elif last_error:
            error_msg = str(last_error)
            if "429" in error_msg or "RateLimitError" in str(type(last_error).__name__) or "SetLimitExceeded" in error_msg:
                return None, "API限流错误：账户已达到推理限制，请检查账户设置或稍后重试"
        return None, "API调用失败：所有重试均失败"

    # 解析结果（容错处理，支持被截断的内容）
    if not raw_output:
        return None, "API返回内容为空"
    
    # 尝试多种分隔符
    separators = ["###", "---", "===", "\n\nResponse", "\nResponse"]
    prompt = None
    response = None
    
    for sep in separators:
        if sep in raw_output:
            parts = raw_output.split(sep, 1)
            if len(parts) >= 2:
                prompt = parts[0].strip()
                response = parts[1].strip()
                break
            elif len(parts) == 1 and sep == "###":
                # 可能只有Prompt部分，尝试查找Response标记
                if "Response" in raw_output.lower():
                    response_idx = raw_output.lower().find("response")
                    prompt = raw_output[:response_idx].strip()
                    response = raw_output[response_idx:].strip()
                    # 移除Response标签
                    response = response.replace("Response：", "").replace("Response:", "").strip()
                    break
    
    # 如果没找到分隔符，尝试智能解析
    if not prompt or not response:
        # 尝试查找Prompt和Response关键词
        prompt_markers = ["Prompt：", "Prompt:", "prompt：", "prompt:"]
        response_markers = ["Response：", "Response:", "response：", "response:"]
        
        prompt_start = -1
        response_start = -1
        
        for marker in prompt_markers:
            idx = raw_output.find(marker)
            if idx != -1:
                prompt_start = idx + len(marker)
                break
        
        for marker in response_markers:
            idx = raw_output.find(marker)
            if idx != -1:
                response_start = idx + len(marker)
                break
        
        if prompt_start != -1 and response_start != -1:
            # 提取Prompt（从Prompt标记到Response标记之间）
            prompt = raw_output[prompt_start:response_start - len(response_markers[0])].strip()
            # 提取Response（从Response标记到结尾）
            response = raw_output[response_start:].strip()
        elif response_start != -1:
            # 只有Response标记，Prompt可能是前面的内容
            prompt = raw_output[:response_start - len(response_markers[0])].strip()
            response = raw_output[response_start:].strip()
        else:
            # 完全无法解析，使用兜底方案（英文）
            default_prompt = f"I really love 《{liked_movies[0]['name']}》, 《{liked_movies[1]['name']}》 and 《{liked_movies[2]['name']}》. Can you recommend similar movies?"
            cleaned_response = raw_output.strip().replace("分", "").replace("分数", "")
            if cleaned_response and len(cleaned_response) > 20:  # 至少要有一定长度
                return default_prompt, cleaned_response
            else:
                return None, f"解析失败：无法提取有效内容。原始输出：{raw_output[:300]}"
    
    # 清理标签
    prompt = prompt.replace("Prompt：", "").replace("Prompt:", "").strip()
    response = response.replace("Response：", "").replace("Response:", "").strip()
    
    # 移除参考编号部分（API可能会在末尾添加references）
    # 移除 "[参考编号] 资料名称" 及其后面的所有内容
    response = re.sub(r'\[参考编号\][^\n]*\n.*', '', response, flags=re.DOTALL)
    # 移除以 "[数字]" 开头的行（参考编号列表）
    response = re.sub(r'\n\s*\[\d+\][^\n]*', '', response)
    # 移除行内参考编号
    response = re.sub(r'\[\d+\]', '', response)
    response = re.sub(r'\[参考\d+\]', '', response)
    response = re.sub(r'\[参考[^\]]+\]', '', response)
    # 移除 "[参考编号]" 标记及其后面的内容
    response = re.sub(r'\[参考编号\].*', '', response, flags=re.DOTALL)
    # 清理多余的空白字符和换行
    response = re.sub(r'\n\s*\n+', '\n\n', response)  # 多个连续换行合并为两个
    response = re.sub(r'\s+', ' ', response)  # 多个空格合并为一个
    response = response.strip()
    
    # 移除分数相关表述
    response = response.replace("分", "").replace("分数", "").replace("相似性", "风格相似度")
    
    # 验证提取的内容
    if not prompt or len(prompt) < 5:
        return None, f"解析失败：Prompt太短或为空。原始输出：{raw_output[:300]}"
    
    if not response or len(response) < 10:
        # 即使Response被截断，也尝试使用（至少要有一定内容）
        if len(response) >= 5:
            # Response被截断但有一定内容，添加提示
            response = response + "...（内容可能被截断）"
        else:
            return None, f"解析失败：Response太短或为空。原始输出：{raw_output[:300]}"
    
    return prompt, response


# ===================== 主流程（断点续跑） =====================
def main():
    # 读取输入数据
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 未找到输入文件：{INPUT_CSV}")
        return
    
    # 检查必要字段（移除rec_overview_en依赖，不再需要简介）
    required_fields = ["liked_1_name", "rec_name"]
    
    # 读取输入数据
    input_df = pd.read_csv(INPUT_CSV, dtype=str)
    print(f"✅ 读取输入数据成功：共{len(input_df)}条记录")
    
    missing = [f for f in required_fields if f not in input_df.columns]
    if missing:
        print(f"❌ 缺少字段：{missing}")
        return
    
    # 尝试读取输出文件（如果存在，用于断点续跑）
    if os.path.exists(OUTPUT_CSV):
        try:
            output_df = pd.read_csv(OUTPUT_CSV, dtype=str)
            print(f"✅ 读取输出文件成功：共{len(output_df)}条记录")
            
            # 合并数据：以输入数据为主，用输出数据中的prompt和response填充
            # 使用user_id和rec_id作为唯一标识（如果存在）
            if "user_id" in input_df.columns and "rec_id" in input_df.columns:
                # 创建合并键
                input_df["_merge_key"] = input_df["user_id"].astype(str) + "_" + input_df["rec_id"].astype(str)
                output_df["_merge_key"] = output_df["user_id"].astype(str) + "_" + output_df["rec_id"].astype(str)
                
                # 合并prompt和response
                if "prompt" in output_df.columns:
                    prompt_map = dict(zip(output_df["_merge_key"], output_df["prompt"]))
                    input_df["prompt"] = input_df["_merge_key"].map(prompt_map).fillna("")
                if "response" in output_df.columns:
                    response_map = dict(zip(output_df["_merge_key"], output_df["response"]))
                    input_df["response"] = input_df["_merge_key"].map(response_map).fillna("")
                
                # 删除临时列
                input_df = input_df.drop(columns=["_merge_key"])
            else:
                # 如果没有唯一标识，按索引合并（假设顺序一致）
                if len(input_df) == len(output_df):
                    if "prompt" in output_df.columns:
                        input_df["prompt"] = output_df["prompt"].fillna("")
                    if "response" in output_df.columns:
                        input_df["response"] = output_df["response"].fillna("")
                else:
                    print("⚠️ 输入和输出文件记录数不一致，无法合并，将从头开始")
        except Exception as e:
            print(f"⚠️ 读取输出文件失败：{str(e)}，将从头开始")
    else:
        print("ℹ️ 输出文件不存在，将从头开始生成")
    
    df = input_df.copy()
    
    # 初始化输出字段（如果不存在）
    for col in ["prompt", "response"]:
        if col not in df.columns:
            df[col] = ""

    # 筛选未生成的记录（断点续跑：跳过已成功生成的记录）
    # 已成功生成的条件：prompt和response都不为空，且不是"生成失败"
    ungenerated_mask = (
        (df["prompt"].isna()) | 
        (df["prompt"] == "") | 
        (df["response"] == "") |
        (df["prompt"] == "生成失败")  # 也重新生成失败的记录
    )
    
    ungenerated = df[ungenerated_mask].copy()
    print(f"📌 待生成：{len(ungenerated)}条 | 已生成：{len(df) - len(ungenerated)}条")
    
    # 显示生成状态统计
    success_count = len(df[(df["prompt"] != "") & (df["prompt"] != "生成失败") & (df["prompt"].notna()) & 
                          (df["response"] != "") & (df["response"].notna())])
    fail_count = len(df[df["prompt"] == "生成失败"])
    empty_count = len(ungenerated)
    print(f"   其中：成功 {success_count}条 | 失败 {fail_count}条 | 待处理 {empty_count}条")

    if len(ungenerated) == 0:
        print("🎉 所有记录已生成完成！")
        return

    # 创建线程锁，保护DataFrame写入操作
    df_lock = threading.Lock()
    
    # 并发生成函数
    def process_row(row_data):
        """处理单行数据"""
        idx, row = row_data
        result = generate_movie_quote(row)
        return result
    
    # 使用线程池并发处理
    total = len(ungenerated)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_row = {
            executor.submit(process_row, (idx, row)): (idx, row) 
            for idx, row in ungenerated.iterrows()
        }
        
        # 处理完成的任务
        completed = 0
        with tqdm(total=total, desc="生成推荐语录") as pbar:
            for future in as_completed(future_to_row):
                original_idx, row = future_to_row[future]
                try:
                    result = future.result()
                    
                    # 处理返回值：可能是(prompt, response)或(None, error_message)
                    if result and len(result) == 2:
                        prompt, response = result
                        if prompt and response and prompt != "生成失败":
                            # 使用锁保护DataFrame写入
                            with df_lock:
                                df.at[original_idx, "prompt"] = prompt
                                df.at[original_idx, "response"] = response
                            # 输出推荐内容预览
                            preview = response[:200].replace('\n', ' ')
                            if len(response) > 200:
                                preview += "..."
                            tqdm.write(f"✅ 用户{row.get('user_id', 'N/A')}：生成成功")
                            tqdm.write(f"   📝 推荐内容：{preview}")
                        else:
                            # 失败时保存错误信息（但不标记为"生成失败"，以便下次重试）
                            error_msg = response if response else "未知错误"
                            # 如果是限流错误，不标记为失败，留空以便下次重试
                            with df_lock:
                                if "限流" in error_msg or "429" in error_msg or "SetLimitExceeded" in error_msg:
                                    df.at[original_idx, "prompt"] = ""  # 留空，下次重试
                                    df.at[original_idx, "response"] = ""
                                    tqdm.write(f"⏸️ 用户{row.get('user_id', 'N/A')}：遇到限流，已跳过，下次重试")
                                else:
                                    df.at[original_idx, "prompt"] = "生成失败"
                                    df.at[original_idx, "response"] = error_msg[:200]  # 限制长度
                                    tqdm.write(f"❌ 用户{row.get('user_id', 'N/A')}：生成失败 - {error_msg[:100]}")
                    else:
                        with df_lock:
                            df.at[original_idx, "prompt"] = "生成失败"
                            df.at[original_idx, "response"] = "返回值格式错误"
                        tqdm.write(f"❌ 用户{row.get('user_id', 'N/A')}：生成失败 - 返回值格式错误")
                    
                    completed += 1
                    pbar.update(1)
                    
                    # 每处理一定数量就保存一次（确保断点续跑）
                    if completed % BATCH_SIZE == 0:
                        with df_lock:
                            try:
                                df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
                            except Exception as e:
                                print(f"\n⚠️ 保存文件失败：{str(e)}")
                                # 尝试保存到临时文件
                                try:
                                    df.to_csv(OUTPUT_CSV + ".backup", index=False, encoding="utf-8-sig")
                                    print(f"   已保存到备份文件：{OUTPUT_CSV}.backup")
                                except:
                                    print(f"   备份保存也失败，数据可能丢失！")
                    
                    # 短暂延迟，避免请求过快
                    time.sleep(DELAY / MAX_WORKERS)
                    
                except Exception as e:
                    tqdm.write(f"❌ 处理用户{row.get('user_id', 'N/A')}时发生异常：{str(e)}")
                    with df_lock:
                        df.at[original_idx, "prompt"] = "生成失败"
                        df.at[original_idx, "response"] = f"处理异常：{str(e)[:200]}"
                    completed += 1
                    pbar.update(1)
    
    # 最终保存
    try:
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"\n⚠️ 最终保存文件失败：{str(e)}")
        try:
            df.to_csv(OUTPUT_CSV + ".backup", index=False, encoding="utf-8-sig")
            print(f"   已保存到备份文件：{OUTPUT_CSV}.backup")
        except:
            print(f"   备份保存也失败，数据可能丢失！")

    # 最终统计
    success = len(df[(df["prompt"] != "") & (df["prompt"] != "生成失败")])
    fail = len(df[df["prompt"] == "生成失败"])
    print(f"\n🎉 生成完成！")
    print(f"📊 结果：总{len(df)}条 | 成功{success}条 | 失败{fail}条")
    print(f"输出文件：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()