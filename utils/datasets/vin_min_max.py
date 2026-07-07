import json

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 寻找score的最大和最小值
def find_score_extremes(data):
    min_score = float('inf')  # 初始化为正无穷大
    max_score = float('-inf') # 初始化为负无穷大
    
    for entry in data.values():
        if 'score' in entry:
            score = entry['score']
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score

    return min_score, max_score

# 主函数
def main():
    file_path = '/root/autodl-tmp/PP_generate_v1/data/final_vina_results.json'  # 指定文件路径
    data = read_json_file(file_path)
    min_score, max_score = find_score_extremes(data)
    print(f"最小score: {min_score}")
    print(f"最大score: {max_score}")

if __name__ == '__main__':
    main()
