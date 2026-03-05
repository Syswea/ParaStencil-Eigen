import random

# 参数设置
N = 512
d = 512
num_matrices = 3  # Q, K, V
total_rows = num_matrices * N

# 写入文件
with open("data.in", "w") as f:
    for i in range(total_rows):
        # 生成一行 512 个随机整数（范围 0-100）
        row_data = [str(random.randint(0, 100)) for _ in range(d)]
        
        # 将这一行拼接成空格分隔的字符串
        line = " ".join(row_data)
        
        # 写入文件并换行
        f.write(line + "\n")

print(f"成功生成 data.in: 共 {total_rows} 行，每行 {d} 个整数。")