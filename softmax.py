import math

# 配置参数与 C++ 一致
N, d = 512, 512

def softmax(row):
    # 数值稳定处理：减去最大值
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    sum_exps = sum(exps)
    return [x / sum_exps for x in exps]

def matrix_multiply(A, B_T, rows_A, cols_A, rows_B):
    # 计算 A * B^T (B_T 是已经转置的 B)
    result = [[0.0] * rows_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(rows_B):
            dot_product = sum(A[i][k] * B_T[j][k] for k in range(cols_A))
            result[i][j] = dot_product
    return result

def run_check():
    print("正在读取 data.in...")
    with open("data.in", "r") as f:
        all_data = [float(x) for line in f for x in line.split()]

    # 切分 Q, K, V
    Q = [all_data[i*d : (i+1)*d] for i in range(N)]
    K = [all_data[(N+i)*d : (N+i+1)*d] for i in range(N)]
    V = [all_data[(2*N+i)*d : (2*N+i+1)*d] for i in range(N)]

    print("计算标准答案 (Scaled Dot-Product)...")
    # 1. Scores = (Q * K^T) / sqrt(d)
    scale = math.sqrt(d)
    scores = matrix_multiply(Q, K, N, d, N)
    
    # 2. Softmax & 3. Weighted Sum
    final_output = [[0.0] * d for _ in range(N)]
    for i in range(N):
        # 缩放并做 Softmax
        row_scaled = [x / scale for x in scores[i]]
        probs = softmax(row_scaled)
        
        # 与 V 相乘得到最终行
        for j in range(d):
            final_output[i][j] = sum(probs[k] * V[k][j] for k in range(N))

    print("对拍数据")
    for i in range(2):
        print(final_output[i][:20])


if __name__ == "__main__":
    run_check()