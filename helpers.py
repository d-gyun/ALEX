import math

import matplotlib.pyplot as plt
import numpy as np


def minmax(min_val, max_val, val):
    if max(min_val, val) == val and min(val, max_val) == val:
        return val
    elif max(min_val, val) != val and min(val, max_val) == val:
        return min_val
    else:
        return max_val

# data_list를 이용하여 gapped_array 생성
# 코드 구현 시에 GappedArray를 구성할 필요가 없이 density를 0.6에서 0.8 사이로 맞춰서 array를 유지하도록 구현
'''
def create_gapped_array(data_list, gap_ratio=2/3):
    # 데이터 리스트의 길이
    data_length = len(data_list)

    # 간격의 길이
    gap_length = math.floor(data_length * gap_ratio)

    # 데이터의 간격을 정렬된 상태에서 랜덤하게 선택
    gap_indices = np.sort(np.random.choice(np.arange(1, data_length + gap_length), gap_length, replace=False))

    # GappedArray 초기화
    gapped_array = [np.nan] * (data_length + gap_length)

    # 데이터 복사 및 간격 추가
    data_index = 0
    gap_index = 0
    for i in range(data_length + gap_length):
        if gap_index < gap_length and i == gap_indices[gap_index]:
            gap_index += 1
        else:
            gapped_array[i] = int(data_list[data_index])
            data_index += 1
    return np.array(gapped_array)
'''

def hist_plot(err_board, title):
    occurrences = {}
    for i, count in enumerate(err_board):
        occurrences[i] = count
    checks = list(occurrences.keys())
    counts = list(occurrences.values())
    plt.bar(checks, counts, align='center', color='red', edgecolor='black')
    plt.xlabel('Number of errors')
    plt.ylabel('Occurrences')
    plt.title(title)
    plt.show()
