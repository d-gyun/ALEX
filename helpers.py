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

def hist_plot(err_board, title, log_scale=False):
    occurrences = {}
    for i, count in enumerate(err_board):
        occurrences[i] = count
    checks = list(occurrences.keys())
    counts = list(occurrences.values())
    plt.bar(checks, counts, align='center', color='red', edgecolor='black')
    plt.xlabel('Number of checks')
    plt.ylabel('Occurrences')
    plt.title(title)
    if log_scale:
        plt.yscale('log')  # y축에 로그 스케일 적용
    plt.show()



def plot_cdf(alex_error_board, flex_error_board, title):
    # ALEX의 누적 빈도 계산 및 정규화
    alex_cumulative_freq = np.cumsum(alex_error_board)
    alex_cdf = alex_cumulative_freq / alex_cumulative_freq[-1]  # ALEX의 CDF 계산

    # FLEX의 누적 빈도 계산 및 정규화
    flex_cumulative_freq = np.cumsum(flex_error_board)
    flex_cdf = flex_cumulative_freq / flex_cumulative_freq[-1]  # FLEX의 CDF 계산

    # CDF 그래프 그리기
    plt.figure(figsize=(10, 6))

    # ALEX의 CDF 그래프
    plt.plot(alex_cdf, marker='o', linestyle='-', color='b', label='ALEX')

    # FLEX의 CDF 그래프
    plt.plot(flex_cdf, marker='x', linestyle='-', color='r', label='FLEX')

    # 그래프 제목 및 축 레이블 설정
    plt.title(f"CDF of {title}", fontsize=22)
    plt.xlabel("Number of Checks", fontsize=20)
    plt.ylabel("Cumulative Distribution", fontsize=20)
    plt.grid(True)

    # 범례 추가
    plt.legend(loc='lower right', fontsize=18)

    # 그래프 출력
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def plot_bar(Alex_split_cnt, Alex_retrain_cnt, Flex_split_cnt, Flex_retrain_cnt):
    labels = ['Split', 'Retrain', 'Split', 'Retrain']
    counts = [Alex_split_cnt, Alex_retrain_cnt, Flex_split_cnt, Flex_retrain_cnt]

    # 막대 너비 설정
    bar_width = 0.5

    # x축 위치 설정
    x_alex = np.array([0, 0.8])
    x_flex = np.array([2, 2.8])

    # 막대 그래프 그리기
    plt.figure(figsize=(8, 4))

    plt.bar(x_alex, counts[:2], color='blue', edgecolor='black', width=bar_width, label='ALEX')
    plt.bar(x_flex, counts[2:], color='red', edgecolor='black', width=bar_width, label='FLEX')

    # 레이블 및 제목 추가
    plt.xticks(np.concatenate([x_alex, x_flex]), labels, fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title('Comparison of Split and Retrain in ALEX and FLEX', fontsize=18)

    # 범례 추가
    plt.legend(fontsize=18)

    # 그래프 보여주기
    plt.tight_layout()
    plt.show()


def plot_memory_usage(alex_memory, flex_memory):
    plt.figure(figsize=(10, 6))
    # plt.yscale('log')

    # 메모리 사용량 그래프
    plt.plot(alex_memory, label='ALEX', marker='o', linestyle='-', color='b')
    plt.plot(flex_memory, label='FLEX', marker='x', linestyle='-', color='r')

    plt.title('Cumulative Memory Usage Over Time', fontsize=18)
    plt.xlabel('Time Steps (Search and Insertion)', fontsize=15)
    plt.ylabel('Memory Usage (KB)', fontsize=15)
    plt.legend(loc="upper left", fontsize=18)
    plt.grid(True)
    plt.show()