import numpy as np
import matplotlib.pyplot as plt
from ALEX import RMI as ALEX_RMI
from FLEX import RMI as FLEX_RMI
from helpers import hist_plot, plot_cdf, plot_bar, plot_splits_retrains_over_time, plot_memory_usage
import tracemalloc

# 데이터셋 로드 함수
def load_dataset(file_path):
    return np.loadtxt(file_path, delimiter=",")

def main():
    # 미리 생성된 데이터셋 로드
    data_file = "datasets/dataset_LOGNORMAL_10000.csv"
    insert_file = "datasets/dataset_LOGNORMAL_1000.csv"
    data = load_dataset(data_file)
    insert_data = load_dataset(insert_file)

    ALEX_idx = ALEX_RMI(data)
    FLEX_idx = FLEX_RMI(data)

    # 메모리 사용량 추적 시작
    tracemalloc.start()

    # 초기 메모리 사용량 측정
    alex_memory, flex_memory = [], []
    time_steps = []  # 시간 스텝 추적
    alex_split_history, alex_retrain_history = [], []
    flex_split_history, flex_retrain_history = [], []

    ALEX_error_board = ALEX_idx.find_all(data)
    ALEX_error_board = ALEX_error_board.astype(int)
    current, peak = tracemalloc.get_traced_memory()
    alex_memory.append(current / 1024)  # KB 단위로 변환

    FLEX_error_board = FLEX_idx.find_all(data)
    FLEX_error_board = FLEX_error_board.astype(int)
    current, peak = tracemalloc.get_traced_memory()
    flex_memory.append(current / 1024)  # KB 단위로 변환

    Before_ALEX_cost = sum(err*(i+1) for i, err in enumerate(ALEX_error_board))
    Before_ALEX_leafnode_cnt = ALEX_idx.leafNodeCnt()

    Before_FLEX_cost = sum(err*(i+1) for i, err in enumerate(FLEX_error_board))
    Before_FLEX_leafnode_cnt = FLEX_idx.leafNodeCnt()

    plot_cdf(ALEX_error_board, FLEX_error_board, 'LOGNORMAL Distribution')

    for step, key in enumerate(insert_data):
        time_steps.append(step)

        # Insert into ALEX and FLEX
        ALEX_idx.insert(key)
        FLEX_idx.insert(key)

        # Record split and retrain counts
        alex_split_history.append(ALEX_idx.split_cnt)
        alex_retrain_history.append(ALEX_idx.retrain_cnt)
        flex_split_history.append(FLEX_idx.split_cnt)
        flex_retrain_history.append(FLEX_idx.retrain_cnt)

        # 메모리 사용량 측정
        current, peak = tracemalloc.get_traced_memory()
        alex_memory.append(current / 1024)
        flex_memory.append(current / 1024)

    tracemalloc.stop()

    print("ALEX_split_cnt:", ALEX_idx.split_cnt)
    print("FLEX_split_cnt:", FLEX_idx.split_cnt)

    print("ALEX_retrain_cnt:", ALEX_idx.retrain_cnt)
    print("FLEX_retrain_cnt:", FLEX_idx.retrain_cnt)

    plot_splits_retrains_over_time(alex_split_history, alex_retrain_history, flex_split_history, flex_retrain_history, time_steps)

    print(alex_memory)
    print(flex_memory)
    plot_memory_usage(alex_memory, flex_memory)

if __name__ == '__main__':
    main()
