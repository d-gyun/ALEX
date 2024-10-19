import numpy as np
import matplotlib.pyplot as plt
from ALEX import RMI as ALEX_RMI
from FLEX import RMI as FLEX_RMI
from DataGen import DataGen, Distribution
from helpers import hist_plot, plot_cdf, plot_bar, plot_memory_usage
import psutil

def main():
    data = DataGen(Distribution.LOGNORMAL, 100000, 1000000).generate()
    insert_data = DataGen(Distribution.LOGNORMAL, 30000, 1000000).generate()

    ALEX_idx = ALEX_RMI(data)
    FLEX_idx = FLEX_RMI(data)

    # 초기 메모리 사용량 측정
    alex_memory, flex_memory = [], []

    ALEX_error_board = ALEX_idx.find_all(data)
    ALEX_error_board = ALEX_error_board.astype(int)
    alex_memory.append(psutil.Process().memory_info().rss / 1024)  # KB 단위로 변환

    FLEX_error_board = FLEX_idx.find_all(data)
    FLEX_error_board = FLEX_error_board.astype(int)
    flex_memory.append(psutil.Process().memory_info().rss / 1024)  # KB 단위로 변환

    Before_ALEX_cost = sum(err*(i+1) for i, err in enumerate(ALEX_error_board))
    Before_ALEX_leafnode_cnt = ALEX_idx.leafNodeCnt()

    Before_FLEX_cost = sum(err*(i+1) for i, err in enumerate(FLEX_error_board))
    Before_FLEX_leafnode_cnt = FLEX_idx.leafNodeCnt()

    plot_cdf(ALEX_error_board, FLEX_error_board, 'LOGNORMAL Distribution')

    ALEX_idx.bulk_load(insert_data)
    alex_memory.append(psutil.Process().memory_info().rss / 1024)

    FLEX_idx.bulk_load(insert_data)
    flex_memory.append(psutil.Process().memory_info().rss / 1024)

    ALEX_split_cnt = ALEX_idx.split_cnt
    FLEX_split_cnt = FLEX_idx.split_cnt

    ALEX_retrain_cnt = ALEX_idx.retrain_cnt
    FLEX_retrain_cnt = FLEX_idx.retrain_cnt

    print("ALEX_split_cnt:", ALEX_split_cnt)
    print("FLEX_split_cnt:", FLEX_split_cnt)

    print("ALEX_retrain_cnt:", ALEX_retrain_cnt)
    print("FLEX_retrain_cnt:", FLEX_retrain_cnt)

    plot_bar(ALEX_split_cnt, ALEX_retrain_cnt, FLEX_split_cnt, FLEX_retrain_cnt)

    print(alex_memory)
    print(flex_memory)
    plot_memory_usage(alex_memory, flex_memory)

if __name__ == '__main__':
    main()
