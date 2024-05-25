import numpy as np
import matplotlib.pyplot as plt
from ALEX import RMI as ALEX_RMI, LearnedIndexNode as ALEX_LearnedIndexNode
from EnhancedALEX import RMI as enhancedALEX_RMI, LearnedIndexNode as enhancedALEX_LearnedIndexNode
from LearnedIndex import RMI as LI_RMI, LearnedIndexNode as LI_LearnedIndexNode
from DataGen import DataGen, Distribution
from helpers import hist_plot


def main():
    data = DataGen(Distribution.LONGITUDES, 10000, 100000).generate()
    insert_data = DataGen(Distribution.LONGITUDES, 2000, 1000000).generate()
    # LI_idx = LI_RMI(data)
    ALEX_idx = ALEX_RMI(data)
    enhancedALEX_idx = enhancedALEX_RMI(data, density=ALEX_idx.density())

    # LI_error_board = LI_idx.find_all(data)
    # LI_error_board = LI_error_board.astype(int)

    ALEX_error_board = ALEX_idx.find_all(data)
    ALEX_error_board = ALEX_error_board.astype(int)

    enhancedALEX_error_board = enhancedALEX_idx.find_all(data)
    enhancedALEX_error_board = enhancedALEX_error_board.astype(int)

    # print(LI_error_board)
    # LearnedIndex_cost = sum(err*(i+1) for i, err in enumerate(LI_error_board))

    print(ALEX_error_board)
    Before_ALEX_cost = sum(err*(i+1) for i, err in enumerate(ALEX_error_board))
    Before_ALEX_leafnode_cnt = ALEX_idx.leafNodeCnt()

    print(enhancedALEX_error_board)
    Before_enhancedALEX_cost = sum(err*(i+1) for i, err in enumerate(enhancedALEX_error_board))
    Before_enhancedALEX_leafnode_cnt = enhancedALEX_idx.leafNodeCnt()

    # hist_plot(LI_error_board, 'LI Linear Regression in RANDOM Data N=10000')
    hist_plot(ALEX_error_board, 'ALEX Linear Regression in RANDOM Data')
    hist_plot(enhancedALEX_error_board, 'enhancedALEX Linear Regression in RANDOM Data')

    ################################# 데이터 bulk load #################################
    print("데이터 bulk load start")
    ALEX_idx.bulk_load(insert_data)
    enhancedALEX_idx.bulk_load(insert_data)

    ALEX_split_cnt = ALEX_idx.split_cnt
    enhancedALEX_split_cnt = enhancedALEX_idx.split_cnt

    ALEX_error_board = ALEX_idx.find_all(data)
    ALEX_error_board = ALEX_error_board.astype(int)
    enhancedALEX_error_board = enhancedALEX_idx.find_all(data)
    enhancedALEX_error_board = enhancedALEX_error_board.astype(int)

    After_ALEX_cost = sum(err * (i + 1) for i, err in enumerate(ALEX_error_board))
    After_ALEX_leafnode_cnt = ALEX_idx.leafNodeCnt()
    After_enhancedALEX_cost = sum(err * (i + 1) for i, err in enumerate(enhancedALEX_error_board))
    After_enhancedALEX_leafnode_cnt = enhancedALEX_idx.leafNodeCnt()

    #insert된 data에 대해서도 find 수행
    ALEX_error_board_inserted = ALEX_idx.find_all(insert_data).astype(int)
    enhancedALEX_error_board_inserted = enhancedALEX_idx.find_all(insert_data).astype(int)


    # print("LearnedIndex_cost:", LearnedIndex_cost)
    print("Before ALEX_cost:", Before_ALEX_cost)
    print("Before enhancedALEX_cost:", Before_enhancedALEX_cost)
    print("Before ALEX_leafnode_cnt:", Before_ALEX_leafnode_cnt)
    print("Before enhancedALEX_leafnode_cnt:", Before_enhancedALEX_leafnode_cnt)

    print("After ALEX_cost:", After_ALEX_cost)
    print("After enhancedALEX_cost:", After_enhancedALEX_cost)
    print("After ALEX_leafnode_cnt:", After_ALEX_leafnode_cnt)
    print("After enhancedALEX_leafnode_cnt:", After_enhancedALEX_leafnode_cnt)

    print("ALEX_split_cnt:", ALEX_split_cnt)
    print("enhancedALEX_split_cnt:", enhancedALEX_split_cnt)

    hist_plot(ALEX_error_board, 'ALEX Linear Regression in RANDOM Data')
    hist_plot(enhancedALEX_error_board, 'enhancedALEX Linear Regression in RANDOM Data')

    hist_plot(ALEX_error_board_inserted, 'ALEX Linear Regression in inserted Data')
    hist_plot(enhancedALEX_error_board_inserted, 'enhancedALEX Linear Regression in inserted Data')

if __name__ == '__main__':
    main()







