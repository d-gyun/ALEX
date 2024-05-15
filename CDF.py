import numpy as np
import matplotlib.pyplot as plt

def main():
    #rand 데이터 생성
    # data = []
    # for i in range(1000):
    #     sample = np.random.randint(10000)
    #     data.append(sample)
    # data = np.array(data)

    # 임의의 데이터 생성 (로그 정규분포)
    data = np.random.lognormal(mean=0, sigma=1, size=1000)

    data = np.sort(data)
    # CDF 계산
    cdf = np.arange(1, len(data) + 1) / len(data)

    # CDF 시각화
    plt.plot(data, cdf, label='CDF')
    plt.title('CDF of Data')
    plt.xlabel('Data')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

    # 1차 도함수 (기울기) 계산
    cdf_gradients = np.gradient(cdf, data)

    # 기울기 변화율 계산 (1차 도함수의 절대값)
    gradient_changes = np.abs(cdf_gradients)

    # 기울기 변화율 시각화
    plt.plot(data, gradient_changes, label='Gradient Changes')
    plt.title('Gradient Changes of CDF')
    plt.xlabel('Data')
    plt.ylabel('Gradient Changes')
    plt.legend()
    plt.show()

    # 기울기 변화율이 특정 임계값 이상인 지점 찾기
    threshold = 50  # 변화율 임계값 설정
    change_points = np.where(gradient_changes > threshold)[0]

    # 인접한 change_points 병합
    merged_change_points = []
    prev_cp = change_points[0]
    merged_change_points.append(prev_cp)
    for cp in change_points[1:]:
        if cp - prev_cp > 1:
            merged_change_points.append(cp)
        prev_cp = cp

    print("Change Points:", merged_change_points)

    # 분할 지점 포함하기
    split_points = [0] + list(merged_change_points) + [len(data)]
    nodes = [data[split_points[i]:split_points[i + 1]] for i in range(len(split_points) - 1)]

    # 각 노드에 할당된 데이터 확인
    for i, node in enumerate(nodes):
        print(f"Node {i+1}: {len(node)} elements")

    # 변화 지점 시각화
    plt.plot(data, cdf, label='CDF')
    for cp in merged_change_points:
        plt.axvline(x=data[cp], color='r', linestyle='--', label='Change point')
    plt.title('CDF of Data with Change Points')
    plt.xlabel('Data')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
