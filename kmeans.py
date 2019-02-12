def getdist(X1, X2):
    # 유클리디안 거리
    from math import sqrt
    if len(X1) != len(X2):
        exit(1)

    return sqrt(sum((X1[i] - X2[i]) ** 2 for i in range(len(X1))))
    

def KMeans(X, k, medoid=False):
    import random
    import numpy as np
    from pandas import DataFrame
    
    dt = np.c_[np.array(X), np.zeros(shape=len(X))]

    # 거리기반이므로 정규화
    for i in dt:
        i = (i - np.mean(i)) / np.std(i)
    # 처음엔 랜덤한 데이터 k 개 잡기
    centers = np.array([dt[i][:-1] for i in random.sample(range(len(X)), k)])

    while 1:
        sums = np.zeros_like(centers)
        cnts = [0] * k

        for i in dt:
            mindist, group = 100000000, 0
            
            # 여기서 가장 가까운 중심점 찾기
            for j in range(k):
                temp = getdist(i[:-1], centers[j])
                if temp < mindist:
                    mindist, group = temp, j

            # 각 그룹당 합과 개수 모두 있어야 하므로 여기서 더해주기
            sums[group] += i[:-1]
            cnts[group] += 1
            i[-1] = group
            
        # 일반 k means 알고리즘의 경우 이 avg 를 새로운 중심점으로 이용
        new = np.array([sums[i] / cnts[i] if cnts[i] > 0 else np.zeros_like(sums[i]) for i in range(k)])
        # medoid=True 면 k medoid 알고리즘을 이용하기 위해 avg 에서 가장 가까운 점들을 찾기
        if medoid:
            medoids = np.zeros_like(new)
            dists = np.array([100000000] * len(new))
            for i in dt:
                group = int(i[-1])
                tmp = getdist(new[group], i[:-1])
                if dists[group] > tmp:
                    medoids[group] = np.array(i[:-1])
                    dists[group] = tmp

            new = medoids

        # 루프 하나가 끝났으면 출력해주기
        print("one loop done")
        # 중심점에 변화가 없다면 끝
        if np.array_equal(new, centers):
            break
        # 아니면 중심점 바꿔주기
        centers = new

    # 각 군집에 몇개의 데이터들이 들어갔는지 출력해주고
    print([i for i in cnts])

    # 만들어진 변수들과 군집의 최종 중심점을 출력
    # 중심점은 표준화되어있음을 유의해야함
    return dt[:, -1], centers

from sklearn import datasets
from pandas import DataFrame
import numpy as np

iris = datasets.load_iris()
data = DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
x = data.drop(columns=['target'], axis=1)

data['group'] = KMeans(data.drop(columns=['target'], axis=1), 3, medoid=True)[0]
print('k medoid')
for i in range(3):
    tmp = data[data['target'] == i]
    print(i, [len(tmp[tmp['group'] == j]) for j in range(3)])

del data['group']
data['group'] = KMeans(data.drop(columns=['target'], axis=1), 3)[0]
print('k means')
for i in range(3):
    tmp = data[data['target'] == i]
    print(i, [len(tmp[tmp['group'] == j]) for j in range(3)])

# 몇번 실행해본 결과 실행할 때마다 편차가 꽤나 있고 medoid 를 쓰건 안쓰건 성능차이가 눈에 띄진 않는다.
# 그러나 medoid 가 일반적으로 더 빨리 수렴하는 것으로 보인다.
# 일반적으로 target = 0 인 데이터는 잘 걸러지는 반면 1, 2 인 데이터들은 섞인다.