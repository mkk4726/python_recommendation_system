import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import time, os
from chapter4_modules import *
import pickle

TRAIN_SIZE = 0.75
ratings = load_ratings()

split_kwargs = {
  'test_size': 1-TRAIN_SIZE,
  'stratify': ratings['rating']
}

rating_train, ratings_test, _, _ = train_test_split(ratings, ratings['rating'], **split_kwargs)
R_temp = load_ratings().pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

CPU = mp.cpu_count()

def get_result(K):
    print(f"mf.test : K={K} , PID : {os.getpid()} start")
    
    verbose = False
    # if (K % 50 == 0): verbose = True # multiprocess 모두 출력하지 않고, 한 개만 결과를 출력받자.
    
    mf_kwargs = {
      'ratings': R_temp,
      'K': K,
      'alpha': 0.001,
      'beta': 0.02,
      'iterations':300,
      'verbose':verbose
    }
    
    mf = NEW_MF(**mf_kwargs)
    _ = mf.set_test(ratings_test)
    result = mf.test()
    print(f"mf.test : K={K} , PID : {os.getpid()} end")
    return result

# 결과 저장하기.
def save_result_to_pickle(path, result):
  with open(path, 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
# 시간이 오래걸리니 병렬처리하기 -> ipynb에서는 실행안됨.
# print(mp.cpu_count(), mp.current_process().name) # CPU 개수 확인
if __name__ == '__main__':
  start = int(time.time())
  
  with mp.Pool(CPU) as p:
    result= p.map(get_result, range(50, 261, 10)) 
  
  save_result_to_pickle('results/find_K_300.pickle', result)
  
  print("***run time(sec) :", int(time.time()) - start)