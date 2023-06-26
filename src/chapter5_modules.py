import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def get_dataset_1() -> "pd.DataFrame":
  """users, movies, ratings dataframe을 반환하는 함수

  Returns:
      pd.DataFrame: users, movies, ratings dataframe
  """
  u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
  i_cols = ['movie_id', 'title', 'release_date', 'video release date', 'IMDB URL', 'unknown',
            'Action', 'Adventure', 'Animation', 'children\s', 'Comedy', 'Crime', 'Documentary', 'Drama',
            'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
            'Western']
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

  users = pd.read_csv('../data2/u.user', sep='|', names=u_cols, encoding='latin-1')
    
  movies = pd.read_csv('../data2/u.item', sep='|', names=i_cols, encoding='latin-1')
  # movies = movies[['movie_id', 'title']]
  
  ratings = pd.read_csv('../data2/u.data', sep='\t', names=r_cols, encoding='latin-1')
  ratings.drop('timestamp', axis=1, inplace=True)
  
  return users, movies, ratings

def load_ratings() -> "pd.DataFrame":
  """rating df를 불러오는 함수

  Returns:
      pd.DataFrame: user_id, movie_id, rating column을 가지는 df
  """
  path = 'C:/Users/Hi/Desktop/python_recommendation_system/data2/'
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv(path + 'u.data', names=r_cols, sep='\t', encoding='latin-1')
  ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)
  return ratings

def RMSE(y_true, y_pred) -> float:
  """RMSE를 계산하는 함수

  Args:
      y_true (_type_): 실제값
      y_pred (_type_): 예측값

  Returns:
      float: RMSE
  """
  y_true = np.array(y_true); y_pred = np.array(y_pred)
  
  return np.sqrt(np.mean((y_true - y_pred) ** 2))

def get_data_y() -> tuple[list, list, int]:
  """get data and y

  Returns:
      (list, list, int): data , y , num_x
  """
  ratings = load_ratings()

  # User encoding
  user_dict = {} # user_id - index mapping
  for i in set(ratings['user_id']): # 중복제거, 순서대로 출력
    user_dict[i] = len(user_dict)
  n_user = len(user_dict)

  # Item encoding
  item_dict = {}
  start_point = n_user
  for i in set(ratings['movie_id']):
    item_dict[i] = start_point + len(item_dict)
  n_item = len(item_dict)

  num_x = n_user + n_item
  
  ratings = shuffle(ratings, random_state=1)
  
  # Generate X data , sparse matrix -> coordinate format matrix 
  w0 = np.mean(ratings['rating']) # global bias
  y = (ratings['rating'] - w0).values.tolist()

  data = []; 

  for i in range(len(ratings)):
    case = ratings.iloc[i]
    
    x_index = []; x_value = []
    x_index.append(user_dict[case['user_id']]); x_value.append(1)
    x_index.append(item_dict[case['movie_id']]); x_value.append(1)
    
    data.append([x_index, x_value])
    
    if (i % 10000) == 0:
      print(f'Encoding {i} cases...')
    
  return data, y, num_x


class FM:
  def __init__(self, N:int, K:int, data:list, y, alpha:float, beta:float, 
               train_ratio:float=0.75, iterations:int=100, 
               tolerance:float=0.005, l2_reg:bool=True, verbose:bool=True):
    """FM inital function

    Args:
        N (int): # of x
        K (int): # of latent feature
        data (list): coo matrix
        y (list): rating data
        alpha (float): learnign rate
        beta (float): regularization rate
        train_ratio (float, optional): ratio of train set. Defaults to 0.75.
        iterations (int, optional): # of learning. Defaults to 100.
        tolerance (float, optional): 반복을 중단하는 RMSE의 기준. Defaults to 0.005.
        l2_reg (bool, optional): 정규화를 할지 여부. Defaults to True.
        verbose (bool, optional): 학습상황을 표시할지 여부. Defaults to True.
    """
    self.K = K; self.N = N; self.n_cases = len(data)
    self.alpha = alpha; self.beta = beta
    self.iterations = iterations
    self.tolerance = tolerance; self.l2_reg = l2_reg
    self.verbose = verbose
    # w 초기화
    self.w = np.random.normal(scale=1./self.N, size=(self.N))
    # v 초기화 (latent matrix)
    self.v = np.random.normal(scale=1./self.K, size=(self.N, self.K))
    # Train / Test 분리
    cutoff = int(train_ratio * self.n_cases)
    self.train_X = data[:cutoff]; self.test_X = data[cutoff:]
    self.train_y = y[:cutoff]; self.test_y = y[cutoff:]
  
  def predict(self, x_idx:list, x_value:list) -> float:
    """x_idx와 x_value값으로 y_hat을 예측하는 함수

    Args:
        x_idx (list): x의 idx list
        x_value (list): x의 value list

    Returns:
        float: y_hat
    """
    
    x_0 = np.array(x_value)
    x_1 = x_0.reshape(-1, 1) # 2차원으로 변경 (vx와의 연산을 위함)
    # cal bias score
    bias_score = np.sum(self.w[x_idx] * x_0)
    # cal latent score
    vx = self.v[x_idx] * x_1
    sum_vx = np.sum(vx, axis=0); sum_vx_2 = np.sum(vx * vx, axis=0)
    
    latent_score = 0.5 * np.sum(np.square(sum_vx) - sum_vx_2)
    # cal prediction
    y_hat = bias_score + latent_score
    
    return y_hat
  
  def sgd(self, X_data:list, y_data:list) -> float:
    """한번의 SGD를 진행하고 , RMSE값을 return 함.

    Args:
        X_data (list): input variables
        y_data (list): rating data

    Returns:
        float: RMSE of before w, v
    """
    y_pred = []
    
    for data, y in zip(X_data, y_data):
      x_idx = data[0] # index
      x_0 = np.array(data[1]) # value
      x_1 = x_0.reshape(-1, 1) # 2차원으로 변경 (vx와의 연산을 위함)
      vx = self.v[x_idx] * x_1
      
      y_hat = self.predict(x_idx, data[1])
      y_pred.append(y_hat)
      
      error = y - y_hat
      # update w, v
      if self.l2_reg: 
        self.w[x_idx] += error * self.alpha * (x_0 - self.beta * self.w[x_idx])
        self.v[x_idx] += error * self.alpha * (x_1 * sum(vx) - (vx * x_1) - self.beta * self.v[x_idx])
      else:
        self.w[x_idx] += error * self.alpha * x_0
        self.v[x_idx] += error * self.alpha * (x_1 * sum(vx) - (vx * x_1))
      
    return RMSE(y_data, y_pred)
    
  def test(self)-> list[float]:
    """train하면서 RMSE를 계산하는 함수

    Returns:
        list[float]: iter별로 RMSE를 저장한 list 
    """
    # SGD를 iterations 숫자만큼 수행
    best_RMSE = 10000; best_iteration = 0
    training_process = []
    
    for i in range(self.iterations):
      rmse1 = self.sgd(self.train_X, self.train_y)
      rmse2 = self.test_rmse(self.test_X, self.test_y)
      training_process.append((i, rmse1, rmse2))
      
      if self.verbose and (i+1) % 10 == 0:
        print(f"Iteration = {i+1} / Train RMSE = {rmse1:.6f} / Test RMSE = {rmse2:.6f}")
      if best_RMSE > rmse2:
        best_RMSE = rmse2; best_iteration = i
      # rmse2가 tolerance 이상으로 best rmse보다 크다면 중단 and 적어도 30개는 넘었을 때.
      elif (rmse2 - best_RMSE) > self.tolerance and i > 30: 
        break
    
    print(best_iteration, best_RMSE)
    return training_process  
    
  def test_rmse(self, x_data, y_data) -> float:
    """현재 w와 v로 계산한 예측치의 RMSE를 계산하는 함수

    Args:
        x_data (list): _description_
        y_data (list): _description_

    Returns:
        float: RMSE
    """
    y_pred = []
    
    for data, y in zip(x_data, y_data):
      y_hat = self.predict(data[0], data[1])
      y_pred.append(y_hat)
      
    return RMSE(y_data, y_pred)
  