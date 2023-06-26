import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

  


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



class MF:
  def __init__(self, ratings: "pd.DataFrame", 
               K:int, alpha:float, beta:float, 
               iterations:int, verbose=True):
    """MF 초기화함수

    Args:
        ratings (DataFrame | Series | array): 평가 데이터 \n
        K (int): # of latent factor \n
        alpha (float): learning rate \n
        beta (float): regularization rate \n
        iterations (int): # of learning \n
        verbose (bool, optional): 학습결과 출력여부. Defaults to True. \n
    """
    self.R = np.array(ratings)
    self.num_users, self.num_items = np.shape(self.R)
    self.K = K; self.alpha = alpha;self.beta = beta
    self.iterations = iterations
    self.verbose = verbose
    
  def rmse(self) -> float:
    """caculate rmse of function

    Returns:
        float: result of rmse
    """
    xs, ys = self.R.nonzero() # 0이 아닌 index를 return해주는 method
    self.predictions = []
    self.errors = []
    
    for x, y in zip(xs, ys):
      prediction = self.get_prediction(x, y)
      self.predictions.append(prediction)
      self.errors.append(self.R[x, y] - prediction)
      
    self.predictions = np.array(self.predictions)
    self.errors = np.square(self.errors) 
    
    return np.sqrt(np.mean(self.errors))
  
  def get_prediction(self, i:int, j:int) -> float:
    """latent matrix(P, Q)를 이용해 R의 i,j 항을 예측하는 함수

    Args:
        i (int): row 
        j (int): column

    Returns:
        float: prediction of r
    """
    prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
    return prediction
  
  def train(self) -> list:
    """train model

    Returns:
        list: results of trains
    """
    # Initializing user-feature and movie-feature latent matrix
    self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
    self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
    
    # Initializing the bias terms
    self.b_u = np.zeros(self.num_users)
    self.b_d = np.zeros(self.num_items)
    self.b = np.mean(self.R[self.R.nonzero()])
    
    # List of training samples
    rows, columns = self.R.nonzero()
    self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]
    
    # Stochastic gradient descent for given number of iterations
    training_process = []
    for i in range(self.iterations):
      # self.sample = shuffle(self.samples) # sample을 한번 더 섞어준다. 큰 차이는 없다.
      self.sgd()
      rmse = self.rmse()
      training_process.append((i+1, rmse))
      if self.verbose:
        if (i + 1) % 10 == 0:
          print(f"Iteration {i+1:d} / Train RMSE = {rmse:.4f}")
    
    return training_process
  
  def sgd(self):
    """ P, Q, b_u, b_d를 업데이트하는 함수.
    """
    for i, j, r in self.samples:
      prediction = self.get_prediction(i, j)
      e = (r - prediction)
      
      self.P[i, :] += self.alpha * (2 * e * self.Q[j, :] - self.beta * self.P[i, :])
      self.Q[j, :] += self.alpha * (2 * e * self.P[i, :] - self.beta * self.Q[j, :])
      
      self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
      self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])  
      
class NEW_MF(MF):
  def __init__(self, ratings: "pd.DataFrame", 
               K:int, alpha:float, beta:float, 
               iterations:int, verbose=True):
    super().__init__(ratings, K, alpha, beta, iterations, verbose)
    # 사용자 아이디, 아이템 아이디가 내부의 인덱스와 일치하지 않을 경우를 대비해 
    # 실제 아이디와 내부 인덱스를 매핑해줘야함
    item_id_to_index = []; index_to_item_id = []
    for i, item_id in enumerate(ratings.columns):
      item_id_to_index.append([item_id, i])
      index_to_item_id.append([i, item_id])
    self.item_id_to_index = dict(item_id_to_index); self.index_to_item_id = dict(index_to_item_id)
    
    user_id_to_index = []; index_to_user_id = []
    for i, user_id in enumerate(ratings.index):
      user_id_to_index.append([user_id, i])
      index_to_user_id.append([i, user_id])
    self.user_id_to_index = dict(user_id_to_index); self.index_to_user_id = dict(index_to_user_id)
      
  def set_test(self, ratings_test:"pd.DataFrame") -> "pd.DataFrame":
    """test set 설정하기. test에 쓰이는 것들은 R에서 0으로 바꿔주기

    Args:
        ratings_test (pd.DataFrame): test용 rating df

    Returns:
        pd.DataFrame: test set
    """
    test_set = []
    for i in range(len(ratings_test)):
      x = self.user_id_to_index[ratings_test.iloc[i, 0]]
      y = self.item_id_to_index[ratings_test.iloc[i, 1]]
      z = ratings_test.iloc[i, 2]
      test_set.append([x, y, z])
      self.R[x, y] = 0 # test set에서 사용되는 것들은 학습에 사용되지 않도록 설정하기
    self.test_set = test_set
    
    return test_set
  
  def test_rmse(self) -> float:
    """test set에 대하여 rmse값을 구하는 함수

    Returns:
        float: rmse for test set
    """
    error = 0
    for test_set in self.test_set:
      predicted = self.get_prediction(test_set[0], test_set[1])
      error += pow(test_set[2] - predicted, 2)
    return np.sqrt(error / len(self.test_set))
  
  def test(self) -> list:
    """학습과 동시에 test set에 대한 rmse도 구함

    Returns:
        list: (iter, train rmse, test rmse)의 값을 가진 list
    """
    # Initializing user-feature and movie-feature latent matrix
    self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
    self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
    
    # Initializing the bias terms
    self.b_u = np.zeros(self.num_users)
    self.b_d = np.zeros(self.num_items)
    self.b = np.mean(self.R[self.R.nonzero()])
    
    # List of training samples
    rows, columns = self.R.nonzero()
    self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]
    
    # Stochastic gradient descent for given number of iterations
    training_process = []
    for i in range(self.iterations):
      # self.sample = shuffle(self.samples) # sample을 한번 더 섞어준다. 큰 차이는 없다.
      self.sgd()
      rmse = self.rmse()
      rmse2 = self.test_rmse()
      training_process.append((i+1, rmse, rmse2))
      if self.verbose:
        if (i + 1) % 10 == 0:
          print(f"Iteration {i+1:d} / Train RMSE = {rmse:.4f} / Test RMSE = {rmse2:.4f}")
    
    return training_process
  
  def get_one_prediction(self, user_id: int, item_id:int) -> float:
    """한 개의 값에 대한 예측치를 구하는 함수

    Args:
        user_id (int): 유저 id
        item_id (int): 아이템 id

    Returns:
        float: rating 예측치
    """
    return self.get_prediction(self.user_id_to_index[user_id], self.item_id_to_index[item_id])
  
  def full_prediction(self) -> "pd.DataFrame":
    """전체 평점에 대한 예측치를 구하는 함수

    Returns:
        pd.DataFrame: 전체 평점에 대한 예측치
    """
    # np.newaxis를 이용해 2차원으로 변경. np.newaxis가 있는쪽이 1. ex: [:, np.newaxis] -> (5, 1)
    return self.b + self.b_u[:, np.newaxis] + self.b_d[np.newaxis, :] + self.P.dot(self.Q.T)
      
      
