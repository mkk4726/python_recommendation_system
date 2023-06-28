import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def get_dataset_2() -> "pd.DataFrame":
  """users, movies, ratings dataframe을 반환하는 함수
  path = 'C:/Users/Hi/.surprise_data/ml-100k/ml-100k/'

  Returns:
      pd.DataFrame: users, movies, ratings dataframe
  """
  path = 'C:/Users/Hi/.surprise_data/ml-100k/ml-100k/'
  
  u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
  i_cols = ['movie_id', 'title', 'release_date', 'video release date', 'IMDB URL', 'unknown',
            'Action', 'Adventure', 'Animation', 'children\s', 'Comedy', 'Crime', 'Documentary', 'Drama',
            'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
            'Western']
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

  users = pd.read_csv(path + 'u.user', sep='|', names=u_cols, encoding='latin-1')
    
  movies = pd.read_csv(path + 'u.item', sep='|', names=i_cols, encoding='latin-1')
  # movies = movies[['movie_id', 'title']]
  
  ratings = pd.read_csv(path + 'u.data', sep='\t', names=r_cols, encoding='latin-1')
  ratings.drop('timestamp', axis=1, inplace=True)
  
  return users, movies, ratings

def RMSE(y_true, y_pred) -> float:
  """loss function
  
  Args:
      y_true (_type_): _description_
      y_pred (_type_): _description_

  Returns:
      float: RMSE
  """
  return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def RMSE2(y_true, y_pred) -> float:
  """calculating RMSE

  Args:
      y_true (_type_): _description_
      y_pred (_type_): _description_

  Returns:
      float: RMSE
  """
  y_true = np.array(y_true); y_pred = np.array(y_pred)
  return np.sqrt(np.mean((y_true - y_pred)**2))

# --------------------------------------- MF ----------------------------------------------# 
class MF:
  def __init__(self, ratings: "pd.DataFrame", 
               K:int, alpha:float, beta:float, 
               iterations:int, verbose=True):
    """MF 초기화함수

    Args:
        ratings (DataFrame | Series | array): 평가 데이터, pivot 한 rating matrix 형태 \n
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
      
# --------------------------------------- NEW_MF ------------------------------------- #
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
  
# ------------------------------------ CF --------------------------------------------------- #
class CF:
  def __init__(self, ratings:"pd.DataFrame", use_split:bool=True, ratings_train:"pd.DataFrame"=None, ratings_test:"pd.DataFrame"=None, 
               SIG_LEVEL:int=3, MIN_RATINGS:int=2):
    """init CF

    Args:
        ratings (pd.DataFrame): rating df \n
        use_split (bool) : train, test를 직접입력할건지 여부 (true: 직접입력 x)\n
        rating_train (pd.DataFrame): rating_train df \n 
        rating_test (pd.DataFrame): rating_test df \n
        SIG_LEVEL (int, optional): 최소 공통으로 평가한 유저의 수 . Defaults to 3. \n
        MIN_RATING (int, optional): 최소 유사도 개수. Defaults to 2. \n
    """
    self.SIG_LEVEL = SIG_LEVEL
    self.MIN_RATINGS = MIN_RATINGS
    
    if use_split:
      # train set에 대해서 측정
      X = ratings.copy()
      y = ratings['user_id']
      self.X_train, self.X_test, _, _ = train_test_split(X, y, test_size=0.25, stratify=y, random_state=25)
    else:
      self.X_train = ratings_train
      self.X_test = ratings_test
      
    self.rating_matrix = ratings.pivot(index="user_id", columns='movie_id', values='rating')
    for i in range(len(self.X_test)):
      user_id = ratings_test.iloc[i, 0]
      item_id = ratings_test.iloc[i, 1]
      self.rating_matrix.loc[user_id, item_id] = None
    
    matrix_dummy = self.rating_matrix.copy().fillna(0)

    user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
    self.user_similarity = pd.DataFrame(user_similarity, index=self.rating_matrix.index, columns=self.rating_matrix.index)
    
    user_corr = matrix_dummy.T.corr()
    self.user_corr = pd.DataFrame(user_corr, index=self.rating_matrix.index, columns=self.rating_matrix.index)

    self.rating_mean = self.rating_matrix.mean(axis=1) # 유저의 평가경향
    self.rating_no_bias = (self.rating_matrix.T - self.rating_mean).T # 평가경향 제거
    
    # 공통으로 평가한 영화의 수
    rating_binary = np.array((self.rating_matrix > 0).astype(float))
    counts = np.dot(rating_binary, rating_binary.T)
    self.counts = pd.DataFrame(counts, index=self.rating_matrix.index, columns=self.rating_matrix.index).fillna(0)
  
  def check_result(self, result:float) -> float:
    """rating의 값은 1~5사이의 값을 가지므로, 1이하면 1로, 5이상이면 5로 값을 바꿔주기

    Args:
        result (float): 모델의 결과값. 가중평균. 예측치.

    Returns:
        float: 수정된 결과값.
    """
    if (result < 1): return 1
    if (result > 5): return 5
    return result
  
  def CF_knn_bias_sig(self, user_id:str, movie_id:str, simil: str, neighbor_size: int) -> float:
    """
    주어진 영화에 대해서 평가한 사용자에 대해서, 평점을 기반으로 유사도를 계산하고, 
    유사도와 평점을 가중평균해 예측치를 구함.\n
    유사도 기준 상위 neighbor_size(=k)만큼을 이웃으로 정의. 이웃에 대해서만 가중평균을 진행.\n
    사용자의 평가경향을 고려 \n
    신뢰도(공통으로 평가한 아이템의 개수)가 특정값이상(전역변수로 정의)인 유저만 이웃으로 사용. \n
    
    Args:
        user_id (str): 사용자 id \n
        movie_id (str): 영화 id \n
        simil (str): similarity계산 방식 ( cosine or corr ) \n
        neighbor_size (int): 이웃의 수 \n
    Returns:
        float: user id와 movie id를 평가한 사용자에 대한, 유사도로 평점을 가중평균한 예측치
    """
    
    # 구할 수 없는 경우에는 평균평점으로 대체한다.
    
    default_rating = self.rating_mean[user_id]
    if np.isnan(default_rating): 
      default_rating = 3.0 # rating_mean[user_id]가 nan이라면 기본값= 3.0
    
    # 유사도 기준 설정. 
    if simil == 'cosine':
      similarity = self.user_similarity
    else:
      similarity = self.user_corr
    
    # 해당 movie id에 대해서 평가한 값이 있는지 확인 ( train set에 movie id가 있는지 확인 )
    if movie_id in self.rating_matrix.columns:
      movie_ratings = self.rating_no_bias[movie_id].copy() #  rating_bias를 이용해 평가경향 제거해줌.
      
      common_counts = self.counts[user_id] # 공통으로 평가한 영화의 수
      low_significance = common_counts < self.SIG_LEVEL # 공통으로 평가한 영화의 수 < SIG_LVEL
      no_rating = movie_ratings.isnull() # movie_id에 대해서 평가하지 않은 user 
      none_rating_idx = movie_ratings[no_rating | low_significance].index # 평가한 영화가 없거나, 신뢰도가 낮은 user id.
      movie_ratings = movie_ratings.dropna()
      
      sim_scores = similarity[user_id].copy()
      sim_scores = sim_scores.dropna()
      # 평가하지 않은 유저 + 신뢰도가 SIG_LEVEL보다 작은 유저 뺴주기
      sim_scores = sim_scores.drop(none_rating_idx, axis=0)
      
      # 유사도 개수가 MIN_RATINGS보다 작으면 평균값으로 예측
      if len(sim_scores) < self.MIN_RATINGS:
        return default_rating
      
      # 주어진 영화에 대해서 평가한 각 사용자에 대해서 평점을 유사도로 가중평균한 예측치를 구함
      # k가 0인경우 ( 안주어진 경우 )
      if neighbor_size == 0:
        mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
      # K가 주어진 경우
      else:
        sim_scores = sim_scores.sort_values(ascending=False)
  
        neighbor_size = min(neighbor_size, len(sim_scores))
        
        sim_scores = sim_scores[:neighbor_size]      
        movie_ratings = movie_ratings[sim_scores.index][:neighbor_size]
        
        # 사용자의 평가경향 더 해줌
        mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum() + default_rating
    # 없으면 3.0으로 예측
    else:
      mean_rating = default_rating
      
    return self.check_result(mean_rating)
