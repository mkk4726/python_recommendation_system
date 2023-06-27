import tensorflow as tf
import pandas as pd
import numpy as np

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
