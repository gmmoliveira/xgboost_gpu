from xgboost.dask import DaskDMatrix, train as dask_xgboost_train, predict as dask_xgboost_predict
from dask.dataframe import from_array, from_pandas
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

import numpy as np
import pandas as pd


def train_gpu_XGBClassifier2(X, y,
							inplace=True, data_chunksize=None,
							n_gpus=None, n_threads_per_gpu=1,
							params=None,
							):
	'''
	:param X: a 2D matrix object of either type numpy ndarray or pandas DataFrame;
	:param y: a 1D array of one of the following types: numpy ndarray, pandas Series or pandas DataFrame;
	
	:param data_chunksize: number of rows to partition input data (both X and y simultaneously) to split among multiple
		GPU devices. Default value None splits evenly among devices;
	:param inplace: whether X and y should be modified in place (inplace is memory efficient and should be preferred for
		large datasets);
	:param n_gpus: number of GPUs to be used. Default value None selects all available devices.
	:param n_threads_per_gpu: number of threads per GPU;
	:param params: xgboost trainning params as a python dict, refer to
		https://xgboost.readthedocs.io/en/latest/parameter.html
	:return:
	'''
	local_gpus = LocalCUDACluster(n_workers=n_gpus, threads_per_worker=n_threads_per_gpu)
	local_dask_client = Client(local_gpus)
	
	if data_chunksize is None:
		data_chunksize = X.shape[0] // len(local_gpus.cuda_visible_devices)
	if not inplace:
		X = X.copy()
		y = y.copy()
	if params is None:
		params = {
					'learning_rate': 0.3,
					'max_depth': 8,
					'objective': 'binary:hinge',
					'verbosity': 3,
					'tree_method': 'gpu_hist'
				}
	
	if isinstance(X, pd.DataFrame):
		X = from_pandas(X, chunksize=data_chunksize)
	else:
		X = from_array(X, chunksize=data_chunksize)
	
	if isinstance(y, pd.DataFrame):
		y = from_pandas(y, chunksize=data_chunksize)
	else:
		y = from_array(y, chunksize=data_chunksize)
	
	dtrain = DaskDMatrix(local_dask_client, X, y)
	
	xgb_model = dask_xgboost_train(local_dask_client, params, dtrain, num_boost_round=100, evals=[(dtrain, 'train')])
	return xgb_model


def predict_gpu_xgbmodel(xgb_model, X, data_chunksize=None, n_gpus=None, n_threads_per_gpu=1):
	local_gpus = LocalCUDACluster(n_workers=n_gpus, threads_per_worker=n_threads_per_gpu)
	local_dask_client = Client(local_gpus)
	
	if data_chunksize is None:
		data_chunksize = X.shape[0] // len(local_gpus.cuda_visible_devices)
		
	if isinstance(X, pd.DataFrame):
		ndarray = False
		X = from_pandas(X, chunksize=data_chunksize)
	else:
		ndarray = True
		X = from_array(X, chunksize=data_chunksize)
	
	y_predicted = dask_xgboost_predict(local_dask_client, xgb_model, X)
	y_predicted = pd.DataFrame(y_predicted)
	if ndarray:
		return y_predicted.to_numpy()
	return y_predicted

def example():
	n = 10000
	X_train = np.random.random(size=(n, n))
	y_train = np.array([1 if np.sum(X_train[i, :]) > 0.5 * n else 0 for i in range(X_train.shape[0])])
	classification_xgbmodel = train_gpu_XGBClassifier2(X_train, y_train, n_gpus=1, n_threads_per_gpu=1)
	
	X_test = np.random.random(size=(n, n))
	y_test = np.array([1 if np.sum(X_train[i, :]) > 0.5 * n else 0 for i in range(X_train.shape[0])])
	y_pred = predict_gpu_xgbmodel(classification_xgbmodel, X_test, n_gpus=1, n_threads_per_gpu=1)
	y_pred = pd.DataFrame(y_pred)
	
	classification_xgbmodel['booster'].save_model('../models/001.model')
	
	


if __name__ == '__main__':
	from time import time
	t_start = time()
	example()
	t_end = time() - t_start
	print('executed in {:.2f} seconds'.format(t_end))