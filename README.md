<h1>XGBoost_GPU</h1>
This script implements functions to facilitate the execution of the XGBoost algorithm on multiple GPUs on a single-
machine. Below, the detailed documentation of each function:


```python
def train_xgboost_gpu(
			X, y,
			data_chunksize=None,
			n_gpus=None, n_threads_per_gpu=1,
			params=None,
			xgboost_model=None,
			gpu_cluster=None, client=None
			):
'''
Trains a XGBoost model on the GPU.
	
:param X: a 2D matrix object of either type numpy ndarray or pandas DataFrame;
:param y: a 1D array of one of the following types: numpy ndarray, pandas Series or pandas DataFrame;

:param data_chunksize: number of rows to partition input data (both X and y simultaneously) to split among multiple
	GPU devices. Default value None splits evenly among devices;
:param n_gpus: number of GPUs to be used. Default value None selects all available devices.
:param n_threads_per_gpu: number of threads per GPU;
:param params: xgboost trainning params as a python dict, refer to
	https://xgboost.readthedocs.io/en/latest/parameter.html
:param xgboost_model: xgbooster object to continue training, it may be either a regular XGBoost model or a
	dask xgboost dict
:param gpu_cluster: an existing dask cluster object to use. This param should be used if you call this method
	too many times in quick successions. Note that this function doesn't close an externally created cluster.
:param client: an existing dask client object to use. This param should be used if you call this method
	too many times in quick successions. Note that this function doesn't close an externally created client.

:return:
A dictionary containing 2 keys:
	* 'booster': maps to a XGBoost model
	* 'history': maps to  another dict which informs the history of the training process, as in the following the
		examṕle: {'train': {'logloss': ['0.48253', '0.35953']}, 'eval': {'logloss': ['0.480385', '0.357756']}}}
'''
```


```python
def predict_xgboost_gpu(
			xgb_model, X,
			data_chunksize=None,
			n_gpus=None, n_threads_per_gpu=1,
			gpu_cluster=None, client=None
			):
'''
Predicts the output for the input features X using the 'xgb_model' running on the GPU. Please, note my tests have
	shown much greater performance when executing the predictions over the CPU rather than the GPU based on
	an input features array X of shape (10 ** 7, 10).

:param xgb_model: a dask XGBoost model to use for predictions
:param X: the input features to use for predictions, must be either a numpy ndarray or a pandas DataFrame
:param data_chunksize: chunk sizes to be used on a dask dataframe, leave the default value None for auto decision
:param n_gpus: number of GPUs to be used. Default value None selects all available devices;
:param n_threads_per_gpu: number of threads per GPU;
:param gpu_cluster: an existing dask cluster object to use. This param should be used if you call this method
	too many times in quick successions. Note that this function doesn't close an externally created cluster.
:param client: an existing dask cluster object to use. This param should be used if you call this method
	too many times in quick successions. Note that this function doesn't close an externally created client.
:return:
	If the input features X is a pandas DataFrame, returns a array-like DataFrame of single column containing
	the predictions;

	Otherwise, if the input features X is a numpy ndarray, returns a 1D ndarray containing the predictions.
'''
```
