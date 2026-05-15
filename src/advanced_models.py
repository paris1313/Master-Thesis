from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


kernel = DotProduct() + WhiteKernel()


def get_advanced_models():

    return {
        'Bayesian': BayesianRidge(),
        'SGD': SGDRegressor(),
        'KNN': KNeighborsRegressor(n_neighbors=7),
        'GPR': GaussianProcessRegressor(kernel=kernel)
    }
