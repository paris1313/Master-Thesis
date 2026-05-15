from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor


def get_models():

    return {
        'LR': LinearRegression(),
        'RF': RandomForestRegressor(n_estimators=100),
        'DT': DecisionTreeRegressor(max_depth=5),
        'GBR': HistGradientBoostingRegressor(),
        'SVR': SVR(C=1.0, epsilon=0.2),
        'EN': ElasticNet()
    }
