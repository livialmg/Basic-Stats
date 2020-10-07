import numpy as np
from scipy.stats import pearsonr
import pandas as pd

def mbe(Obs, Cal):
    """
    Mean Bias Error
    INPUT:
    Cal: (N,) array_like of calculated values
    Obs: (N,) array_like of observed values
    :return: Scalar value MBE between cal and obs
    .. math::
        MBE = \frac{1}{n}\sum (C_{i} - O_{i})
    """ 
    differences = np.array(Cal) - np.array(Obs)
    mbe_val = np.nanmean(differences)
    mbe_val = round(mbe_val, 2)
    
    return mbe_val

def mae(Obs, Cal):
    """
    Mean Absolute Error
    INPUT:
    Cal: (N,) array_like of calculated values
    Obs: (N,) array_like of observed values
    :return: Scalar value MAE between cal and obs
    .. math::
        MAE = \frac{1}{n}\sum \vert C_{i} - O_{i} \vert
    """
    differences = abs(np.array(Cal) - np.array(Obs))
    mae_val = np.nanmean(differences)
    mae_val = round(mae_val, 2)
    
    return mae_val

def rmse(Obs, Cal):
    """
    Root Mean Square Error
    INPUT:
    Cal: (N,) array_like of calculated values
    Obs: (N,) array_like of observed values
    :return: Scalar value RMSE between cal and obs
    .. math::
        RMSE = \sqrt{\frac{1}{n}\sum (C_{i}-O_{i})^2}
    """
    differences = np.array(Cal) - np.array(Obs)
    differences_squared = differences ** 2
    mean_of_differences_squared = np.nanmean(differences_squared)
    rmse_val = np.sqrt(mean_of_differences_squared)
    rmse_val = round(rmse_val, 2)
    
    return rmse_val

def d(Obs, Cal, order = 2):
    """
    Index of Agreement (Willmott et al., 1984) range from 0.0 to 1.0 
    and the closer to 1 the better the performance of the model.
    INPUT:
    Cal: (N,) array_like of calculated values
    Obs: (N,) array_like of observed values
    order: exponent to be used in the computation. Default is 2.
    OUTPUT: Index of Agreement between 'Cal' and 'Obs'
    .. math::
        d = 1 - \frac{\sum (C_{i} - O_{i} )^2}
                     {\sum (\vert C_{i} - \bar{O} \vert + \vert O_{i} - \bar{O} \vert)^2}
    """
    obs_mean = np.nanmean(Obs)
    denominator = np.nansum(np.power(np.abs(Cal - obs_mean) + np.abs(Obs - obs_mean), order))
    if denominator == 0.0:
        return np.nan

    nominator = np.nansum(np.power(Cal - Obs, order))
    return round(1 - (nominator / denominator), 2)

def r2(Obs, Cal):
    """
    'R2' is the Coefficient of Determination
    The coefficient of determination is such that 0 <  R2 < 1,  and denotes the strength 
    of the linear association between cal and obs.
    INPUT:
    Cal: (N,) array_like of calculated values
    Obs: (N,) array_like of observed values
    :return: Scalar value Coefficient of Determination between cal and obs
    .. math::
        r^2 = \frac{[\sum (C_{i} - \bar{C})(O_{i} - \bar{O})]^2}
                   {\sum ( C_{i} - \bar{C})^2 + \sum (O_{i} - \bar{O} )^2}
    """
    
    dic = {'obs': Obs, 'cal': Cal}
    temp = pd.DataFrame.from_dict(dic)
    temp = temp.dropna()
    Obs_ = temp['obs'].to_numpy()
    Cal_ = temp['cal'].to_numpy()
    result = round(pearsonr(Obs_, Cal_)[0] ** 2.0, 2)
    return result


