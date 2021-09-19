import numpy as np
from numpy import ndarray
from numpy.core.numeric import zeros_like
import scipy.linalg as la
import solution
from utils.gaussparams import MultiVarGaussian
from config import DEBUG
from typing import Sequence


def get_NIS(z_pred_gauss: MultiVarGaussian, z: ndarray):
    """Calculate the normalized innovation squared (NIS), this can be seen as 
    the normalized measurement prediction error squared. 
    See (4.66 in the book). 
    Tip: use the mahalanobis_distance method of z_pred_gauss, (3.2) in the book

    Args:
        z_pred_gauss (MultiVarGaussian): predigted measurement gaussian
        z (ndarray): measurement

    Returns:
        NIS (float): normalized innovation squared
    """

    z_bar = z_pred_gauss.mean
    S = z_pred_gauss.cov
    nu = z-z_bar
    S_inv = np.linalg.inv(S)
    NIS = nu.T@S_inv@nu

    return NIS


def get_NEES(x_gauss: MultiVarGaussian, x_gt: ndarray):
    """Calculate the normalized estimation error squared (NEES)
    See (4.65 in the book). 
    Tip: use the mahalanobis_distance method of x_gauss, (3.2) in the book

    Args:
        x_gauss (MultiVarGaussian): state estimate gaussian
        x_gt (ndarray): true state

    Returns:
        NEES (float): normalized estimation error squared
    """

    x_bar, P = x_gauss
    x_error = x_bar-x_gt
    P_inv = np.linalg.inv(P)
    NEES = x_error.T@P_inv@x_error

    return NEES


def get_ANIS(z_pred_gauss_data: Sequence[MultiVarGaussian],
             z_data: Sequence[ndarray]):
    """Calculate the average normalized innovation squared (ANIS)
    Tip: use get_NIS

    Args:
        z_pred_gauss_data (Sequence[MultiVarGaussian]): Sequence (List) of 
            predicted measurement gaussians
        z_data (Sequence[ndarray]): Sequence (List) of true measurements

    Returns:
        ANIS (float): average normalized innovation squared
    """
    NIS_list = np.zeros_like(z_data)
    for i in range(len(z_data)):
        NIS_list[i]=get_NIS(z_pred_gauss_data[i], z_data[i])
    ANIS = np.mean(NIS_list)

    return ANIS


def get_ANEES(x_upd_gauss_data: Sequence[MultiVarGaussian],
              x_gt_data: Sequence[ndarray]):
    """Calculate the average normalized estimation error squared (ANEES)
    Tip: use get_NEES

    Args:
        x_upd_gauss_data (Sequence[MultiVarGaussian]): Sequence (List) of 
            state estimate gaussians
        x_gt_data (Sequence[ndarray]): Sequence (List) of true states

    Returns:
        ANEES (float): average normalized estimation error squared
    """

    NEES_list = np.zeros_like(x_gt_data)
    for i in range(len(x_gt_data)):
        NEES_list[i]=get_NEES(x_upd_gauss_data[i], x_gt_data[i])
    ANEES = np.mean(NEES_list)

    return ANEES
