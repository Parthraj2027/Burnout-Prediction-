import numpy as np

def focal_loss_objective(y_true, y_pred, alpha=0.25, gamma=2.0):
    p = 1.0 / (1.0 + np.exp(-y_pred))

    grad = np.where(y_true == 1,
                    alpha * (p - 1) * (1 - p) ** (gamma - 1) * (gamma * p * np.log(p) + p - 1),
                    -(1 - alpha) * p * (1 - p) ** (gamma - 1) * (gamma * (1 - p) * np.log(1 - p) + p))

    hess = np.where(y_true == 1,
                    alpha * (1 - p) ** (gamma - 2) * (p * (gamma * (2 * p - 1) - p) + (1 - p) ** 2),
                    (1 - alpha) * p * (1 - p) ** (gamma - 2) * ((1 - p) * (gamma * (1 - 2 * p) + p) + p ** 2))

    return grad, hess

def weighted_logloss_objective(y_true, y_pred, pos_weight=2.0):
    p = 1.0 / (1.0 + np.exp(-y_pred))

    grad = np.where(y_true == 1,
                    pos_weight * (p - 1),
                    p)

    hess = np.where(y_true == 1,
                    pos_weight * p * (1 - p),
                    p * (1 - p))

    return grad, hess
