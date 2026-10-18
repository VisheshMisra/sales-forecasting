import numpy as np


def rmspe(y, yhat):
    # Official Kaggle metric; rows with zero actual sales are ignored.
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = y != 0
    return float(np.sqrt(np.mean(((y[mask] - yhat[mask]) / y[mask]) ** 2)))


def mape(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def report(name, y, yhat):
    return {
        "model": name,
        "RMSPE": round(rmspe(y, yhat), 4),
        "MAPE_%": round(mape(y, yhat), 2),
        "MAE": round(mae(y, yhat), 1),
    }
