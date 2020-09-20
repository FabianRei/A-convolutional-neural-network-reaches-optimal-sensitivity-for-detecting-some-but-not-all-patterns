import numpy as np
import pandas as pd
from psychopy.data import FitWeibull


def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=';')
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    if exclude_from is not None:
        sort_val = sort_val[sort_idxs]
        col = col[sort_val >= exclude_from]
    return col


class FixedBetaFitWeibull(FitWeibull):
    """
    Fits the weibull cmf given a fixed beta.

    should probably have the _baseFunctionFit class as superclass, but FitWeibull is easier to import and
    essentially the same..
    """
    glob_beta = None

    def __init__(self, fixed_beta, *args, **kwargs):
        self.fixed_beta = fixed_beta
        FixedBetaFitWeibull.glob_beta = self.fixed_beta
        super().__init__(*args, **kwargs)

    def _eval(self, xx, alpha):
        _chance = 0
        xx = np.asarray(xx)
        yy = _chance + (1.0 - _chance) * (1 - np.exp(-(xx / alpha) ** self.fixed_beta))
        return yy

    def _inverse(self, yy, alpha):
        _chance = 0
        xx = alpha * (-np.log((1.0 - yy)/(1 - _chance))) ** (1.0/self.fixed_beta)
        return xx



def get_alphas_fixed_beta(x, *ys):
    """
    We can expect that y1.max() == y2.max()
    svm dprime max is sometimes smaller
    :param x:
    :param y1:
    :param y2:
    :return:
    """
    fits = []
    for i in range(len(ys)):
        ys[i][ys[i] == ys[i].max()] = np.max(ys)
        fits.append(FitWeibull(x, ys[i]/ys[i].max(), expectedMin=0))
        # print(f"fit{i} is {fits[i].params}.")

    fixed_beta = np.mean([fit.params[1] for fit in fits])
    # print(f"fixed_beta = {fixed_beta}")
    new_fits = []
    for i, y in enumerate(ys):
        new_fit = FixedBetaFitWeibull(fixed_beta, x, y/y.max(), expectedMin=0)
        new_fits.append(new_fit)
        # print(f"new_fit{i} is {new_fits[i].params}")
    alphas = [new_fit.params[0] for new_fit in new_fits]
    return alphas, new_fits


class ScaledWeibull():
    def __init__(self, xx, yy, y_max=-1, **kwargs):
        self.xx = np.copy(xx)
        self.yy = np.copy(yy)
        if y_max != -1:
            self.y_max = y_max
        else:
            self.y_max = yy.max()
        self.yy /= self.y_max
        self.WeibullFit = FitWeibull(self.xx, self.yy, expectedMin=0, **kwargs)

    def interpol(self, target_dprime):
        result = self.WeibullFit.inverse(target_dprime/self.y_max)
        return result

    def evaluate(self, xx=None):
        if xx is None:
            xx = self.xx
        yy = self.WeibullFit.eval(xx)
        return yy*self.y_max


def get_weibull_interpolation(xx, yy, target_dprime):
    # max y has to be scaled to 1
    WB = ScaledWeibull(xx, yy)
    result = WB.interpol(target_dprime)
    return result


if __name__ == '__main__':
    dprimes = np.array([-0.07554976,  0.02006533, -0.08515377, -0.00879679,  0.02481237,
                         0.04760913,  0.15243606,  0.61620849,  2.43833205,  4.78657647,
                         6.7055888 ,  7.08037796])
    metric_values = np.array([1.00000000e-05, 1.99526231e-05, 3.98107171e-05, 7.94328235e-05,
                              1.58489319e-04, 3.16227766e-04, 6.30957344e-04, 1.25892541e-03,
                              2.51188643e-03, 5.01187234e-03, 1.00000000e-02, 1.99526231e-02])
    dprime_target = 1.5
    SWeibull = ScaledWeibull(metric_values, dprimes, sems=1)
    y_pred = SWeibull.evaluate(metric_values)
    # interpolation = get_weibull_interpolation(metric_values, dprimes, dprime_target)
    print('nice!')

