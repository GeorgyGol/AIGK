"""Detachmentt of the seasonal components of the time series

    The script contains the function of dividing the time series into a seasonal wave and a trend.
    The dividing occurs according to the algorithm of V. Gubanov (http://www2.forecast.ru/Metodology/Gubanov/VGubanov_05.pdf)
    The function removes additive or multiplacial seasonality, the wave can be dynamic or static

    This file can also be imported as a module and contains the following functions:

    * seasonal_decompose - returns the trend, wave, error and row, cleared from outliers
    * test - retrun pandas DataFrame with seasonal_decompose of testing row

"""

__author__ = "G. Golyshev, V.Gubanov, V. Salnikov"
__copyright__ = "CMASF 2020"
__version__ = "0.0.1"
__maintainer__ = "G. Golyshev"
__email__ = "g.golyshev@forecast.ru"
__status__ = "Production"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import stats


class _SeasonWave():
    _row=None
    _period=12
    _static=1
    _model='add'
    _seas_matrix=None

    def __init__(self, source_arr, period=12, model='add', static=1):
        self._model = model

        if model[:3].lower()=='add':
            self._row=source_arr
        elif model[:4].lower()=='mult':
            self._row = np.log(source_arr)
        else:
            raise TypeError('sesonal decompose model undefined')
        self._period=period

        self._static=static
        self._seas_matrix=self._calc_season_matrix()

    def var_trend(self, row):
        return sum([(row[i + 2] - 2 * row[i + 1] + row[i]) ** 2 for i in range(len(row) - 2)]) / len(row)
        # return np.var(row, ddof=0)

    def var_wave(self, row, period):
        SumW = 0
        for i in range(self._period):
            for k in range(row.size // self._period - 1):
                SumW += (row[(k + 1) * self._period + i] - row[k * self._period + i]) ** 2
        return SumW / len(row)

    def _input_model(self, row):
        if self._model[:3].lower()=='add':
            self._row=row
        elif self._model[:4].lower()=='mult':
            self._row = np.log(row)
        else:
            raise TypeError('sesonal decompose model undefined')

    def _output_model(self, wave):
        if self._model[:3] == 'add':
            return self._row - wave, wave, self._row

        elif self._model[:4] == 'mult':
            return np.exp(self._row - wave), np.exp(wave), np.exp(self._row)
        else:
            raise TypeError('sesonal decompose model undefined')

    def _calc_season_matrix(self):
        """расчет матрицы сезонноси"""
        npbase = np.zeros(self._period)
        npbase[0] = -2
        npbase[-1] = 1
        npbase[1] = 1
        res = [np.roll(npbase, i) for i in range(self._period)]
        res[-1] = np.ones(self._period)
        return np.linalg.inv(np.array(res))

    def _calc_sec_diff(self, row):
        res = np.zeros((len(row) // self._period, self._period))
        src = np.append(row, [row[-1], np.nan])

        for k in range(res.shape[0]):
            res[k, :] = [src[i + k * self._period] - (2 * src[i + 1 + k * self._period]) + src[i + 2 + k * self._period] for i in
                         range(self._period)]
        res[:, -1] = 0
        return res

    def _norm_vect(self, cnt_periods, gamma):
        return np.array([(1 - gamma) / (1 + gamma - gamma ** k - gamma ** (cnt_periods - k + 1)) for k in
                         range(1, cnt_periods + 1)])

    def _weight_matrix(self, row, gamma):
        """правая часть уравнения по Губанову"""
        m_dif2=self._calc_sec_diff(row)
        d_ = np.zeros((m_dif2.shape[1], m_dif2.shape[0]))

        v_weight = self._norm_vect(m_dif2.shape[1], gamma)

        for k in range(m_dif2.shape[0]):
            for j in range(m_dif2.shape[1]):
                ds_ = np.sum([gamma ** (abs(k - l)) * m_dif2[l, j] for l in range(m_dif2.shape[0])])
                d_[j, k] = ds_ * v_weight[j]
        return d_

    def _get_wave(self, row, gamma):
        wave = np.ravel(np.matmul(self._seas_matrix, self._weight_matrix(row, gamma)), order='F')
        return np.insert(wave, 0, 0)[:len(row)]

    @property
    def kper(self):
        """количество целых периодов внутри ряда длинной row_lenght"""
        return len(self._row) // self._period

    @property
    def period(self):
        return self._period

    @property
    def last_period_len(self):
        return len(self._row) % self._period

    @property
    def row(self):
        return self._row

    @property
    def season_matrix(self):
        return self._seas_matrix

    @property
    def sec_diff(self):
        return self._calc_sec_diff(self._row)

    def seasX4(self, gamma):
        offs = self.last_period_len
        offs = self._period if offs == 0 else offs
        ins_a = [np.nan] * offs

        w_start = np.append(self._get_wave(self._row[:-offs], gamma), ins_a)
        w_end = np.insert(self._get_wave(self._row[offs:], gamma), 0, ins_a)

        wave = np.nanmean(np.array([w_start, w_end]), axis=0)

        try:
            wave[0] = 2 * w_start[self._period] - w_start[self._period * 2]
        except:
            wave[0] = w_start[self._period]

        return self._output_model(wave)

    def Err_X4(self, trend, wave):
        return self.var_trend(trend) * (-0.5 * self._static + 0.5) + (0.5 * self._static + 0.5) * self.var_wave(wave, self._period)

    def Variance(self, gamma):
        trend, wave, _ = self.seasX4(gamma)
        return self.Err_X4(trend, wave)

def seasonal_decompose(row, period=12, gamma=0.01, static=0.5, model='additive', precision=0.001):
    """Detachmentt of the seasonal components of the time series
        params: row - source row - time series, 1D numpy.array
                periods - points in one period
                gamma - dynamic param for wave varians, if > 1 - function find optimal gamma itself
                static - if =1 the wave will be static, if = 0 - wave will be dynamic, between 0 and 1 - partial static
                model - 'additive' aor 'multiplicative', define wave model
                precision -  precision for gamma calculation if gamma calculating itself

        return: trend, wave, variance (error), cleared source row
                variance (error) - scalar, trend, wave and row - numpy 1D arrays

        example: trend, wave, err, out_row = seasonal_decompose(row, period=12, gamma=2, static=0.1, model='add')
    """
    x=_SeasonWave(row, period=period, static=static, model=model)

    if 0 < gamma <= 1:
        trend, wave, out_row = x.seasX4(gamma)
        err = x.Err_X4(trend, wave)
        return trend, wave, err, out_row

    r = (3 - math.sqrt(5)) / 2
    A = 0
    c = 1
    k = 0

    while k < 30:
        B = A + r * (c - A)
        d = c + r * (A - c)

        err1 = x.Variance(d)
        err2 = x.Variance(B)

        if abs(1 - err1 / err2) < precision: break

        if err1 < err2:
            A = B
        else:
            c = A
            A = d
        k += 1

    alfa = d if err1 < err2 else B

    trend, wave, out_row = x.seasX4(alfa)
    err = x.Err_X4(trend, wave)

    return trend, wave, err, out_row

def test():
    # inp= np.array( (305.6, 296.6, 286.1, 303.4, 287.5, 293.4, 286.2, 292, 290, 285.8, 307.8, 302.1,
    #                 311, 301, 280.7, 311, 299.1, 307.7, 297.3, 299.4, 302.5, 299.9, 326.9, 313.7, 311.3, 311.4, 297.9,
    #                 328, 315.6, 323.5, 312, 321.3, 321.7, 322, 341.7, 331.2, 341.7, 340.2, 315.7, 353.1, 336.5, 347.7,
    #                 339.8, 346.5, 350.2, 342.5, 367.4, 357.5, 373.9, 362.3, 345.4, 376.6, 367.1, 371.6, 357.8, 366.1,
    #                 373.1, 366.1, 380.2, 375.7325, 386.6126) ) #np.around(np.random.random(24), 2)

    inp = np.array((305.6, 296.6, 286.1, 303.4, 287.5, 293.4, 286.2, 292, 290, 285.8, 307.8, 302.1,
                    311, 301, 280.7, 311, 299.1, 307.7, 297.3, 299.4, 302.5, 299.9, 326.9, 313.7, 311.3, 311.4, 297.9,
                    328, 315.6, 323.5, 312, 321.3, 321.7, 322, 341.7, 331.2, 341.7, 340.2, 315.7, 353.1, 336.5, 347.7,
                    339.8, 346.5, 350.2, 342.5, 367.4, 357.5, 373.9, 362.3, 345.4, 376.6, 367.1, 371.6, 357.8, 366.1,
                    373.1, 366.1, 380.2, 375.7325, 386.6126))  # np.around(np.random.random(24), 2)

    np.set_printoptions(precision=3, suppress=True)
    row = inp[:-2]
    trend, wave, err, out_row = seasonal_decompose(row, period=12, gamma=2, static=0.1, model='add')
    pdf = pd.DataFrame({'row': out_row, 'wave': wave, 'trend': trend})

    return pdf


if __name__ == "__main__":

    p=test()

    p.plot.line()
    plt.show()

    print('All done')
