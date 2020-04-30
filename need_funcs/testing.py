"""For development and testing - not importing to main program
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import stats

# def _inc(i):
#     return i + 1

def var_trend(row):
    return sum([ (row[i+2] - 2*row[i+1] + row[i])**2 for i in range(len(row)-2)])/len(row)
    # return np.var(row, ddof=0)

def var_wave(row, period):
    SumW = 0
    for i in range(period):
        for k in range( row.size // period - 1):
            SumW += (row[ (k+1) * period + i] - row[k * period + i]) ** 2
    return SumW / len(row)

def var1_wave(wave, period):
    kp=len(wave) // period
    dop_kp=len(wave) % period
    if dop_kp:
        w=np.append(wave, [np.nan]*(period-dop_kp))
        kp+=1
    else:
        w=wave

    w=np.reshape(w, (kp, period) )
    werr=np.nansum([stats.sem(w[:, k]) for k in range(period)])
    return werr



def Err_X4(trend, wave, period, dymanics):
    return var_trend(trend)*(-0.5*dymanics+0.5) + (0.5*dymanics+0.5) * var_wave(wave, period)

def season_matrix(n):
    """расчет матрицы сезонноси"""
    npbase=np.zeros(n)
    npbase[0]=-2
    npbase[-1] = 1
    npbase[1]=1
    res=[np.roll(npbase, i) for i in range(n)]
    res[-1]=np.ones(n)
    return np.linalg.inv(np.array(res))

def sec_diff(row, period):
    res = np.zeros((row.size // period, period))
    src = np.append(row, [row[-1], np.nan])

    for k in range(res.shape[0]):
        res[k, :] = [src[i + k * period] - (2 * src[i + 1 + k * period]) + src[i + 2 + k * period] for i in range(period)]
    res[:, -1] = 0
    return res

def kper(row_lenght, period=12):
    """количество целых периодов внутри ряда длинной row_lenght"""
    return row_lenght // period

def norm_vect(gamma, cnt_periods):
    return np.array( [(1 - gamma) / (1 + gamma - gamma ** k - gamma ** (cnt_periods - k + 1)) for k in range(1, cnt_periods+1)] )

def weight_matrix(m_dif2, gamma):
    """правая часть уравнения по Губанову"""
    d_=np.zeros( (m_dif2.shape[1], m_dif2.shape[0]) )

    v_weight=norm_vect(gamma, m_dif2.shape[1])

    for k in range(m_dif2.shape[0]):
        for j in range(m_dif2.shape[1]):
            ds_= np.sum([gamma ** (abs(k-l)) * m_dif2[l, j] for l in range(m_dif2.shape[0])])
            d_[j, k] = ds_ *  v_weight[j]
    return d_

def get_wave(row, period_len, gamma):
    d2 = sec_diff(row, period_len)

    mw = weight_matrix(d2, gamma)
    ms = season_matrix(period_len)
    wave = np.ravel(np.matmul(ms, mw), order='F')

    return np.insert(wave, 0, 0)[:len(row)]

def seasX4(source_row, periods, gamma, model='additive'):

    if model[:4]=='mult':
        row=np.log(source_row)
    else:
        row = source_row

    offs = len(row) % periods
    offs = periods if offs == 0 else offs
    ins_a=[np.nan] * offs

    w_start = np.append(get_wave(row[:-offs], periods, gamma), ins_a)
    w_end = np.insert(get_wave(row[offs:], periods, gamma), 0, ins_a)

    wave = np.nanmean( np.array( [w_start,  w_end] ), axis=0)

    try:
        wave[0] = 2 * w_start[periods]-w_start[periods*2]
    except:
        wave[0] = w_start[periods]
    if model[:3]=='add':
        return row - wave, wave

    elif model[:4]=='mult':
        return np.exp(row - wave), np.exp(wave)
    else:
        raise TypeError('sesonal decompose model undefined')

def season(row, period=12, gamma=0.01, dynamics=0.5, model='additive', precision=0.001):

    if  0 < gamma <= 1:
        trend, wave = seasX4(row, period, gamma, model=model)
        err = Err_X4(trend, wave, period, dynamics)
        return trend, wave, err

    r = (3 - math.sqrt(5)) / 2
    A = 0
    c = 1
    k = 0

    while k<30:
        B = A + r * (c - A)
        d = c + r * (A - c)
        trend, wave = seasX4(row, period, d, model=model)
        err1 = Err_X4(trend, wave, period, dynamics)
        trend, wave = seasX4(row, period, B, model=model)
        err2 = Err_X4(trend, wave, period, dynamics)

        if abs(1 - err1 / err2) < precision:
            break

        if  err1 < err2:
            A = B
        else:
            c = A
            A = d
        k+=1

    alfa = d if err1 < err2 else B

    trend, wave=seasX4(row, period, alfa, model=model)
    err=Err_X4(trend, wave, period, dynamics)

    return trend, wave, err

def test_wave(wave, period):
    def reject_outliers(data, m=3):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    kp=len(wave) // period
    dop_kp=len(wave) % period
    if dop_kp:
        w=np.append(wave, [np.nan]*(period-dop_kp))
        kp+=1
    else:
        w=wave

    w=np.reshape(w, (kp, period) )

    return w

if __name__ == "__main__":
    #print(np.round(season_matrix(12), 2))
    # print(norm_vect(gamma=0.01, cnt_periods=8))
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
    #print(np.round(diff2(inp), 1))

    # print(len(inp)//6)
    # w=get_wave(inp, 12, 0.01)
    # print(w)
    # print(len(inp), len(w))
    # #
    # pdf=pd.DataFrame({'val':inp, 'wave':w, 'trend':inp-w})
    # pdf.plot.line()
    # plt.show()



    # trend, wave=seasX4(inp, 12, 0.01, model='add')
    #

    #
    # print(var1_wave(wave, 12))
    # pdf=pd.DataFrame({'row':inp, 'wave':wave, 'trend':trend})
    # pdf.plot.line()
    # plt.show()
    # print(wave)
    # #
    # # print(trend.std(), math.sqrt(trend.std()))
    # # print(trend.var(), math.sqrt(trend.var()))
    # # print('sem=',stats.sem(trend))
    # #
    # print(var_trend(trend))
    # # print()
    # print(var_wave(wave, 12))
    def clear_outliers(row, period):
        def pik_outliers(data, m=3):
            np.place(data, abs(data - np.nanmean(data)) > m * np.nanstd(data), [np.nanmean(data)])
            return data

        w = test_wave(row, period)
        w = np.ravel([pik_outliers(w[k, :]) for k in range(w.shape[0])], order='C')
        return w[~np.isnan(w)]

    print('All done')
