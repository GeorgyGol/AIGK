from need_funcs.season import seasonal_decompose
# import statsmodels.api as sm
from database.paths import *
import sqlite3
import sqlalchemy as sa
import matplotlib.pyplot as plt
import pandas as pd
from os import path
import re
import json

def get_row(con=None, code=0, parse_dates=['date',]):
    assert con
    assert code

    strSelectH='select * from headers where '
    strSelectD='select date, value from datas where code = {index}'

    if type(code)==type(int()):
        strSelectH += 'code = {}'.format(code)
    elif type(code)==type(str()):
        if re.search('[А-Я ]]', code):
            strSelectH+='name LIKE "{}"'.format(code)
        else:
            strSelectH += 'code2="{}"'.format(code)
    else:
        raise TypeError('get row Error - uknown  type of DB inedex')


    pdfH = pd.read_sql(strSelectH, con=con, index_col='code')

    s=strSelectD.format(index=pdfH.index.tolist()[0])
    pdfD = pd.read_sql(s, con=con, index_col='date', parse_dates=parse_dates)

    try:
        params = json.loads(pdfH['params'].values[0])
        # print(pd.__version__)
        trend, wave, err, cor_row = seasonal_decompose(pdfD['value'].values, **params['SEASON'])
        # trend, wave, err, cor_row = seasonal_decompose(pdfD['value'].as_matrix(), period=4, static=0)
        pdfD[strSAdjField]=trend
        pdfD['wave - model:{type}, static:{stat}'.format(type=params['SEASON']['model'],
                                                                         stat=params['SEASON']['static'])] = wave
        pdfD['pct']=pdfD[strSAdjField].pct_change() *100
    except KeyError:
        pass
    except ValueError:
        pass

    return pdfH, pdfD

def read_quart_db():
    coni_q = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_q))
    pdfQ = pd.read_sql('select * from headers where code2 != "{}"'.format(strGroupHeadKey), con=coni_q)
    return {str(i): get_row(con=coni_q, code=str(i)) for i in pdfQ['code2'].values}

def read_month_db():
    coni_m = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_m))
    pdfM = pd.read_sql('select * from headers where code2 != "{}"'.format(strGroupHeadKey), con=coni_m)
    return {str(i): get_row(con=coni_m, code=str(i)) for i in pdfM['code2'].values}

def read_year_db():
    coni_y = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_y))
    pdfY = pd.read_sql('select * from headers', con=coni_y)
    return {int(i): get_row(con=coni_y, code=int(i), parse_dates=None) for i in pdfY['code'].values}

def read_full_db():
    return read_year_db, read_quart_db, read_month_db

def test_get_row():
    coni_q = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_q))
    coni_m = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_m))

    row_name = 'QT_D_M2new'
    pdfH, pdfD = get_row(con=coni_q, code=row_name)
    params = json.loads(pdfH['params'].values[0])
    print(params['SEASON'])
    # pdfD.to_csv('{name}_mod-{model}_stat-{stat}.csv'.format(name=row_name, model=params['SEASON']['model'], stat=params['SEASON']['static']), sep=';', encoding='cp1251')
    pdfD.plot.line(secondary_y='pct', title=str(pdfH['code2'].values[0]))
    plt.show()
    # print(pdfD)

def test_read_database():
    dcQ = read_quart_db()
    row_name='Qr_X_Gdp'
    pdH, pdD = dcQ[row_name] #Qr_S_Ind Qr_I_build Qr_X_Gdp Ipc_P_Cpi Qt_Mort_sup Qt_H_Inc Qr_H_Wavg Qr_H_Incdsp QT_D_M2new
    # pdD['pct']=pdD[strSAdjField].pct_change()

    pdD.plot.line(secondary_y='pct', title=pdH['code2'].values[0])
    plt.show()

if __name__ == "__main__":
    test_get_row()
    #test_read_database()
    print('All done')