from need_funcs.season import seasonal_decompose
from database.paths import *
import sqlite3
import sqlalchemy as sa
import matplotlib.pyplot as plt
import pandas as pd
from os import path

def season_test():


    coni_y = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_y))
    coni_q = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_q))
    coni_m = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_m))

    pdf_Qr_S_Ind=pd.read_sql('select date, value from datas where code = 5',
                             con=coni_q, parse_dates=['date',], index_col='date')

    trend, _, _, _=seasonal_decompose(pdf_Qr_S_Ind['value'].as_matrix(), period=4, static=0.6)
    pdf_Qr_S_Ind['seas_adj']=trend
    pdf_Qr_S_Ind['pct'] = pdf_Qr_S_Ind['seas_adj'].pct_change()
    print(pdf_Qr_S_Ind)
    pdf_Qr_S_Ind.plot.line(secondary_y='pct')
    plt.show()

if __name__ == "__main__":
    season_test()

    print('All done')