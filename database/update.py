import pandas as pd
import datetime as dt
# from os import path
import re
# import sqlite3
import sqlalchemy as sa
from need_funcs import pandas_sql
from need_funcs.season import seasonal_decompose
from database.paths import *

import warnings
warnings.filterwarnings('ignore')

# strBaseFile=r'База данных.xlsm'
# strBasePath='..'
#
# strSQL_path=path.join(strBasePath, 'DB')
# strSQL_y='year.sqlite3'
# strSQL_q='quar.sqlite3'
# strSQL_m='month.sqlite3'
# strGroupHeadKey='MGH'
# strGroupField='mgroup_id'

def set_group_key(pdf, grp_mask):
    """определяет заголовки групп (главных) по маске, расставляет ключи этих групп остальным рядам"""
    pdf.loc[grp_mask, 'code2'] = strGroupHeadKey
    lst_ind_mgh = pdf.loc[grp_mask].index.tolist()
    for i in lst_ind_mgh:
        pdf.loc[i:, strGroupField] = i

    return pdf


def read_main_file(strFile=path.join(strBasePath, strBaseFile),
                   strSQLite_y=path.join(strSQL_path, strSQL_y),
                   strSQLite_q=path.join(strSQL_path, strSQL_q),
                   strSQLite_m=path.join(strSQL_path, strSQL_m)):
    """read existing excel database (one file) and write data to sqlite-file(s)"""

    coni_y = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_y))
    coni_q = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_q))
    coni_m = sa.create_engine('sqlite+pysqlite:///{db_name}'.format(db_name=strSQLite_m))

    def year_sheet(if_exists='upsert'):
        def data_work(dtf_data):
            pdfd_sql = dtf_data.stack().reset_index().rename(columns={'level_1': 'date', 0: 'value'})
            pds = pandas_sql.DataFrameDATA(pdfd_sql.loc[~pdfd_sql['code'].isin(lstCalcF)].set_index(['code', 'year']))
            pds.to_sql('datas', con=coni_y, if_exists=if_exists)
            return pds

        def header_work(dtf_head):
            pdfRowHead = dtf_head.loc[~dtf_head.index.isin(lstCalcF)]  # инфа по рядам, названия и проч.

            pdfRowHead = set_group_key(pdfRowHead, mskMainGroup)
            pds=pandas_sql.DataFrameDATA(pdfRowHead)
            pds['params']=''
            pds.to_sql('headers', con=coni_y, if_exists=if_exists)
            return pds

        cstrYearSh='YEAR'
        dctRenameY={3:'code', 'Показатели':'name', 'Ед. измер.':'unit', 'Комментарий':'source', 'Код':'code2'}

        lstCalcF=[3, 5, 7, 9] # exclude rows - calculated

        pdfDB=pd.read_excel(strFile, sheet_name=cstrYearSh, header=2).rename(columns=dctRenameY)
        lstWorkCol=[c for c in pdfDB.columns.tolist() if c in dctRenameY.values() or str(c).strip().isdigit()]

        lstDataF=[c for c in lstWorkCol if str(c).isdigit()]
        lstNameF=[c for c in lstWorkCol if not str(c).isdigit()]

        pdfDB=pdfDB[lstWorkCol].set_index('code').dropna(how='all')

        mskGroupHead = pdfDB[lstDataF].isnull().all(axis=1)
        mskMainGroup = mskGroupHead & pdfDB['name'].str.isupper()

        pdf_d=data_work(pdfDB.loc[~mskGroupHead, lstDataF])
        pdf_h=header_work(pdfDB[lstNameF[1:]])
        return pdf_h, pdf_d

    def quar_sheet(if_exists='upsert', sheet_name='QUAR', sql_conn=coni_q):
        def header_work(dtf_head):
            pdfRowHead = set_group_key(dtf_head, mskGroupHead)
            pds=pandas_sql.DataFrameDATA(pdfRowHead)
            pds['params']=''
            pds.to_sql('headers', con=sql_conn, if_exists=if_exists)
            return pds

        def data_work(dtf_data):
            pdfd_sql = dtf_data.stack().reset_index().rename(columns={'level_1': 'date', 0: 'value'})
            pds = pandas_sql.DataFrameDATA(pdfd_sql.set_index(['code', 'date']))

            pds.to_sql('datas', con=sql_conn, if_exists=if_exists)
            return pds

        cstrQuarSh = sheet_name
        dctRename = {'Показатели': 'name', 'Ед. измер.': 'unit', 'Обознач.': 'code2'}

        pdfDB = pd.read_excel(strFile, sheet_name=cstrQuarSh, header=1).rename(columns=dctRename).dropna(how='all')

        lstHead=list(dctRename.values())
        lstData=[c for c in pdfDB.columns if isinstance(c, dt.datetime)]

        pdfDB=pdfDB[lstHead+lstData].loc[:54]
        pdfDB.index.name = 'code'

        mskCalculated = pdfDB['name'].str.contains('(со снятой сезонностью)|(Темп прироста)')
        mskGroupHead=pdfDB[lstData].isnull().all(axis=1)
        pdfWork=pdfDB[~mskCalculated]

        pdf_d = data_work( pdfDB[lstData])
        pdf_h = header_work(pdfWork[lstHead])
        return pdf_h, pdf_d


        # print(pdfDB[~mskCalculated])
        # print(pdfDB)

    year_sheet()
    quar_sheet()
    quar_sheet(sheet_name='MONTH', sql_conn=coni_m) # monthly sheet




def main():
    print(path.exists(path.join(strBasePath, strBaseFile)))

if __name__ == "__main__":
    read_main_file()


    print('All done')