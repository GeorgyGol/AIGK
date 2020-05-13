from os import path

strBaseFile=r'База данных.xlsm'
strBasePath='..'

strSQL_path=path.join(strBasePath, 'DB')
strSQL_y='year.sqlite3'
strSQL_q='quar.sqlite3'
strSQL_m='month.sqlite3'

strSQLite_y = path.join(strSQL_path, strSQL_y)
strSQLite_q = path.join(strSQL_path, strSQL_q)
strSQLite_m = path.join(strSQL_path, strSQL_m)

strGroupHeadKey='MGH'
strGroupField='mgroup_id'

strSAdjField='seas_adj'
strPCTField='pct'