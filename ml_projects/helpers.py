import numpy as np 
import matplotlib.pyplot as plt
import sqlite3
from pandas import DataFrame as df



def plot_by_feature(df_, output_feature):
    input_features = df_.columns.to_list()
    input_features.remove(output_feature)

    for input_feature in input_features:
        if df_[input_feature].dtype not in [np.int64, np.float64]:
            continue
            
        plt.plot(df_[input_feature], df_[output_feature], 'r.')
        plt.xlabel(input_feature)
        plt.ylabel(output_feature)
        plt.suptitle(input_feature)
        plt.show()
        print('\n\n\n')



class SqlDb(object):
    def __init__(self, *, database_name = None, upgrade_sqlite_version = False):
        if upgrade_sqlite_version:
            self._upgrade_sqlite_version()
        if database_name:
            if database_name.lower().endswith('.db'):
                database_name = database_name.lower().replace('.db', '')
            self.con = sqlite3.connect(f'{database_name}.db')
        else:
            self.con = sqlite3.connect(':memory:')
        self.cur = self.con.cursor()    
    
    def create_table(self, table_name, *, data_df):
        self.delete_table(table_name)
        return data_df.to_sql(name = table_name, con = self.con)

    def delete_table(self, table_name):
        return self.cur.execute(f"DROP TABLE IF EXISTS {table_name}")

    def insert_into(self, table_name, *, query_string):
        data_df = self.query(query_string)
        self.create_table(table_name, data_df = data_df)
        return data_df
    
    def query(self, query_string):
        res = self.cur.execute(query_string)
        header = list(map(lambda x: x[0], res.description))
        data = res.fetchall()
        data_df = df(data = data, columns = header)
        data_df.drop(columns=['index'], inplace=True, errors='ignore')
        return data_df

    @staticmethod
    def _upgrade_sqlite_version():
        # Upgrade new version of SQLite, run this then reset runtime
        # Thanks to this StackOverflow answer: https://stackoverflow.com/a/59429952
        print("""Please run these line in your Google Colab notebook to upgrade SQLite
%cd
!wget https://www.sqlite.org/src/tarball/sqlite.tar.gz?r=release -O sqlite.tar.gz
!tar xzf sqlite.tar.gz
%cd sqlite/
!./configure
!make sqlite3.c
%cd
!npx degit coleifer/pysqlite3 -f
!cp sqlite/sqlite3.[ch] .
!python setup.py build_static build
!cp build/lib.linux-x86_64-3.6/pysqlite3/_sqlite3.cpython-36m-x86_64-linux-gnu.so \
    /usr/lib/python3.6/lib-dynload/_sqlite3.cpython-36m-x86_64-linux-gnu.so
        """)