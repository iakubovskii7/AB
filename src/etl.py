# http://clickhouse.aksimemor.me:8123/
# Логин: appbooster
# Пароль: 47NVdj3RVffSDAe31v
#
# Для входа ходили сюда — http://ui.tabix.io/
#
# Событие с распределением называется SplitTest (столбец event_name),
# в столбце event_value объект, один из его параметров TestName — принадлежность к конкретному тесту,
# Group — принадлежность к группе теста.
#
# В столбце event_revenue — доход от события

# Эксперимет в столбце paywallVariations, группы просто по номерам))
# В столбце со словами total spend — доход от юзера

from typing import List, Set, Dict, Tuple, Optional
import os
import glob
import random
import string
import requests
import pandas as pd
import numpy as np
import json

class JupyterInterruptException():
    pass
def clickhouse_query(query, user="appbooster", password="47NVdj3RVffSDAe31v",
                     host="clickhouse.aksimemor.me:8123", connection_timeout=1500):
    """
    Extract data from clickhouse database with user, password, host
    :param query: query for ClickHouse database
    :param user: user
    :param password: password
    :param host: host (8123 for hhtp, 9000 for local)
    :param connection_timeout: timeout query
    :return:  query results in json format: our interest lies in 'data' key
    """
    query += ' FORMAT JSON'
    query_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(16))

    try:
        r = requests.post('http://{}:{}@{}/?query_id='.format(user, password, host) + query_id, data=query)
        r.raise_for_status()
    except KeyboardInterrupt as e:
        kill_query = "KILL QUERY WHERE user='{}' AND query_id='%s' ASYNC".format(user) % query_id
        kill_r = requests.post('http://{}:{}@{}/'.format(user, password, host), data=kill_query)
        kill_r.raise_for_status()
        raise JupyterInterruptException()
    return r.json()


def clickhouse(query, clickhouse_query) -> pd.DataFrame:
    """
    Process query pandas to dataframe
    :param query:
    :param clickhouse_query:
    :return:
    """
    result = clickhouse_query(query)
    result_df = pd.DataFrame.from_records(result['data'])

    type_converters = {'UInt8': np.uint8, 'UInt16': np.uint16, 'UInt32': np.uint32, 'UInt64': np.uint64,
                       'Int8': np.int8, 'Int16': np.int16, 'Int32': np.int32, 'Int64': np.int64,
                       'Float32': np.float32, 'Float64': np.float64,
                       'Date': pd.to_datetime, 'DateTime': pd.to_datetime}

    for col_data in result['meta']:
        col_type_converter = type_converters.get(col_data['type'], str)
        result_df[col_data['name']] = result_df[col_data['name']].apply(col_type_converter)
    return result_df

def clickhouse_transform_pandas(result) -> pd.DataFrame:
    """
    Process query pandas to dataframe
    :param query:
    :param clickhouse_query:
    :return:
    """
    result_df = pd.DataFrame.from_records(result['data'])

    type_converters = {'UInt8': np.uint8, 'UInt16': np.uint16, 'UInt32': np.uint32, 'UInt64': np.uint64,
                       'Int8': np.int8, 'Int16': np.int16, 'Int32': np.int32, 'Int64': np.int64,
                       'Float32': np.float32, 'Float64': np.float64,
                       'Date': pd.to_datetime, 'DateTime': pd.to_datetime}

    for col_data in result['meta']:
        col_type_converter = type_converters.get(col_data['type'], str)
        result_df[col_data['name']] = result_df[col_data['name']].apply(col_type_converter)
    return result_df

# ab_test_df = pd.read_csv("Data/Sweet_sex_datimg_AB.csv")
def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = [value , dict1[key]]
    return dict3


# import os
# import pandas as pd
# os.chdir("/Users/iakubovskii/AppBooster/Base/AB_testing/")
# am7 = pd.read_csv("amplitude_users-7.csv")
# am6 = pd.read_csv("amplitude_users-6.csv")
# am6['test'] = 6
# am7['test'] = 7
# ab_test_df = pd.concat([am6, am7])

# ab_test_df.columns = ab_test_df.columns.str.replace("\t", "")
# for col in ab_test_df.columns:
#     try:
#         ab_test_df[col] = ab_test_df[col].str.replace("\t", "")
#     except:
#         pass
# gp_columns = [i for i in ab_test_df.columns if "gp" in i]
# for col in gp_columns:
#     ab_test_df[col] = pd.to_numeric(ab_test_df[col], errors = 'coerce')
#     ab_test_df[col] = ab_test_df[col].fillna(0)

# ab_test_df[ab_test_df['test'] == 6]['gp:paywallVariations'].value_counts()
# from typing import List, Set, Dict, Tuple, Optional
# def create_utility_dict(df,
#                         test_group_name: str,
#                         revenue_group_name:str) -> Dict[str, float]:
#     """
#     Create utility dict from amplitude dataframe
#     :param df: amplitude dataframe
#     :param test_group_name: name of columns with group names
#     :param revenue_group_name: name of columns with REVENUE VALUE
#     :return: utility dictionary ;
#     """
#     groups_unique = df[test_group_name].unique()
#     ut_d = {str(int(group)) : df[df[test_group_name] == group][revenue_group_name].values.tolist()
#                     for group in groups_unique}
#     return(ut_d)

# revenue_dict_6 = create_utility_dict(ab_test_df[ab_test_df['test'] == 6],
#                                      "gp:paywallVariations",
#                                      "gp:[Apphud] total_spent")
# revenue_dict_7 = create_utility_dict(ab_test_df[ab_test_df['test'] == 7],
#                                      "gp:paywallVariations",
#                                      "gp:[Apphud] total_spent")
# import json
# with open('revenue_dict_6.json', 'w') as f:
#     json.dump(revenue_dict_6, f)
# with open('revenue_dict_7.json', 'w') as f:
#     json.dump(revenue_dict_7, f)




