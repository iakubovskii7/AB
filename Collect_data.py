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
import os
import pandas as pd
os.chdir("/Users/iakubovskii/AppBooster/Base/AB_testing/")
am7 = pd.read_csv("amplitude_users-7.csv")
am6 = pd.read_csv("amplitude_users-6.csv")
am6['test'] = 6
am7['test'] = 7
ab_test_df = pd.concat([am6, am7])

ab_test_df.columns = ab_test_df.columns.str.replace("\t", "")
for col in ab_test_df.columns:
    try:
        ab_test_df[col] = ab_test_df[col].str.replace("\t", "")
    except:
        pass
gp_columns = [i for i in ab_test_df.columns if "gp" in i]
for col in gp_columns:
    ab_test_df[col] = pd.to_numeric(ab_test_df[col], errors = 'coerce')
    ab_test_df[col] = ab_test_df[col].fillna(0)

ab_test_df[ab_test_df['test'] == 6]['gp:paywallVariations'].value_counts()
from typing import List, Set, Dict, Tuple, Optional
def create_utility_dict(df,
                        test_group_name: str,
                        revenue_group_name:str) -> Dict[str, float]:
    """
    Create utility dict from amplitude dataframe
    :param df: amplitude dataframe
    :param test_group_name: name of columns with group names
    :param revenue_group_name: name of columns with REVENUE VALUE
    :return: utility dictionary ;
    """
    groups_unique = df[test_group_name].unique()
    ut_d = {str(int(group)) : df[df[test_group_name] == group][revenue_group_name].values.tolist()
                    for group in groups_unique}
    return(ut_d)

revenue_dict_6 = create_utility_dict(ab_test_df[ab_test_df['test'] == 6],
                                     "gp:paywallVariations",
                                     "gp:[Apphud] total_spent")
revenue_dict_7 = create_utility_dict(ab_test_df[ab_test_df['test'] == 7],
                                     "gp:paywallVariations",
                                     "gp:[Apphud] total_spent")
import json
with open('revenue_dict_6.json', 'w') as f:
    json.dump(revenue_dict_6, f)
with open('revenue_dict_7.json', 'w') as f:
    json.dump(revenue_dict_7, f)




