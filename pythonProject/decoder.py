
import pandas as pd
import  sklearn
import seaborn as sns

#inputs
df = pd.read_csv('Data/transformed/V3/preprocessed_data.csv')
df =df.drop(['Unnamed: 0'], axis=1)
df = df.reset_index(drop=True)
# print('new', df)
result_df = pd.read_csv('Data/transformed/V3/preprocessed_result_for_validation.csv')
result_df =result_df.drop(['Unnamed: 0'], axis=1)

"""##Standard scaler and Ordinal encoding on original and external data"""
label = df[['viability (%)']]
# print('ll', label)
df_cat = df.select_dtypes(include=['object'])
df_cat = df_cat.drop(['source'], axis=1)
df_n = df.select_dtypes(include=['float64'])
df_num = df_n.drop(['viability (%)'], axis=1)

from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
oe.fit(df_cat)
X_c = oe.transform(df_cat)
# Xval_c = oe.transform(Xvalcat_col)
X_c = pd.DataFrame(X_c, columns=df_cat.columns)
# Xval_c = pd.DataFrame(Xval_c, columns=Xvalcat_col.columns)

# print('cat',X_c)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_all = sc.fit_transform(df_num)
X_val_all = sc.transform(df_num)
X_sc = pd.DataFrame(X_all, columns=df_num.columns, index =df_num.index)
# print(('num', X_sc))

dataf = pd.concat([X_c, X_sc, label], axis=1)
# print('final',dataf)

def encode_data(df):
    dfc = df[df_cat.columns]
    print(dfc.info())
    dfn = df[df_num.columns]
    encode_cat = oe.transform(dfc)
    encode_num = sc.transform(dfn)
    df_c = pd.DataFrame(encode_cat, columns=df_cat.columns)
    df_n =pd.DataFrame(encode_num, columns=df_num.columns)
    df_out = pd.concat([df_c, df_n], axis=1)
    df_out.to_csv('scaled_result_validation.csv')
    ''' this step is required if you want to mannually encode data'''
    return df_out

# print(encode_data(result_df))
encode_data(result_df)
def decode_transformed(df):
    # label = df[['viability (%)']]
    dc = df[df_cat.columns]
    # print('dc',dc)# df_cat = df.iloc[:, 0:9]
    dn = df[df_num.columns]
    # df_num = df.iloc[:, 9:27]
    trans_cat = oe.inverse_transform(dc)
    trans_num = sc.inverse_transform(dn)
    X_c = pd.DataFrame(trans_cat, columns=df_cat.columns)
    X_sc = pd.DataFrame(trans_num, columns=df_num.columns)
    dataf = pd.concat([X_c, X_sc], axis=1)
    return dataf

