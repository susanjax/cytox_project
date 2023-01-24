import pandas as pd
import os
import glob
import models

df = pd.read_csv('scaled_result_validation.csv')
# df = pd.read_csv('Data/transformed/V3/scaled_external_material_cell.csv')
# df = df.drop(['viability (%)'], axis=1)
# print(df.loc[2])
df = df.drop([ 'Unnamed: 0'], axis=1)
def merge_all_result():
    dff = pd.DataFrame()
    path = os.getcwd()
    csv_files = glob.glob(os.path.join(path, "output/results/result_eval_late/*.csv"))
    for f in csv_files:
        # read the csv file
        df = pd.read_csv(f)
        dff = pd.concat([dff, df])
        # print(dff)

    df2 = dff.sort_values('Fitness', ascending=False)
    df3 = df2[df2['Fitness']>20]
    df3.drop_duplicates()
    df3.to_csv('output/results/result_eval_late/all.csv')
    return

def result_validation_predict(df):
    df = df.reset_index(drop=True)
    predict = models.lgbm_predict(df)
    return predict

predict = result_validation_predict(df)

result_df = pd.read_csv('Data/transformed/V3/preprocessed_result_for_validation.csv')
result_df['predicted'] = predict
print(result_df.info())
result_df.to_csv('Data/transformed/V3/predicted_res.csv')