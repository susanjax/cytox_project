import Decoder
import pandas as pd
import numpy as np

#generate new database with the features of materials
db1 = pd.read_csv('Cytotoxicity.csv')
db = pd.read_csv('Additional_material_feature.csv')
# dbr = pd.read_csv('output/results/ovary/result_evaluation_trial.csv')
decoderr = pd.read_csv('decoded1.csv')
df = db.iloc[: , 1:]
db1 = pd.merge(db1, df, on=['material'], how='left')
#print(db1)

#db1.to_csv('combined_dataset_latest.csv')

#transform using scaler and decoder
colmn = ['time (hr)', 'concentration (ug/ml)', 'Hydrodynamic diameter (nm)', 'Zeta potential (mV)', 'mcd', 'electronegativity', 'rox', 'radii', 'norm_viability']

def transformed_input(dbr):
    input1 = db1.iloc[:, [3, 0, 2, 1]] # choosing only cell type and test
    input2 = db1.iloc[:, [3, 4, 6, 7, 8, 9, 10, 11, 5]]
    print(input2)
    decoder = Decoder.Decoder()
    scaler = Decoder.Scaler()
    input1_trans = decoder.transform(input1)
    input1_transform = input1_trans.iloc[:,[2, 1, 3 ]]
    #print(input1_transform)
    input2_transform = scaler.fit_transform(input2)
    part = dbr.iloc[:, [6, 7, 8, 9, 10, 11, 12, 13, 14]]
    part2 = dbr.iloc[:, [ 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    #print(part)
    original_in = scaler.inverse_transform(part)
    scalerdata = pd.DataFrame(original_in, columns=colmn)
    original_in2 = scaler.inverse_transform(part2)
    scalerdata2 = pd.DataFrame(original_in2, columns=colmn)
    new_scalerdata2 = scalerdata2.rename(columns={'norm_viability':'canc_viability'})
    only_canc = new_scalerdata2[new_scalerdata2.columns[-1]]
    #print(only_canc)
    final_trans = pd.concat([scalerdata, only_canc], axis=1, join='inner')
    #print(final_trans)
    input2_data = pd.DataFrame(input2_transform, columns= [ 'time (hr)', 'concentration (ug/ml)', 'Hydrodynamic diameter (nm)', 'Zeta potential (mV)', 'mcd', 'electronegativity', 'rox', 'radii', 'viability (%)'])
    compound_list = pd.concat([input1_transform, input2_data], axis=1, join="inner")
    return final_trans
    #return compound_list


# transformed_input()

def rev_transform(dbr):
    material_decoder = decoderr.iloc[:, 3]
    material = dbr[['material']]
    a = 0
    mate = []
    for a in range(a, len(material)):
        mate.append(material_decoder[int(material.iloc[a].values) - 1])

    test_decoder = decoderr.iloc[:, 5]
    test_type = dbr[['test']]
    b = 0
    tes = []
    while b < len(test_type):
        tes.append(test_decoder[int(test_type.iloc[b].values) - 1])
        b += 1

    cell_decoder = decoderr.iloc[:, 1]
    norm_cell = dbr[["Normal cell"]]
    # print('norm', norm_cell)
    canc_cell = dbr[["Cancer cell"]]
    c = 0
    norm = []
    canc = []
    for c in range (c, len(norm_cell)):
        norm.append((cell_decoder[int(norm_cell.iloc[c].values)-1]))
        canc.append((cell_decoder[int(canc_cell.iloc[c].values)-1]))

    # print(tes)
    scaler = transformed_input(dbr=dbr)
    all = {'test': tes, 'material': mate, 'Normal cell': norm, 'Cancer cell': canc }
    print (all)
    datafram = pd.DataFrame(all)
    final = pd.concat([datafram, scaler], axis=1, join='inner')
    #print(final)
    # final.to_csv('output/results/ovary/result_transformation_80.csv')
    return final

# rev_transform(dbr)

#df = transformed_input()
#df.to_csv('output/transformed_data_new.csv')
