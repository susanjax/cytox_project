import pandas as pd
import models
import random
import decoder
population_size = 100

df = original = pd.read_csv('C:/Users/jax/Desktop/pythonProject/v2/data/original_preprocessed.csv')
print(df.info())
X = df.drop([ 'Unnamed: 0', 'viability (%)'], axis=1)

# it might be better not to choose random generation from unique set, we can just choose from all data ( higher number of data have higher chance to pick, this will imporve the predictibility of the material as model predict best for those who have higher number of data)
''' generate dataframe with unique cell lines'''
uniq_cell_data = X.drop_duplicates('cell line')

# materiall = df.loc[df['material'] == 2]
# Y = materiall.drop([ 'Unnamed: 0', 'viability (%)'], axis=1)
"""uniq value datasets"""
uniq = [] # stores all the unique characters available in the dataset, it helps to make a new population with random parameters
for a in range(len(X.columns)):
  uni = pd.unique(X.iloc[:, a])
  uniq.append(uni)
# uniq[0]

"""create individual with values that are picked from the uniq array above"""

def individuals():
  indv = []
  for a in range(len(X.columns)):
    uniqas = random.choice(uniq[a])
    indv.append(uniqas)
  return indv
# individuals()

"""generate population with specific population size"""
#population with specific material descriptors were generated but cell line were still random
def population(size):
  pops = []
  for indv in range(size):
    single = individuals() 
    pops.append(single)
  new = pd.DataFrame(data=pops, columns=X.columns)
  material_uniq = X.iloc[[random.randrange(0, len(X)) for _ in range(len(new))]]
  material_uniq = material_uniq.reset_index(drop=True)
  # print(new.columns)

  new[['material','electronegativity', 'Valance_electron', 'rox',
       'amw', 'mcd', 'chi0v', 'hallKierAlpha', 'lipinskiHBD',
       'NumRotatableBonds','chi2v', 'radii',
       'CrippenClogP', 'kappa1', 'chi1v']] =  material_uniq[['material','electronegativity', 'Valance_electron', 'rox',
       'amw', 'mcd', 'chi0v', 'hallKierAlpha', 'lipinskiHBD',
       'NumRotatableBonds','chi2v', 'radii',
       'CrippenClogP', 'kappa1', 'chi1v']]
  return new

#
dff = population(population_size)
# print(dff.columns)
# print(dff)


"""change cell type into cancer and normal cell line"""
# cell = pd.read_csv('Data/cell_line/cell_decode.csv')
# canc =cell.loc[cell['Cell type'] =='SKOV-3', 'oe'].values.tolist()
# norm = cell.loc[cell['Cell type'] =='CHO-K1', 'oe'].values.tolist()

"""selecting SKOV-3 as cancer cell line and  CHO-K1 as normal cell line (both of them are ovary cell line """
popn = []
popc = []
#ovary normal- CHOK1: 14, canc - SKOV3: 61
# lung normal- BEAS 2B:9, canc- H1299:18, normal:HFL-1: 26
# liver normal - L02: 39, canc - HepG2: 33
def cell_lines(dat):
  #making same normal and cancer dataframe
  # single_norm = uniq_cell_data.loc[df['cell line'] == norm[0]]
  # single_canc = uniq_cell_data.loc[df['cell line'] == canc[0]]
  single_norm = uniq_cell_data.loc[df['cell line'] == 9] # cho-k1 cell line
  single_canc = uniq_cell_data.loc[df['cell line'] == 33] # skov-3 cell line
  pop_norm =pd.concat([single_norm]*len(dat), ignore_index=True)
  pop_canc = pd.concat([single_canc] * len(dat), ignore_index=True)
  df_norm= dat.copy()
  df_canc = dat.copy()
  #replaced random data for cell descriptors with normal and cancer cell line
  df_norm[['cell line', 'cell type', 'organism', 'morphology', 'disease','tissue']] = pop_norm[['cell line', 'cell type', 'organism', 'morphology', 'disease','tissue']]
  df_canc[['cell line', 'cell type', 'organism', 'morphology', 'disease','tissue']] = pop_canc[['cell line', 'cell type', 'organism', 'morphology', 'disease','tissue']]
  # print('norm',df_norm.columns)
  return df_norm, df_canc


def fitness(df):
  norm, canc = cell_lines(df)
  # print('norm',norm.columns, canc)
  norm_viability = models.lgbm_predict(norm)
  canc_viability = models.lgbm_predict(canc)
  fitness = []
  norm_v = []
  canc_v = []
  for a in range(len(norm_viability)):
    n = norm_viability[a]
    c = canc_viability[a]
    fit = n - c
    fitnn = fit.tolist()
    norm_v.append(n)
    canc_v.append(c)
    fitness.append(fitnn)
  copy = norm.assign(norm_v=norm_v)
  copy2 = copy.assign(canc_v=canc_v)
  copy3 = copy2.assign(Fitness = fitness)
  copy3 = copy3.sort_values('Fitness', ascending=False)
  return copy3


# print(fitness(dff))

def result_evaluation(df):
  out = decoder.decode_transformed(df)
  return out