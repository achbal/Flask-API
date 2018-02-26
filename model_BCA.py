import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.cluster import KMeans
from module_func_pred import * # import module
from module_pred_bca import *
import sys





if __name__ == '__main__' :

    # ---- load file ---------------------------------------------------------------------------------------------
    df_easy = pd.read_table('LP_NL_BCA.csv', sep='\t')

    # -----------------------------------------------
    #
    #
    #       Matching columns names
    #
    #
    # -----------------------------------------------

# ---- Rename if necessary -----------------------------------------------------------------------------------
df_easy = df_easy.rename(columns={'class_rotation':'ROTATION'})
df_easy = df_easy.rename(columns={'FRE':'TOTAL_FREVO'})
try:
    df_easy = df_easy.rename(columns={'DATE_VENTE':'ORG_11_DATE_VENTE'})
    df_easy = df_easy.rename(columns={'ORG_17_DATE_MEC':'ORG_3_DATE_MEC'})
except:
    print("Couldn't rename")

# ---- Replace different string ------------------------------------------------------------------------------

df_easy.ORG_22_TOTAL_DAMAGE = df_easy.ORG_22_TOTAL_DAMAGE.replace('#n/a', 0.)


# ---- Error on the sale date --------------------------------------------------------------------------------

df_easy = df_easy[(df_easy.ORG_11_DATE_VENTE>'0000-00-00 00:00:00')]



# ---- if we have a file in input we can clean it -------------------------------------------------------------
df_easy = func_selection(df_easy)


# ---- if we haven't the variable MOIS in the file ------------------------------------------------------------
df_easy['MOIS'] = df_easy.ORG_3_DATE_MEC.dt.month

# ---- Creation of the BCA notation ---------------------------------------------------------------------------
df_easy = func_note_meca(df_easy, 'ORG_22_TOTAL_DAMAGE')

# ---- List parameters

list_ = ['KM', 'Age','VO_MARQUE_ID', 'VO_MODELE_ID',  'PORTE_CORRECTED','LITRE','COTE_VO',\
         'ROTATION', 'ORG_6_NOTE_MECA','PUISSANCE_CORRECTED','MOIS', 'Month_sale']

# ---- Clean data with empty trends ---------------------------------------------------------------------------
df_easy.dropna(subset=list_, inplace=True)

# ---- Kmeans and predicted group -----------------------------------------------------------------------------
kmeans_loaded_model = pickle.load(open('model_BCA_kmeans.sav', 'rb'))
df_easy.loc[:,'Group'] = kmeans_loaded_model.predict(df_easy[list_])

# ---- Predic BCA value ---------------------------------------------------------------------------------------

test = func_predict_set(df_easy, list_)


# ----- Resume volatility -------------------------------------------------------------------------------------
print('The volatility of the B2B cars is : {}%'.format(   func_volatility(test[(test.Age>84) & (test.KM>=100000)]['Predict'], test[(test.Age>84) & (test.KM>=100000)]['ORG_20_SALE_AMOUNT'].astype(float))))
print('The volatility of the B2C cars is : {}%'.format(  func_volatility(test[(test.Age<=84) & (test.KM<100000)]['Predict'], test[(test.Age<=84) & (test.KM<100000)]['ORG_20_SALE_AMOUNT'].astype(float))))

