import numpy as np
import pandas as pd
import sklearn
import os
from sklearn.cluster import KMeans
from module_func_pred import * # import module
from module_pred_bca import *
import sys
import json
from datetime import datetime 
import os.path
import argparse

#   List of parameters -------------|
#                                   |
#   COTE_CORRECTED                  |
#   KM                              |
#   VO_MARQUE_ID                    |
#   VO_MODELE_ID                    |
#   PORTE                           |
#   LITRE                           |
#   ROTATION                        |
#   PUISSANCE                       |
#   Date_MEC / ANNEE_CORRECTED-MOIS |
#   Date_vente / default = now()    |
#   NOTE_MECA / FRE                 |
#   TOTAL_FREVO                     |
#   --------------------------------|

#   COTE_VO --> COTE_CORRECTED
#   ANNEE_CORRECTED = ANNEE
#   DATE_VENTE = COTE_DATE

# Get the parameters 
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='''This code calculates the estimation in auctions price for one or more vehicle

 Example : \n \n python3 model_BCA_API.py  218000 84 256 2 9 2 2.0 4545 3 1500 150 2008 4 2018-02-23 \n \n Return 2316 ''', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("KM", type=int, help="Mileages")
parser.add_argument("VO_MARQUE_ID", type=int, help="ID of the brand of vehicle")
parser.add_argument("VO_MODELE_ID", type=int, help="ID model of the vehicle")
parser.add_argument("CARBURANT_ID", type=int, help="ID fuel of the vehicle")
parser.add_argument("CARROSSERIE_ID", type=int, help="ID body car of the vehicle")
parser.add_argument("PORTE_CORRECTED", type=float, help="Number of door")
parser.add_argument("LITRE", type=float, help="Liter")
parser.add_argument("COTE_VO", type=float, help="Quotation of the car")
parser.add_argument("ROTATION", type=float, help="Rotation of the vehicle")
parser.add_argument("TOTAL_FREVO", type=float, help="Statistical costs")
parser.add_argument("PUISSANCE_CORRECTED", type=int, help="Power of the car")
parser.add_argument("ANNEE", help="Year of the first registration")
parser.add_argument("MOIS", help="Month of the first registration")
parser.add_argument("ORG_11_DATE_VENTE",  help="Creation Date - format YYYY-MM-DD")
# type=lambda d: datetime.strptime(d, '%Y-%m-%d')

# ---- Transform in namespace 
args = vars(parser.parse_args())



#python3 model_BCA_API.py  48000 55 2723 5 3.0 28595 3 700 300 2011 2 2018-02-09

def func_prediction(df) :
    #print(args)    
    #index = args.keys()#['COTE_VO', 'VO_MODELE_ID', 'PORTE_CORRECTED', 'VO_MARQUE_ID','KM','ROTATION','ORG_11_DATE_VENTE', 'LITRE', 'MOIS','TOTAL_FREVO', 'PUISSANCE_CORRECTED','ANNEE']
    #df = pd.DataFrame(data=list(args.values()), index = index).T
    # ---- if we have a file in input we can clean it -------------------------------------------------------------
    #print(df)
    try :
        df['ORG_3_DATE_MEC'] = df['ANNEE']+'-'+df['MOIS']
        df['ORG_3_DATE_MEC'] = pd.to_datetime(df['ORG_3_DATE_MEC'], format='%Y-%m')
    except :
        print("\nCouldn't generate the Date_MEC\n")
    
    try:
        df['ORG_11_DATE_VENTE'] = pd.to_datetime(df['ORG_11_DATE_VENTE'])
    except:
        print("The selling date couldn't be generated")

    # ---- List of parameters
    list_ = ['KM', 'Age','VO_MARQUE_ID', 'VO_MODELE_ID',  'PORTE_CORRECTED','LITRE','COTE_VO',\
         'ROTATION', 'ORG_6_NOTE_MECA','PUISSANCE_CORRECTED','MOIS', 'Month_sale', 'CARBURANT_ID', 'CARROSSERIE_ID']



    # ---- Creation of the BCA notation ---------------------------------------------------------------------------
    df = func_note_meca(df, 'TOTAL_FREVO')

    # ---- Selection and clean data -------------------------------------------------------------------------------
    df = func_selection(df)
    # ---- Kmeans and predicted group -----------------------------------------------------------------------------
    kmeans_loaded_model = pickle.load(open('BCA_models/Models/model_BCA_kmeans.sav', 'rb'))
    df.loc[:,'Group'] = kmeans_loaded_model.predict(df[list_])

    results = func_predict_set(df, list_)

    #print('BCA_Cotation : {}'.format(results))
    return results


if __name__ == '__main__' :
    # ---- Generate DataFrame from input parameters of the parser -------------------------------------------------
    df = pd.DataFrame(data=list(args.values()), index=list(args.keys())).T
    # ---- Test if a file for the model
    #if  os.path.isfile('BCA_models') :
    #     os.system('mkdir BCA_models')
    #     os.system('mkdir BCA_models/Models')
    #     os.system('mkdir BCA_models/Log_files')
    #     print("Couldn't use the models - empty file")
    # ---- Get the execution time ------------------------------------------------------------
    now = datetime.now()
    #print(args)    
    predict = func_prediction(df)
    # ---- Creation of Json file ------------------------------------------------------------
    jsonData = json.dumps(predict.astype(str).to_dict())
    with open('BCA_models/Log_files/BCA_prediction_'+str(now)+'.json', 'w') as f:
        json_str = json.dump(jsonData,f)
    #print('\n')
    print(predict['COTE_BCA_PREDICT'].item())

