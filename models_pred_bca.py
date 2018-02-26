import pandas as pd
from sklearn.cluster import KMeans
from module_func_pred import * # import module
from sklearn import ensemble
import pickle
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
import os



def func_create_models(df, name, list_):
    """
        Function to generate ML models and save them with pickle :
        input :
            - df    : dataframe initial data
            - name  : name of the column to predict
            - list_ : list of features
        output :
            - kmeans model / gradient boosting models for each kmeans group
                    -> automaticaly saved in .sav file
    """
    
    for i in df.Group.unique():     #    loop on kmeans group
        # ---- Parameters for the model -----------------------------------------------------------------------------------
        params = {'n_estimators': int(round(df.shape[0]*0.1)) if df.shape[0]*0.1>500 else 800, \
                            'max_depth': 8, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'lad'}
        # ---- Generation of the model ------------------------------------------------------------------------------------
        clf = ensemble.GradientBoostingRegressor(**params)
        # ---- Sub dataframe - selection of each data in the kmeans group -------------------------------------------------
        clf = clf.fit(df[df.Group==i][list_], df[df.Group==i][name])
        # ---- Name file generation ---------------------------------------------------------------------------------------
        filename = 'BCA_models/Models/model_BCA_group_'+str(i)+'.sav'
        # ---- Save the model in sav binary file --------------------------------------------------------------------------
        pickle.dump(clf, open(filename, 'wb'))
        # ---- Print on the screen the name of the group ------------------------------------------------------------------
        print(filename)


# try :

#  python3 models_pred_bca.py path_file colonne_to_fit

# python3 models_pred_bca.py ../../BCA_Data_SOL.csv ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC

if __name__ == '__main__' :
    # ---- Import the file containing BCA data ----------------------------------------------------------------------------
    #df_1 = pd.read_table('../../BCA_Data_SOL.csv', sep='\t', encoding='ISO-8859-1')
    df_1 = pd.read_table(sys.argv[1], sep='\t', encoding='ISO-8859-1')

    # ----------------------------------------------------------------------------------- #
    #           Faire tests sur le seperateur ainsi que sur l'encoding                    #
    # ----------------------------------------------------------------------------------- #
    

    # ---- Clean and select data ------------------------------------------------------------------------------------------
    df_1 = func_selection(df_1)
    
    # ---- List of features needed for the model --------------------------------------------------------------------------
    list_ = ['KM', 'Age','VO_MARQUE_ID', 'VO_MODELE_ID',  'PORTE_CORRECTED','LITRE','COTE_VO',\
         'ROTATION', 'ORG_6_NOTE_MECA','PUISSANCE_CORRECTED','MOIS', 'Month_sale', 'CARBURANT_ID', 'CARROSSERIE_ID']
         
    # ---- Delete row where features are empty ----------------------------------------------------------------------------
    df_1.dropna(subset=list_,inplace=True)

    # ---- Estimation cluster automatic
    
    bandwidth_X = estimate_bandwidth(df_1[list_], quantile=0.1, n_samples=df_1.shape[0])
    meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
    meanshift_model.fit(df_1[list_])
    labels = meanshift_model.labels_
    
    print('\nNumber of cluster in the data : {}\n'.format(len(np.unique(labels))))
    # ---- Create automatic clusters --------------------------------------------------------------------------------------
    kmeans = KMeans(init='k-means++',n_clusters=len(np.unique(labels)), random_state=0).fit(df_1[list_])
    # ---- Predict groups for the new data --------------------------------------------------------------------------------
    try :
       os.system('mkdir BCA_models')
       os.system('mkdir BCA_models/Models')
       os.system('mkdir BCA_models/Log_files')
    except:
       print("Files already exists")

    df_1.loc[:,'Group'] = kmeans.predict(df_1[list_])
    # ---- Generate the kmeans model and
    filename = 'BCA_models/Models/model_BCA_kmeans.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    print(filename)
    func_create_models(df_1, sys.argv[2], list_)
    print('\n')


