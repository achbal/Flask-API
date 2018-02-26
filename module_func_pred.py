
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.transforms as transforms
import seaborn
from statsmodels import robust
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge                  # import module
from sklearn import gaussian_process, feature_selection, svm, gaussian_process,tree
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, Lars, LassoLars, LassoCV, LassoLarsCV, RidgeCV, BayesianRidge, LogisticRegression, LogisticRegressionCV, ElasticNet, Lasso,PassiveAggressiveClassifier
import time
from sklearn.metrics import r2_score


#---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                                                                                   #
#                                                                                                   #
#----------------------------------------- Traitment data ------------------------------------------#
#                                                                                                   #
#                                                                                                   #
#                                                                                                   #
#---------------------------------------------------------------------------------------------------#




def func_selection(df_1):
    """
       Function to clean and select a dataframe 
        
    """
    
    # ---- Creation of the cotation type ----------------------------------------------------------------------------------
    try:
        df_1.loc[:,'TYPE_COTE'] = df_1.apply(lambda x: 'ref' if x['COTE_VO']!=x['COTE_MCCLBP'] else 'mcclbp', axis=1)
    except:
        pass #print("Couldn't compare different quotations")
    # ---- Replace letters by numbers -------------------------------------------------------------------------------------
    df_1.ORG_6_NOTE_MECA.replace('a', 1, inplace=True)
    df_1.ORG_6_NOTE_MECA.replace('b', 2, inplace=True)
    df_1.ORG_6_NOTE_MECA.replace('c', 3, inplace=True)
    df_1.ORG_6_NOTE_MECA.replace('d', 4, inplace=True)
    df_1.ORG_6_NOTE_MECA.replace('e', 5, inplace=True)
    # ---- replace the values containing '|' by nan values ----------------------------------------------------------------
    df_1 = df_1.applymap(lambda x: np.nan if '|' in str(x) else x )
    # ---- Delete nan values ----------------------------------------------------------------------------------------------
    df_1.dropna(subset=['ORG_3_DATE_MEC','ORG_11_DATE_VENTE'], inplace=True)
    # ---- Cast from string to datetime data ------------------------------------------------------------------------------
    df_1.loc[:,'ORG_3_DATE_MEC']=pd.to_datetime(df_1.ORG_3_DATE_MEC)
    df_1.loc[:,'ORG_11_DATE_VENTE']=pd.to_datetime(df_1.ORG_11_DATE_VENTE)
    # ---- Select all the lines where the sailing date is in the futur or equal to the first apparition -------------------
    df_1=df_1[df_1.ORG_11_DATE_VENTE>=df_1.ORG_3_DATE_MEC]

    # ---- Computation of the variable age (timedelta format in days) -----------------------------------------------------
    df_1.loc[:,'Age'] = df_1.ORG_11_DATE_VENTE-df_1.ORG_3_DATE_MEC
    # ---- Pass from timedelta to int - month data ------------------------------------------------------------------------
    df_1.Age = df_1.Age.apply(lambda x : int(round(x.days/30)))

    # ---- Select row where age and km are positives or equal to 0 --------------------------------------------------------
    df_1 = df_1[df_1.Age>=0]
    df_1 = df_1[df_1.KM>=0]
    # ---- Select rows where the COTE_VO is between 1500 and 35000 --------------------------------------------------------
    df_1 = df_1[(df_1.COTE_VO>4000) & (df_1.COTE_VO<35000)]
    #df_1 = df_1[df_1.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC>=1800]
    # ---- Select rows where the note of BCA are between 1 and 4 ----------------------------------------------------------
    df_1 = df_1[df_1.ORG_6_NOTE_MECA<5]
    # ---- Month sale -----------------------------------------------------------------------------------------------------
    #df_1.dropna(subset=['ORG_11_DATE_VENTE'], inplace=True)
    df_1.loc[:,'Month_sale'] = df_1.ORG_11_DATE_VENTE.dt.month
    #df_1.Month_sale = df_1.Month_sale.apply(lambda x : int(x[1]))
    #df_1.Month_sale = df_1.Month_sale.astype(int)
    # ---- Size of the training set ---------------------------------------------------------------------------------------
    #print('\nThe number of lines in the file is {}\n'.format(df_1.shape[0]))

    return df_1



# ---- func for the computation of the volatility between two data ----------------------------------------------------
def func_volatility(a, b):
    """
        
    """
    return round((100*robust.mad((a/b)/np.median(a/b))),2) # return a % value



def func_groupby(df_1, group):
    # ---- Groupby on the BCA note and the cotation type -------------------------------------------------------------
    # ---- Median Cote for each note and each type of cotation -------------------------------------------------------
    a_1 = df_1.groupby(group).apply(lambda x : int(round(x['COTE_VO'].median()) ))
    # ---- Deviation of the Median Cote for each note and each type of cotation --------------------------------------
    a_2 = df_1.groupby(group).apply(lambda x : int(round(x['COTE_VO'].mad()) ))
    a_2_1 = df_1.groupby(group).apply(lambda x : round(100*int(round(x['COTE_VO'].mad()) )/int(round(x['COTE_VO'].median()) ),2))
    # ---- Median Price for each note and each type of cotation ------------------------------------------------------
    a_3 = df_1.groupby(group).apply(lambda x : int(round(x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'].median())))
    # ---- Diation Median Price for each note and each type of cotation ----------------------------------------------
    a_4 = df_1.groupby(group).apply(lambda x : int(round(x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'].mad())))
    a_4_1 = df_1.groupby(group).apply(lambda x : round(100*int(round(x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'].mad()) )/int(round(x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'].median()) ),2))
    
    # ---- Diff Median Cote-Price in € for each note and each type of cotation ---------------------------------------
    a_7 = df_1.groupby(group).apply(lambda x : int(round(x['COTE_VO'].median()-x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'].median())))
    # ---- Diff Median Cote-Price in % for each note and each type of cotation ---------------------------------------
    a_8 = df_1.groupby(group).apply(lambda x : round(100*(x['COTE_VO'].median()-x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'].median())/x['COTE_VO'].median(),2))
    # ---- Volatility compute for each note and each type of cotation ------------------------------------------------
    a_6 = df_1.groupby(group).apply(lambda x : func_volatility(x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'], x['COTE_VO']))
    # ---- Number of vehicles for each note and each type of cotation ------------------------------------------------
    a_5 = df_1.groupby(group).apply(lambda x : int(round(x['ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC'].count())))
    # ---- Median KM for each note and each type of cotation ---------------------------------------------------------
    a_9 = df_1.groupby(group).apply(lambda x : int(round(x['KM'].median())))
    # ---- Median Age for each note and each type of cotation --------------------------------------------------------
    a_10= df_1.groupby(group).apply(lambda x : int(round(x['Age'].median())))
    # ---- concatenation on each subdataframe ------------------------------------------------------------------------
    a = pd.concat([a_1, a_2, a_2_1, a_3, a_4,a_4_1,a_7,a_8, a_6, a_9,a_10, a_5], axis=1)
    # ---- Rename columns --------------------------------------------------------------------------------------------
    a.columns = ['Median_COTE_Price_€', 'MAD_COTE_€', '%_MAD_COTE','Median_Price_TTC_€', 'MAD_Price_€','%_MAD_PRICE', 'Diff_Cote_Price_€',                                                                  'Diff_Cote_Price_%' ,'Volatility_%', 'Median_KM','Median_Age','NB']
    return a



def func_loop_error_car(df, list_, numb=25):
    """
        Functione to estimate the error of the prediction on a cluster random of 8 cars
        
    """
    start_time = time.time()

    df2_= pd.DataFrame()
    #df3_= pd.DataFrame()
    #df = X_train2[X_train2.Group==3].copy()
    
    params = {'n_estimators': 800, 'max_depth': 8, 'min_samples_split': 2,
        'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    df.dropna(subset=list_,inplace=True)
    y1 = df.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train2 = df[list_]
    #X_train, X_test, y_train, y_test = train_test_split(df[list_], y1)
    clf = clf.fit(X_train2, y1)
#
    df_ = pd.DataFrame()
    for _ in range(numb):
        
        X_test = df.sample(n=8)
        df_=df_.append(func_predict(clf, X_test[list_],X_test.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC)[0], \
                       ignore_index=True)
    df_.columns = ['NB', 'Volatility_%','Median_Price', 'Median_Predict', 'Median_Absolut_Error','Relative_Error_%',\
                   'Diff_medians','Score_%', 'RPI_pred', 'RPI_reel', 'NB_values_diff', 'Sum_Real', 'Sum_pred', 'Diff_sum']
        #df3_ = df3_.append(df_)
    elapsed_time = time.time() - start_time
    print('Elapsed time is: ', elapsed_time)
    #print(df_.sort_values(by='Relative_Error_%', ascending=False).head(10))

    df2_=df2_.append(pd.DataFrame([int(df_.NB.sum()),df_['Volatility_%'].median(), int(round(df_.Median_Price.mean())),int(round(df_.Median_Predict.mean())),int(round(df_.Median_Absolut_Error.mean())), round(df_['Relative_Error_%'].mean(),2),int(round(df_.Diff_medians.mean())), round(df_['Score_%'].mean(),2),round(df_.RPI_pred.mean(),2), round(df_.RPI_reel.mean(),2), int(df_['NB_values_diff'].sum()), df_['Sum_Real'].sum(), df_['Sum_pred'].sum(), df_['Diff_sum'].sum(),round(df_['Diff_sum'].sum()/int(df_.NB.sum()),1) ],index=['NB', 'Median_Volatily_%','Median_Price', 'Median_Predict', 'Median_Absolut_Error','Relative_Error_%' , 'Diff_medians', 'Score_%', 'RPI_pred','RPI_reel','NB_values_diff', 'Total_test', 'Total_pred', 'Total_diff', 'Error_Per_Car']).T)

    return df2_, df_


def func_loop_error_car2(df, list_, numb=1000):
    """
        Functione to estimate the error of the prediction on a cluster random of 8 cars
        
    """
    start_time = time.time()
    df2_= pd.DataFrame()
    params = {'n_estimators': 800, 'max_depth': 8, 'min_samples_split': 2,
        'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    df.dropna(subset=list_,inplace=True)
    y1 = df.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train2 = df[list_]
    clf = clf.fit(X_train2, y1)
    df_ = pd.DataFrame()
    for _ in range(numb):
    
        X_test = df.sample(n=1)
        df_=df_.append(func_predict(clf, X_test[list_],X_test.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC), \
                   ignore_index=True)
    df_.columns = ['NB', 'Volatility_%','Median_Price', 'Median_Predict', 'Median_Absolut_Error','Relative_Error_%',\
                                  'Diff_medians','Score_%', 'RPI_pred', 'RPI_reel', 'NB_values_diff', 'Sum_Real', 'Sum_pred', 'Diff_sum']

    elapsed_time = time.time() - start_time
    print('Elapsed time is: ', elapsed_time)
    
    df2_=df2_.append(pd.DataFrame([int(df_.NB.sum()),df_['Volatility_%'].median(), int(round(df_.Median_Price.mean())),int(round(df_.Median_Predict.mean())),int(round(df_.Median_Absolut_Error.mean())), round(df_['Relative_Error_%'].mean(),2),int(round(df_.Diff_medians.mean())), round(df_['Score_%'].mean(),2),round(df_.RPI_pred.mean(),2), round(df_.RPI_reel.mean(),2), int(df_['NB_values_diff'].sum()), df_['Sum_Real'].sum(), df_['Sum_pred'].sum(), df_['Diff_sum'].sum(),round(df_['Diff_sum'].sum()/int(df_.NB.sum()),1) ],index=['NB', 'Median_Volatily_%','Median_Price', 'Median_Predict', 'Median_Absolut_Error','Relative_Error_%' , 'Diff_medians', 'Score_%', 'RPI_pred','RPI_reel','NB_values_diff', 'Total_test', 'Total_pred', 'Total_diff', 'Error_Per_Car']).T)
                   
    return df2_, df_




def func_fit_models(df,name, list_):
    """
        
        """
    model_dict = {}
    start_time = time.time()
    params = {'n_estimators': 800, 'max_depth': 8, 'min_samples_split': 2,
        'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    df.dropna(subset=list_,inplace=True)
    for i in df.Group.unique():
        clf = clf.fit(df[df.Group==i][list_], df[df.Group==i][name])
        model_dict[str(i)] = clf
    
    
    elapsed_time = time.time() - start_time
    print('Elapsed time is: ', elapsed_time)
    return model_dict



def MakePlot(data,name,namefile=None):
    """
        function to create and save a plot of all the result:
        input :
        - data : Series with the data to plot
        - namefile : name (str) of the save figure
        - name : title of the graph
        output :
        - plot of the data with the boundaries 2-3 sigma
        - figure with png format if namefile is passed
    """
    
    fig, ax = plt.subplots(figsize=(15,8),facecolor='white')
    matplotlib.rcParams.update({'font.size': 22})
    
    ax.errorbar(data.index, data,None,None,fmt='bo',alpha=0.6)
    
    plt.title(name, fontsize= 25)
    plt.xlabel('Truck number', fontsize= 20)
    plt.ylabel( data.name, fontsize= 20)
    plt.grid(True)
    
    ax = plt.gca() # grab the current axis
    plt.axhline(data.median(),color='limegreen',linewidth=6)
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0,data.median(), "{:.2f}".format(data.median()), color="green", transform=trans,
            ha="center", va="center")
        
    plt.text(-1, data.median()+2*data.mad(), "{:.2f}".format(data.median()+2*data.mad()), \
                     color='steelblue')
    plt.text(-1, data.median()-2*data.mad(), "{:.2f}".format(data.median()-2*data.mad()), \
                     color='steelblue')
            
    plt.text(-1, data.median()+3*data.mad(), "{:.2f}".format(data.median()+3*data.mad()), \
                     color='plum')
    plt.text(-1, data.median()-3*data.mad(), "{:.2f}".format(data.median()-3*data.mad()), \
                     color='plum')
            
            
    plt.fill_between(data.index, data.median()-3*data.mad(), data.median()+3*data.mad(), \
                             alpha=0.1, edgecolor='red', facecolor='red',linewidth=4, linestyle='dashdot')
    plt.fill_between(data.index, data.median()-2*data.mad(), data.median()+2*data.mad(), \
                             alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=4, linestyle='dashdot')
    if namefile!=None:
        plt.savefig(namefile+'.png',bbox_inches='tight')

    plt.show()


#---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                                                                                   #
#                                                                                                   #
#--------------------------------------- Predict Models --------------------------------------------#
#                                                                                                   #
#                                                                                                   #
#                                                                                                   #
#---------------------------------------------------------------------------------------------------#


def func_predict(clf, X_test,y_test):
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != pred).sum()))
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    a = pd.Series([y_test.shape[0],vol2 ,int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((y_test-pred)))),\
                   round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),\
                   round(100*clf.score(X_test,y_test),2), \
                   round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2),\
                   (y_test - pred > int(round(np.median((y_test-pred))))).sum(), int(round(y_test.sum())), int(round(pred.sum())),int(round(y_test.sum()- pred.sum()))])
        
    return a, pred




def func_ridge(df_B2C_2,list_):
    """
       function to evaluate the data with the KernelRidgeRegression model
       input :
       - df_B2C_2 : initial dataframe
       - list_    : list of name variables used by the model
       output :
       - return the volatility of the prediction determine by the KernelRidgeRegression
    """
    clf = KernelRidge(alpha=1e-2, kernel='linear')
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)




def func_svr(df_B2C_2,list_):
    """
        function to evaluate the data with the SVR model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the SVR
    """
    clf = svm.SVR(kernel='rbf',gamma=0.1, C=100.)
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((y_test-pred)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_decision_trees(df_B2C_2,list_):
    """
        function to evaluate the data with the Decision Trees model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Decision Trees
    """
    clf = tree.DecisionTreeClassifier()
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((y_test-pred)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)




def func_random_forest(df_B2C_2,list_):
    """
        function to evaluate the data with the Decision Trees model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Decision Trees
        """
    clf = RandomForestClassifier(max_depth=15, random_state=6, n_jobs=20)
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((y_test-pred)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_gaussian_regression(df_B2C_2,list_):
    """
        function to evaluate the data with the Gaussian Process Regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Gaussian Process Regression
    """
    clf = gaussian_process.GaussianProcessRegressor()
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_lasso(df_B2C_2,list_):
    """
        function to evaluate the data with the Lasso model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Lasso
    """
    clf = Lasso(alpha=1e-6)                             # Linear Regression of the sklearn package
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_neural_network(df_B2C_2,list_):
    """
        function to evaluate the data with the Neural Network Regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Neural Network Regression
        """
    clf = MLPRegressor(solver='lbfgs')
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_tree_decision_regressor(df_B2C_2,list_):
    """
        function to evaluate the data with the Neural Network Regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Neural Network Regression
        """
    clf = tree.DecisionTreeRegressor()
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_isotonic_regressor(df_B2C_2,list_):
    """
        function to evaluate the data with the Tree Decision Regressor model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Tree Decision Regressor
        """
    clf = IsotonicRegression()
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_svr(df_B2C_2,list_):
    """
        function to evaluate the data with the SVR model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the SVR
        """
    clf = svm.SVR(kernel='rbf',degree=3, epsilon=0.000001, tol=1e-6, max_iter=1000000)
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_naive_bayes(df_B2C_2,list_):
    """
        function to evaluate the data with the Naive Bayes model with the kernel GaussianNB
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Naive Bayes Model
        """
    clf = GaussianNB()
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != pred).sum()))
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)





def func_gradient_boosting(df_B2C_2,list_):
    """
        function to evaluate the data with the gradient boosting model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Gradient Boosting Model
        """
    params = {'n_estimators': int(round(df_B2C_2.shape[0]*0.1)) if df_B2C_2.shape[0]*0.1>500 else 800, 'max_depth': 8, 'min_samples_split': 2,
        'learning_rate': 0.01, 'loss': 'lad'}
    clf = ensemble.GradientBoostingRegressor(**params)
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != pred).sum()))
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2), \
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_ElasticNet(df_B2C_2,list_):
    """
        function to evaluate the data with the ElasticNet (ridge+Lasso) model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the ElasticNet
        """
    clf = ElasticNet(random_state=0)
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_logistic_regression(df_B2C_2,list_):
    """
        function to evaluate the data with the logistic regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the logistic regression
        """
    clf = LogisticRegression()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_polynomial_regression(df_B2C_2,list_):
    """
        function to evaluate the data with the polynomial regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the polynomial regression
        """
    clf = Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression(fit_intercept=False))])
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_Ridge_complexity(df_B2C_2,list_):
    """
        function to evaluate the data with the ridge with cross validation model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the RidgeCV
        """
    clf = RidgeCV()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_Lasso_complexity(df_B2C_2,list_):
    """
        function to evaluate the data with the Lasso cross-validation  regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the lasso regression
        """
    clf = LassoCV()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_Lasso_Lars_complexity(df_B2C_2,list_):
    """
        function to evaluate the data with the polynomial regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the polynomial regression
        """
    clf = LassoLarsCV()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_Lars(df_B2C_2,list_):
    """
        function to evaluate the data with the Lars regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Lars regression
        """
    clf = Lars()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_LassoLars(df_B2C_2,list_):
    """
        function to evaluate the data with the LassoLars regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the LassoLars regression
        """
    clf = LassoLars()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_OMP(df_B2C_2,list_):
    """
        function to evaluate the data with the Orthogonal Matching Pursuit regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Orthogonal Matching Pursuit regression
        """
    clf = OrthogonalMatchingPursuit()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)



def func_BayesianRidge(df_B2C_2,list_):
    """
        function to evaluate the data with the Bayesian Ridge regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Bayesian Ridge regression
        """
    clf = BayesianRidge()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_LogisticRegressionCV(df_B2C_2,list_):
    """
        function to evaluate the data with the polynomial regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the polynomial regression
        """
    clf = LogisticRegressionCV()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_PassiveAggressiveClassifier(df_B2C_2,list_):
    """
        function to evaluate the data with the polynomial regression model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the polynomial regression
        """
    clf = PassiveAggressiveClassifier()
    #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    df_B2C_2.dropna(subset=list_,inplace=True)
    X_train2 = df_B2C_2[list_]
    y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    vol2 = func_volatility(pred,y_test)
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    return round(vol2,2), int(round(np.median(y_test))), int(round(np.median(pred))),\
        int(round(np.median((pred-y_test)))),round(100*np.median((pred-y_test)/y_test),2), \
        int(round(np.median(y_test)-np.median(pred))),round(100*clf.score(X_test,y_test),2),\
        round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2)


def func_gradient_boosting2(X_train,X_test,list_):
    """
        function to evaluate the data with the gradient boosting model
        input :
        - df_B2C_2 : initial dataframe
        - list_    : list of name variables used by the model
        output :
        - return the volatility of the prediction determine by the Gradient Boosting Model
        """
    params = {'n_estimators': 800, 'max_depth': 8, 'min_samples_split': 2,
        'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    X_train.dropna(subset=list_,inplace=True)

    y1 = X_train.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    #X_train = X_train[list_]
    y_test = X_test.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
    #X_train, X_test, y_train, y_test = train_test_split(X_train2, y1)
    clf = clf.fit(X_train[list_], y1)
    pred = clf.predict(X_test[list_])
    vol2 = func_volatility(pred,y_test)
    #print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != pred).sum()))
    #print('The volatity between the price and the core is : {}%'.format(float('%.2f'%(100*vol2))))
    a = pd.Series([y_test.shape[0],vol2 ,int(round(np.median(y_test))), int(round(np.median(pred))), int(round(np.median((y_test-pred)))),\
                   round(100*np.median((pred-y_test)/y_test),2), int(round(np.median(y_test)-np.median(pred))),\
                   round(100*clf.score(X_test[list_],y_test),2), \
                   round(100*np.sum(pred)/X_test.COTE_VO.sum(),2), round(100*np.sum(y_test)/X_test.COTE_VO.sum(),2),\
                   (y_test - pred > int(round(np.median((y_test-pred))))).sum(), int(round(y_test.sum())), int(round(pred.sum())),int(round(y_test.sum()- pred.sum()))])

    return a




#---------------------------------------------------------------------------------------------------#
#                                                                                                   #
#                                                                                                   #
#                                                                                                   #
#------------------------------------------ Test Models --------------------------------------------#
#                                                                                                   #
#                                                                                                   #
#                                                                                                   #
#---------------------------------------------------------------------------------------------------#


def func_compa_model(df_,list_, fraction=1):
    """
        function to compare the volatility of each prediction model
        
        """
    df_ = df_.sample(frac=fraction)
    #a = func_ridge(df_,list_)
    #b = func_lasso(df_,list_)
    #c = func_decision_trees(df_,list_)
    #d = func_random_forest(df_,list_)
    #e = func_neural_network(df_,list_)
    #f = func_tree_decision_regressor(df_,list_)
    #g = func_naive_bayes(df_,list_)
    h = func_gradient_boosting(df_,list_)
    #i = func_ElasticNet(df_,list_)
    #j = func_logistic_regression(df_,list_)
    #k = func_svr(df_,list_)
    #l = func_polynomial_regression(df_,list_)
    #m = func_Ridge_complexity(df_,list_)
    #n = func_BayesianRidge(df_,list_)
    #o = func_Lasso_complexity(df_,list_)
    #p = func_Lars(df_,list_)
    #q = func_LassoLars(df_,list_)
    #r = func_Lasso_Lars_complexity(df_,list_)
    #s = func_OMP(df_,list_)
    #t = func_LogisticRegressionCV(df_,list_)
    
    
    data = [h]#,t, j,k,f,g, m,o,q,r,c,d,a,b,e,i,l,n,p,s
    df = pd.DataFrame(data=data, index=['Gradient_Boosting']) #,'Logistic_RegressionCV', 'SVR','Logistic_Regression', 'Ridge', 'Lasso', 'Neural_Network', 'LassoLarsCV','LassoCV','Random_Forest',,'ElasticNet',  'Polynomial_Regression',  'BayesianRidge','Lars','OMP', 'LassoLars','Decision_Tree_Regressor','Naive_Bayes','Decision_Trees','RidgeCV'
    df.columns = ['Volatility_%', 'Median_Price', 'Median_Predict', 'Median_Absolut_Error','Relative_Error_%' , 'Diff_medians', 'Score_model_%', 'RPI_pred', 'RPI_reel']
    df=df.sort_values(by='Volatility_%')
    return df


def func_compa_model2(df_,list_, fraction=1):
    """
        function to compare the volatility of each prediction model
        
        """
    df_ = df_.sample(frac=fraction)
    a = func_ridge(df_,list_)
    b = func_lasso(df_,list_)
    c = func_decision_trees(df_,list_)
    d = func_random_forest(df_,list_)
    e = func_neural_network(df_,list_)
    f = func_tree_decision_regressor(df_,list_)
    g = func_naive_bayes(df_,list_)
    h = func_gradient_boosting(df_,list_)
    i = func_ElasticNet(df_,list_)
    
    #k = func_svr(df_,list_)
    #l = func_polynomial_regression(df_,list_)
    #m = func_Ridge_complexity(df_,list_)
    #n = func_BayesianRidge(df_,list_)
    o = func_Lasso_complexity(df_,list_)
    p = func_Lars(df_,list_)
    #q = func_LassoLars(df_,list_)
    #r = func_Lasso_Lars_complexity(df_,list_)
    #s = func_OMP(df_,list_)
    
    
    # ---- Feature selection ----------------------------
    #j = func_logistic_regression(df_,list_)
    #t = func_LogisticRegressionCV(df_,list_)
    
    
    
    data = [a,b,c,d,e,f,g,h,i,p]#,t, j,k,f,g, m,o,q,r,c,d,a,b,e,i,l,n,p,s
    df = pd.DataFrame(data=data, index=['Ridge', 'Lasso','Decision_Trees','Random_Forest','Neural_Network', 'Decision_Tree_Regressor','Naive_Bayes','Gradient_Boosting','ElasticNet','Lars']) #,'Logistic_RegressionCV', 'SVR','Logistic_Regression',   'LassoLarsCV','LassoCV',,  'Polynomial_Regression',  'BayesianRidge','OMP', 'LassoLars','RidgeCV'
    df.columns = ['Volatility_%', 'Median_Price', 'Median_Predict', 'Median_Absolut_Error','Relative_Error_%' , 'Diff_medians', 'Score_model_%', 'RPI_pred', 'RPI_reel']
    df=df.sort_values(by='Volatility_%')
    return df






    #---------------------------------------------------------------------------------------------------#
    #                                                                                                   #
    #                                                                                                   #
    #                                                                                                   #
    #------------------------------------- Feature selection -------------------------------------------#
    #                                                                                                   #
    #                                                                                                   #
    #                                                                                                   #
    #---------------------------------------------------------------------------------------------------#


def func_feature_selection(list_, *args):
    """
       function to determine the feature importance estimated by the Tree decision model.
       input :
           - list_ : list of parameters
           - args  : list of dataframe
       output :
           - df_   : dataframe with the feature importance - % of importance of the parameters to predict the result
    """
    
    df_ = []
    for i in args:
        df_B2C_2 = i
        print('The size of the sample is : {} lines'.format(df_B2C_2.shape[0]))
        model = ExtraTreesClassifier()
        #df_2 = df_B2C_2[(df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC<10000) & \
        #(df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC>1000)]
        df_B2C_2.dropna(subset=list_,inplace=True)
        X_train2 = df_B2C_2[list_]
        y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
        model.fit(X_train2, y1)
        d = {i[0] : float('%.2f'%(100*i[1])) for i in list(zip(list_,model.feature_importances_ ))}
        df_B2C_2 = pd.DataFrame.from_dict(d, orient='index')
        #df_B2C_2.columns = ['Impact_Price_%']
        df_.append(df_B2C_2)
    df_ = pd.concat(df_, axis=1)
    #df_.sort_values(by = 'Impact_Price_%', ascending=False, inplace=True)

    return df_



def func_univariate_selection(list_, *args):
    """
        function to determine the feature importance estimated by the Tree decision model.
        input :
            - list_ : list of parameters
            - args  : list of dataframe
        output :
            - df_   : dataframe with the feature importance - % of importance of the parameters to predict the result
    """
    
    df_ = []
    for i in args:
        df_B2C_2 = i
        print('The size of the sample is : {} lines'.format(df_B2C_2.shape[0]))
        model = SelectKBest(score_func=chi2, k=4)
        #df_2 = df_B2C_2[(df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC<10000) & \
        #(df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC>1000)]
        df_B2C_2.dropna(subset=list_,inplace=True)
        X_train2 = df_B2C_2[list_]
        y1 = df_B2C_2.ORG_7_PRIX_DE_VENTE_AVEC_FEES_TTC
        model.fit(X_train2, y1)
        d = {i[0] : float('%.2f'%(100*i[1])) for i in list(zip(list_,model.scores_ ))}
        df_B2C_2 = pd.DataFrame.from_dict(d, orient='index')
        #df_B2C_2.columns = ['Impact_Price_%']
        df_.append(df_B2C_2)
    df_ = pd.concat(df_, axis=1)
    #df_.sort_values(by = 'Impact_Price_%', ascending=False, inplace=True)

    return df_

