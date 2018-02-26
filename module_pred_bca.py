
import pandas as pd
import pickle
import sklearn




def func_predict_set(df, list_):
    """
        
        """
    df_fin = pd.DataFrame()
    
    
    for i in df.Group.unique():
        clf = pickle.load(open('BCA_models/Models/model_BCA_group_'+str(i)+'.sav', 'rb'))
        df_ = df[df.Group==i]
        df_.loc[:,'COTE_BCA_PREDICT'] = clf.predict(df_[list_])
        df_fin = df_fin.append(df_)
    #df.loc[:,'Predict'] = list_results
    df_fin.COTE_BCA_PREDICT = df_fin.COTE_BCA_PREDICT.apply(lambda x : int(round(x)))
    return df_fin


def func_note_meca(df_easy, name):
    df_easy.loc[:,'Ratio'] = 100*df_easy[name].astype(float)/df_easy['COTE_VO'].astype(int)
    #df_easy[name] = df_easy[name].astype(float)
    q1 = df_easy[df_easy['Ratio']<=5]
    q2 = df_easy[(df_easy['Ratio']>5) & (df_easy['Ratio']<=10)]
    #q3 = test2[(test2['TOTAL_FREVO']>240) & (test2['TOTAL_FREVO']<=600)]
    q4 = df_easy[(df_easy['Ratio']>10) & (df_easy['Ratio']<=20)]
    q5 = df_easy[df_easy['Ratio']>20]
    
    q1['ORG_6_NOTE_MECA'] = 1
    q2['ORG_6_NOTE_MECA'] = 2
    q4['ORG_6_NOTE_MECA'] = 3
    q5['ORG_6_NOTE_MECA'] = 4
    
    test2 = pd.concat([q1,q2,q4,q5], axis=0)
    return test2



