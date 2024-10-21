

#####################################################################################
# Python Script to build Machine Learning Models from CSV data containing smiles    #
# Written by Ruel Cedeno, Ph.D.                                                     #
#####################################################################################



######################
# Import libraries   #
######################

#Data Processing
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pickle
from joblib import dump, load
from mapply.mapply import mapply 
from copy import deepcopy
import datetime
from joblib import parallel_backend
import re
from collections import defaultdict

#Rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, PandasTools
from rdkit import DataStructs
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors, NHOHCount,NOCount
from rdkit.SimDivFilters import rdSimDivPickers
from collections import defaultdict
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

#Sklearn / ML
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, KFold, GridSearchCV, RepeatedKFold,  RepeatedStratifiedKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.model_selection import BaseCrossValidator
from sklearn.feature_selection import VarianceThreshold
from  sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, VotingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import cohen_kappa_score,accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix, matthews_corrcoef
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import datasets, decomposition
from sklearn.manifold import TSNE
import xgboost as xg
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
import plotly.express as px
from sklearn.svm import SVR
from time import process_time
from sklearn.model_selection import KFold
from custom_classes import RemoveZeroVarianceFeatures, RemoveAutocorrelatedFeatures






def canonicalize_smi(smi):
    try:
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi,sanitize=False))
    except TypeError:
        canon_smi = "CCCCC"
    return canon_smi


def get_canon_smi(smi_series):
    return list(mapply(smi_series, lambda x: canonicalize_smi(x), progressbar=False))

def get_BM(smi_series):
    return list(mapply(smi_series, lambda x: Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmilesFromSmiles(x,includeChirality=True), progressbar=False))


def stratified_train_test_split(X, y, test_size=0.2, random_state=None,n_bins=10):
    # Bin the target variable
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')

    # Perform stratified split using the binned target 
    #y  = np.array(y) if len(np.unique(y)) == 2 else np.array(y) >= np.median(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y_binned, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def stratified_train_test_split_idx(canon_series, y, test_size=0.2, random_state=None,n_bins=10):
    # Bin the target variable
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    # Generate an array of indices corresponding to the dataset
    indices = np.arange(len(canon_series))
    
    # Perform stratified split using the binned target variable and indices
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        stratify=y_binned, 
        random_state=random_state
    )
    
    train_test_index = {}
    train_test_index["train"] = list(train_indices)
    train_test_index["test"] = list(test_indices)

    return train_test_index



def scaffold_based_train_test_split_idx(canon_series, y,test_size=0.2, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    df = pd.DataFrame()
    df["smiles"] =canon_series.values
    df["scaff_id"] = get_BM(canon_series)
    # Extract the required columns
    X = df.index.values

    y  = np.array(y) if len(np.unique(y)) == 2 else np.array(y) >= np.median(y)
    #y = y if np.unique(y) == 2 else np.array(y) >= np.median(y)    
    groups = df['scaff_id'].values
    n_splits = 5
    
    # Initialize RepeatedGroupShuffleSplit
    rgss = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,random_state=random_state)
    
    # Find the best split that maintains the ratio of "is_positive"
    best_train_idx, best_test_idx = None, None
    best_diff = float('inf')
    
    for train_idx, test_idx in rgss.split(X, y, groups):
        y_train, y_test = y[train_idx], y[test_idx]
        train_pos_ratio = y_train.mean()
        test_pos_ratio = y_test.mean()
        ratio_diff = abs(train_pos_ratio - test_pos_ratio)
        
        if ratio_diff < best_diff:
            best_diff = ratio_diff
            best_train_idx, best_test_idx = train_idx, test_idx
    
    return {'train': best_train_idx.tolist(), 'test': best_test_idx.tolist()}


class RepeatedScaffoldBasedCV(BaseCrossValidator):
    def __init__(self, n_splits, n_repeats, smiles_series, y, test_size, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.smiles_series = smiles_series
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        np.random.seed(self.random_state)
        for repeat in range(self.n_repeats):
            for split in range(self.n_splits):
                random_state = np.random.randint(0, 10000)
                split_dict = scaffold_based_train_test_split_idx(
                    self.smiles_series, self.y, self.test_size, random_state
                )
                train_idx = split_dict['train']
                test_idx = split_dict['test']
                #print("test_idx",test_idx)
                #print("train_idx",train_idx)
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats


class RepeatedStratifiedBasedCV(BaseCrossValidator):
    def __init__(self, n_splits, n_repeats, smiles_series, y, test_size, random_state=42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.smiles_series = smiles_series
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        np.random.seed(self.random_state)
        for repeat in range(self.n_repeats):
            for split in range(self.n_splits):
                random_state = np.random.randint(0, 10000)
                split_dict = stratified_train_test_split_idx(
                    self.smiles_series, self.y, self.test_size, random_state
                )
                train_idx = split_dict['train']
                test_idx = split_dict['test']
                #print("test_idx",test_idx)
                #print("train_idx",train_idx)
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats



def get_rdkit_desc(mol):

    rdkit_desc = np.array([descriptor_function(mol) for descriptor_name,descriptor_function in Descriptors.descList])
    rdkit_desc_clean = np.clip(np.nan_to_num(np.array(rdkit_desc).astype(float), copy=False, nan=0),a_min=-1e5, a_max=1e5)
    return rdkit_desc_clean


def spearman_corr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred, rowvar=False)[0, 1]


nbits=1024
fpdict = {}
#fpdict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits, useChirality=True)


invgen = AllChem.GetMorganFeatureAtomInvGen()
fpdict['ecfp4_np'] = lambda m: rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=nbits,includeChirality=True).GetFingerprintAsNumPy(m)
fpdict['fcfp4_np'] = lambda m: rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=nbits,includeChirality=True,atomInvariantsGenerator=invgen).GetFingerprintAsNumPy(m)
fpdict['ecfp4_bit'] = lambda m: rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=nbits,includeChirality=True).GetFingerprint(m)
fpdict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpdict['rdkit'] = lambda m: get_rdkit_desc(m)
#fpdict['ecfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3,nBits=nbits, useChirality=True)

def CalculateFP(smiles,fp_name='ecfp4'):
    try:
        #print("smiles",smiles)
        m = Chem.MolFromSmiles(smiles)
        return fpdict[fp_name](m)
    except:
        print(smiles,"INVALID !")
        return fpdict[fp_name](Chem.MolFromSmiles('CCCC'))



def get_desc(smi_series, desc_name='ecfp4_bit'):
    if desc_name == 'ecfp4_bit':  # return bits instead of numpy array
        with parallel_backend('multiprocessing'):
            return list(mapply(smi_series, lambda x: CalculateFP(x, 'ecfp4_bit'), progressbar=True))
    elif desc_name == 'ECFP4':
        with parallel_backend('multiprocessing'):
            return np.array(list(mapply(smi_series, lambda x: CalculateFP(x, 'ecfp4_np'), progressbar=True)))
    elif desc_name == 'RDK5':
        with parallel_backend('multiprocessing'):
            return list(mapply(smi_series, lambda x: CalculateFP(x, 'rdk5'), progressbar=True))
    elif desc_name == 'FCFP4':
        with parallel_backend('multiprocessing'):
            return list(mapply(smi_series, lambda x: CalculateFP(x, 'fcfp4_np'), progressbar=True))
    elif desc_name == 'RDKIT':
        return list(mapply(smi_series, lambda x: CalculateFP(x, 'rdkit'), progressbar=True))
        #return list(smi_series.apply(lambda x: CalculateFP(Chem.MolFromSmiles(x), 'rdkit') ))        

    else:
        raise ValueError(f"Descriptor name '{desc_name}' is not recognized.")





import warnings
warnings.filterwarnings('ignore')

rand_state = 42
from sklearn.impute import SimpleImputer



##################################################################
####  CLASSIFICATION
###################################################################

from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def get_confusion_matrix(clf,X,y):
    y_pred = clf.predict(X) 
    return confusion_matrix(y, y_pred).ravel()


##################################################################
####  CLASSIFICATION
###################################################################

from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def get_confusion_matrix(clf,X,y):
    y_pred = clf.predict(X) 
    return confusion_matrix(y, y_pred).ravel()


class QSAR_Classifier:
    X,y,X_train,X_test,y_train,y_test = [],[],[],[],[],[]
    random_state = 42
    model_performance = pd.DataFrame()
    df_cv = pd.DataFrame()
    best_model = lambda: None
    use_default_mordred_desc = False
    

    def __init__(self,X,y,canon_series,test_size=0.2,CV_split=5,train_test_index=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.test_size = test_size
        self.canon_series = canon_series
        self.CV_split = CV_split
        self.train_test_index = train_test_index
        if train_test_index is None:
            self.X_train, self.X_test, self.y_train, self.y_test = stratified_train_test_split(X, y, test_size=self.test_size, n_bins=10, random_state=42)
        else:
            self.X_train = self.X[train_test_index["train"]]
            self.X_test = self.X[train_test_index["test"]]
            self.y_train = self.y[train_test_index["train"]]
            self.y_test = self.y[train_test_index["test"]]

    

    def compare_model_cv(self,model_obj_list,model_name_list,scaffold_split=True):
        df = pd.DataFrame()
        df_cv_list = []
        n_fold = 5
        n_repeats = int(self.CV_split/n_fold)
        scoring = {'ACC':'accuracy','AUC':'roc_auc','MCC':'matthews_corrcoef','MAP':'average_precision'}
        for obj,name in zip(model_obj_list,model_name_list):
            model_opt = obj
            #folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
            if not scaffold_split:
                folds = RepeatedStratifiedBasedCV(n_fold, n_repeats, self.canon_series.iloc[self.train_test_index["train"]], self.y_train, self.test_size, self.random_state)
            else:
                folds =  RepeatedScaffoldBasedCV(n_fold, n_repeats, self.canon_series.iloc[self.train_test_index["train"]], self.y_train, self.test_size, self.random_state)
            #folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
            scores = cross_validate(model_opt, self.X_train, self.y_train, scoring=scoring, cv=folds,return_train_score=False)
            res_mean = pd.DataFrame(scores).transpose().mean(axis=1)
            res_std = pd.DataFrame(scores).transpose().std(axis=1) # 5 = number of cv folds
            metric_name = list(res_mean.keys()) 


            model_opt_trained = deepcopy(model_opt.fit(self.X_train,self.y_train))
            df['metric'] = metric_name  + ['Test_ACC','Test_AUC', 'Test_MCC','Test(tn, fp, fn, tp)'] + [f"{j}_std" for j in metric_name ]+['model_obj'] 
            
            df_cv_i = pd.DataFrame(scores)            
            df_cv_i["algo"] = name
            df_cv_list.append(df_cv_i)

            
            
            
            print(f"model={name}")
            print(scores)
            y_test_pred = model_opt_trained.predict(self.X_test)
            test_acc,test_auc,test_mcc = [np.round(metric_func(self.y_test,y_test_pred),2) for metric_func in [accuracy_score,roc_auc_score,matthews_corrcoef]] 
            test_conf = confusion_matrix(self.y_test,y_test_pred).ravel()
            df[name] = list(np.round(res_mean.values,2)) + [test_acc,test_auc,test_mcc] +[(test_conf)]  +  list(res_std.values) +[deepcopy(model_opt)] 
            
        self.model_performance = df
        #self.best_model = self.model_performance[df.columns[1]].iloc[-1] #take the first column and the last row             
        
        self.df_cv = pd.concat(df_cv_list,ignore_index=True)
        
        return self.model_performance.set_index("metric"), self.df_cv








def create_pipeline(clf):
    pipeline = Pipeline([
        ('remove_zero_variance', VarianceThreshold(threshold=0.05)),
        ('remove_autocorrelated', RemoveAutocorrelatedFeatures(threshold=0.95)),
        ('scaler', StandardScaler()),
        ('estimator', clf)
    ])
    return pipeline







xgb_c = create_pipeline(xg.XGBClassifier(n_estimators=500,verbosity = 0, device='cpu'))

lgbm_c = create_pipeline(LGBMClassifier(n_estimators=500,objective='binary',verbose=-1,device = "cpu"))


rf_c = create_pipeline(RandomForestClassifier(n_estimators=500)) 
et_c = create_pipeline(ExtraTreesClassifier(n_estimators=500)) 



voting_c = VotingClassifier(estimators=[
                                       ('LGBM', lgbm_c),
                                       ('XGB', xgb_c),
                                       ('RF',rf_c),
                                       ('ET',et_c)], voting='soft', n_jobs=-1)

classifier_dict = {}
classifier_dict["LightGBM"] = lgbm_c
classifier_dict["XGBoost"] = xgb_c
classifier_dict["RandomForest"] = rf_c
classifier_dict["ExtraTrees"] = et_c
classifier_dict["Consensus"] = voting_c



##################################################################
#### REGRESSION
###################################################################



def pearson_r_scorer(clf, X, y):
    y_pred = clf.predict(X)
    return pearsonr(y,y_pred)[0]


def RMSE_scorer(clf, X, y):
    y_pred = clf.predict(X)
    return (mean_squared_error(y,y_pred))**0.5

def MAE_scorer(clf, X, y):
    y_pred = clf.predict(X)
    return (mean_absolute_error(y,y_pred))



class QSAR_Regressor:
    X,y,X_train,X_test,y_train,y_test = [],[],[],[],[],[]
    test_size = 0.15
    random_state = rand_state
    model_performance = pd.DataFrame()
    df_cv = pd.DataFrame()
    best_model = lambda: None
    use_default_mordred_desc = False
    

    def __init__(self,X,y,canon_series,test_size=0.15,CV_split=5,train_test_index=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.test_size = test_size
        self.canon_series = canon_series
        self.CV_split = CV_split
        self.train_test_index=train_test_index
        if train_test_index is None:
            self.X_train, self.X_test, self.y_train, self.y_test = stratified_train_test_split(X, y, test_size=self.test_size, n_bins=10, random_state=42)
        else:
            self.X_train = self.X[train_test_index["train"]]
            self.X_test = self.X[train_test_index["test"]]
            self.y_train = self.y[train_test_index["train"]]
            self.y_test = self.y[train_test_index["test"]]
        
        """
        else:
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = scaffold_based_train_test_split(self.canon_series,X, y, test_size=self.test_size, random_state=42)
            except:  #When scaffold split fails (e.g. all have the same scaffold, revert to random stratified split)
                self.X_train, self.X_test, self.y_train, self.y_test = stratified_train_test_split(X, y, test_size=self.test_size, n_bins=10, random_state=42)
        """

    

    def compare_model_cv(self,model_obj_list,model_name_list,criteria='roc_auc',retrain_to_all=True,scaffold_split=True):
        df = pd.DataFrame()
        df_cv_list = []
        
        scoring = {"pearsonR":pearson_r_scorer, "RMSE": RMSE_scorer, 'MAE':MAE_scorer}
        n_fold = 5
        n_repeats = int(self.CV_split/n_fold)
        for obj,name in zip(model_obj_list,model_name_list):
            model_opt = obj
            #folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)


            if not scaffold_split:
                folds = RepeatedStratifiedBasedCV(n_fold, n_repeats, self.canon_series.iloc[self.train_test_index["train"]], self.y_train, self.test_size, self.random_state)

            else:
                folds =  RepeatedScaffoldBasedCV(n_fold, n_repeats, self.canon_series.iloc[self.train_test_index["train"]], self.y_train, self.test_size, self.random_state)


            scores = cross_validate(model_opt, self.X_train, self.y_train, scoring=scoring, cv=folds,return_train_score=False)
            res_mean = pd.DataFrame(scores).transpose().mean(axis=1)
            res_std = pd.DataFrame(scores).transpose().std(axis=1) # 5 = number of cv folds
            metric_name = list(res_mean.keys()) 

            model_opt_trained = deepcopy(model_opt.fit(self.X_train,self.y_train))
            df['metric'] = metric_name  + ['Test_pearsonR','Test_MAE','Test_RMSE'] + [f"{j}_std" for j in metric_name ]+['model_obj']


            df_cv_i = pd.DataFrame(scores)            
            df_cv_i["algo"] = name
            df_cv_list.append(df_cv_i)

            y_test_pred = model_opt_trained.predict(self.X_test)
            test_pearson_r,test_mae,test_rmse,  = [np.round(metric_func(self.y_test,y_test_pred),2) for metric_func in 
                                                         [lambda y,y_pred: pearsonr(y,y_pred)[0],mean_absolute_error, lambda y,y_pred: mean_squared_error(y,y_pred)**0.5 ]] 
            df[name] = list(np.round(res_mean.values,2)) + [test_pearson_r,test_mae,test_rmse]  +  list(res_std.values) +[deepcopy(model_opt)]

        self.model_performance = df
        self.df_cv = pd.concat(df_cv_list,ignore_index=True)
        
        return self.model_performance.set_index("metric"), self.df_cv






#callbacks = [lightgbm.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)]

rand_state = 42
xgb_r = create_pipeline(xg.XGBRegressor(random_state=rand_state))

lgbm_r = create_pipeline(LGBMRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8,subsample_freq=1,random_state=rand_state))
svr_r = make_pipeline(preprocessing.StandardScaler(), SVR())

rf_r = create_pipeline(RandomForestRegressor(n_estimators=500,oob_score=True,random_state=rand_state))  

et_r = create_pipeline(ExtraTreesRegressor(n_estimators=500)) 

voting = VotingRegressor(estimators=[
                                       ('LGBM', lgbm_r),
                                       ('XGB', xgb_r),
                                       ('RF',rf_r),
                                       ('ET',et_r)], n_jobs=-1)

regressor_dict = {}
regressor_dict["LightGBM"] = lgbm_r
regressor_dict["XGBoost"] = xgb_r
regressor_dict["RandomForest"] = rf_r
regressor_dict['ExtraTrees'] = et_r
regressor_dict["Consensus"] = voting







def benchmark_model(smi_series,y,desc_list=['ECFP4'],model_name=["LightGBM"],task='Binary Classification',test_size=0.15,CV_split=5,split_type='Random Split'):
    res_df = pd.DataFrame()
    cv_df = pd.DataFrame()
    y = list(y)
    
    if split_type =='Random Split':
        scaffold_split = False
        train_test_index = stratified_train_test_split_idx(smi_series, y, test_size=0.2, random_state=42)        
    elif split_type == 'Scaffold Split':
        scaffold_split = True
        train_test_index = scaffold_based_train_test_split_idx(smi_series,y, test_size=0.2, random_state=42)
    
    data_obj_dict = {}

    if task == "Binary Classification":

        
        for desc in desc_list:
            X = get_desc(smi_series,desc)    

            model = QSAR_Classifier(X,y,smi_series,test_size=test_size,train_test_index=train_test_index) 
            model_obj_list = [classifier_dict[i] for i in model_name]
            df_model, df_cv_i = model.compare_model_cv(model_obj_list,model_name,scaffold_split=scaffold_split)
            trans_df = df_model.transpose()
            trans_df['algo'] = list(trans_df.index)
            trans_df['desc'] =  [ desc for i in list(trans_df.index)]
            trans_df.reset_index(drop=True)
            res_df = pd.concat([res_df,trans_df],ignore_index=True)
            data_obj_dict[desc] = model

            df_cv_i['desc'] = desc
            #print(df_cv_i)
            cv_df = pd.concat([cv_df,df_cv_i],ignore_index=True)
   
        res_df = res_df.rename(columns=lambda x: x.replace('test_', 'Val_') if x.startswith('test_') else x) # in sklearn, validation results are labeled as 'test_' 
        prior_cols = ['algo', 'desc','Test_ACC','Test_AUC', 'Test_MCC', 'Test(tn, fp, fn, tp)','Val_ACC','Val_AUC','Val_MCC','Val_MAP']
        non_prior_cols = [i for i in res_df.columns if i not in prior_cols]

    if task == "Regression":
        for desc in desc_list:
            X = get_desc(smi_series,desc)  
            model = QSAR_Regressor(X,y,smi_series,test_size=test_size,train_test_index=train_test_index) 
            model_obj_list = [regressor_dict[i] for i in model_name]
            df_model, df_cv_i = model.compare_model_cv(model_obj_list,model_name,scaffold_split=scaffold_split)
            trans_df = df_model.transpose()
            trans_df['algo'] = list(trans_df.index)
            trans_df['desc'] =  [ desc for i in list(trans_df.index)]
            trans_df.reset_index(drop=True)
            res_df = pd.concat([res_df,trans_df],ignore_index=True)
            data_obj_dict[desc] = model

            df_cv_i['desc'] = desc
            cv_df = pd.concat([cv_df,df_cv_i],ignore_index=True)
   
        res_df = res_df.rename(columns=lambda x: x.replace('test_', 'Val_') if x.startswith('test_') else x)
        prior_cols = ['algo', 'desc','Test_pearsonR','Test_MAE','Test_RMSE','Val_pearsonR','Val_MAE','Val_RMSE']
        non_prior_cols = [i for i in res_df.columns if i not in prior_cols]

    print("cv_df",cv_df)

    return res_df[prior_cols + non_prior_cols], data_obj_dict, cv_df



def eval_predictions(y_exp,y_pred,task="Classifier",thresh=None,y_transformed_binary=False):

    
    thresh = thresh if thresh is not None else np.median(y_exp)
    if y_transformed_binary == True:
        thresh = 0.5


    classifier_metrics_name = ["Accuracy","AUC-ROC","AUC-PR","MCC","ConfusionMatrix(tn,fp,fn,tp)"]
    classifier_metrics_func = [lambda y,y_pred: accuracy_score(y >= thresh,y_pred >= 0.5),
                               lambda y,y_pred: roc_auc_score(y>=thresh,y_pred),
                               lambda y,y_pred: average_precision_score(y>=thresh,y_pred),
                               lambda y,y_pred: matthews_corrcoef(y >= thresh,y_pred >= 0.5),
                               lambda y,y_pred:  [confusion_matrix(y >= thresh,y_pred >=0.5).ravel()]]
    
    regressor_metrics_name = ["PearsonR","MAE","RMSE"]
    regressor_metrics_func = [lambda y,y_pred: pearsonr(y,y_pred)[0],
                            lambda y,y_pred: mean_absolute_error(y,y_pred),
                            lambda y,y_pred: mean_squared_error(y,y_pred)**0.5 ]
    if task == "Classifier":
        result_df =  {name:func(y_exp,y_pred) for name,func in zip(classifier_metrics_name,classifier_metrics_func)}
    if task == "Regressor":
        result_df = {name:func(y_exp,y_pred) for name,func in zip(regressor_metrics_name,regressor_metrics_func)}
    
    return result_df





#smi_col = "smiles"
#df = pd.DataFrame({'smiles': ['CCCCC','CCCCCCCCCCCC','CCCCCCCCCCC','CCCCCCCCCCCCCCCCCCCCC'], 'pMIC_Ng':[3,6,9,12]})
#df = pd.read_csv("denovo_top_50.csv")
#df_res = benchmark_model(df.smiles,[i > 3 for i in df.pMIC_g])
#print(df_res)
#result_global.query('desc == "cddd"').model.values[0].apply_model(X_cddd)







