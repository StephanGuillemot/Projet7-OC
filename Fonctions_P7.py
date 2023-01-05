from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel

import pandas as pd
import numpy as np
import seaborn as sns
import shap
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    """
    Permet de creer des nouvelles colonnes en fonction des modalite prises pour chaque variable qualitative
    Prend un dataframe en entree et renvoi un nouveau dataframe et les noms des nouvelles colonnes
    """
    
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



def missing_df(df):
    """
    Fonction qui permet de renvoyer le taux de remplissage de remplissage des colonnes dans un nouveau dataframe
    Prend un dataframe en entree
    """
    missing_count = df.isnull().sum()  # the count of missing values
    value_count = df.isnull().count()  # the count of all values
    missing_percentage = round(missing_count / value_count * 100,
                               2)  # the percentage of missing values
    # Pourcentage de Valeurs manquantes par colonnes
    missing_df = pd.DataFrame({
        'count': missing_count,
        'percentage': missing_percentage
    })  # create a dataframe
    return missing_df


def application_train(num_rows = None, nan_as_category = False):
    """
    Fonction de traitement des donnees application
    Transformation des variables qualitative en variable onehot encoder
    Correction des valeurs aberrantes
    Feature engineering
    Imputation des valeurs manquantes
    Renvoi un dataframe pret a la modelisation
    """
    
    # Import des donnees
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    
    print("Train samples: {}".format(len(df)))

    # Suppression des valeurs inconnues pour le sexe du credit
    df = df[df['CODE_GENDER'] != 'XNA']
    
    with pd.option_context('display.max_rows', None) :
        # Affichage des valeurs manquantes sur la console pour justifier les changements suivants
        print(missing_df(df).sort_values(by ='percentage'))
        
        
    # Traitement des variables binaires
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
        
    # valeurs aberrantes
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    # Imputation de la moyenne
    df['AMT_GOODS_PRICE'] = np.where(df['AMT_GOODS_PRICE'].isna(), df['AMT_GOODS_PRICE'].mean() ,df['AMT_GOODS_PRICE'])
    df['AMT_ANNUITY'] = np.where(df['AMT_ANNUITY'].isna(), df['AMT_ANNUITY'].mean() ,df['AMT_ANNUITY'])
    df['DAYS_EMPLOYED'] = np.where(df['DAYS_EMPLOYED'].isna(), df['DAYS_EMPLOYED'].mean() ,df['DAYS_EMPLOYED'])
    df['CNT_FAM_MEMBERS'] = np.where(df['CNT_FAM_MEMBERS'].isna(),1 ,df['CNT_FAM_MEMBERS'] )
    
    
    # Nouvelles variables (feature engineering)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']      
    
    # One hot encoding des variables 
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    print(cat_cols)
    

    
    # Imputation des valeurs manquantes
    df.fillna(0, inplace=True)
    
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
    to_drop = [i for i in range(len(upper_tri.columns)) if any(upper_tri[upper_tri.columns[i]] > 0.95)]
    print(to_drop)
    df = df.drop(df.columns[to_drop], axis=1)
    return df


def train_score_model(clf,X_train, y_train, X_test, y_test):
    """ 
    Entraine et score le modele sur les donnees d'entree
    Renvoi le nouveau modele et le score 
    """
    
    clf.fit(X_train,y_train)
    score = roc_auc_score(y_test, [i[1] for i in clf.predict_proba(X_test)])
    return clf, score


def feature_importance_model(clf, X_train):
    """
    Renvoi un dataframe contenant les feature importance du modele suivant si le modèle dispose de coefficient ou de feature importance
    """
    
    Feat_imp = pd.DataFrame({})
    if hasattr(clf[-1], "feature_importances_"):
        Feat_imp['imp'] = clf[-1].feature_importances_
    elif hasattr(clf[-1], "coef_"):
        Feat_imp['imp'] = clf[-1].coef_[0]
    Feat_imp['Variable'] = X_train.columns
    return Feat_imp


def local_interpretability_shap(clf, X_test):
    
    """
    Renvoi les valeurs de shap pour le modèle par rapport aux données de test 
    Valable uniquement pour les modèles d'arbres
    """
    pred = clf.predict(X_test)
    explainer = shap.TreeExplainer(clf[-1])
    observations = clf[0].transform(X_test)
    shap_values = explainer.shap_values(observations)
    return shap_values, observations
    

def local_interpretability_lime(clf, X_train, X_test, index):
    """
    Renvoi l'interpretabilité de LIME pour l'individu considéré
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data = np.array(X_train),
        feature_names = X_train.columns,
        class_names = [0, 1],
        mode = 'classification'
    )
    exp = explainer.explain_instance(
        data_row = X_test.iloc[index], 
        predict_fn = clf.predict_proba
    )

    return exp

def select_features(X_train, y_train, X_test):
    """
    Selectionne les meilleures 10 features pour un modèle logistic
    Renvoi les données d'entrainement et de test avec ces uniquement features
    """
    # configure to select a subset of features
    fs = SelectFromModel( LogisticRegression(solver='liblinear', verbose = True), max_features=10)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def cor_selector(X, y,num_feats):
    """
    Retourne les meilleures variables suivant leur coefficient de correlation entre la variable et la cible
    """
    
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


def chi_selector(X, y,num_feats):
    """
    Retourne les meilleures variables suivant leur chi 2 entre la variable et la cible
    """
    chi_selector = SelectKBest(chi2, k=num_feats)
    X_norm = MinMaxScaler((0,1)).fit_transform(X)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature


def rfe_selector(X, y,num_feats):
    """
    Retourne les variables selectionne a l'issu d'un backward selection
    """
    rfe_selector = RFE(estimator=LogisticRegression(max_iter = 1000), n_features_to_select = num_feats, step = 30, verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_support, rfe_feature

def embeded_selector(X, y,num_feats, model):
    """
    Retourne les variables parmis les meilleures variables suivant leur chi 2 entre la variable et la cible
    """
    embeded_selector = SelectFromModel(model, max_features=num_feats)
    embeded_selector.fit(X, y)

    embeded_support = embeded_selector.get_support()
    embeded_feature = X.loc[:,embeded_support].columns.tolist()
    return embeded_support, embeded_feature

def impact_proba(clf, X_test, y_test):
    
    """
    Calcule les classements du modèle suivant le seuil de probabilite considere
    Nb de dossiers accepte
    Nb de dossiers rejete
    Tx de defaut accepte et rejete
    
    """
    defaut = pd.DataFrame({})
    defaut['defaut'] = y_test
    defaut['proba'] =[i[1] for i in clf.predict_proba(X_test)]

    sns.histplot(data= defaut, x= 'proba', hue = 'defaut')
    plt.show()
    
    
    seuils = np.linspace(-0.01, defaut['proba'].max(), 30)

    liste_accepte=[]
    liste_rejete=[]
    liste_tx_rejete=[]
    liste_tx_defaut_accepte=[]
    liste_tx_defaut_rejete = []

    for seuil in seuils:
        defaut[f'Dossiers_rejete_{round(seuil, 3)}'] = np.where(defaut['proba'] < seuil, 0, 1)
        Nb_dossiers = defaut[f'Dossiers_rejete_{round(seuil, 3)}'].count()
        Nb_dossiers_rejete = defaut[f'Dossiers_rejete_{round(seuil, 3)}'].sum()
        Nb_dossiers_accepte = Nb_dossiers - Nb_dossiers_rejete
        Nb_dossiers_defaut_accepte = defaut[defaut['defaut'] == 1][ defaut[f'Dossiers_rejete_{round(seuil, 3)}'] ==0][f'Dossiers_rejete_{round(seuil, 3)}'].count()
        Nb_dossiers_defaut_rejete = defaut[defaut['defaut'] == 1][defaut[f'Dossiers_rejete_{round(seuil, 3)}'] ==1][f'Dossiers_rejete_{round(seuil, 3)}'].count()

        liste_accepte.append(Nb_dossiers_accepte)
        liste_rejete.append(Nb_dossiers_rejete)
        liste_tx_rejete.append( Nb_dossiers_rejete / Nb_dossiers)
        liste_tx_defaut_accepte.append(Nb_dossiers_defaut_accepte / (Nb_dossiers_accepte))
        liste_tx_defaut_rejete.append(Nb_dossiers_defaut_rejete/ Nb_dossiers_rejete)
        print('########')
        print(seuil, ' : Accepte : ',Nb_dossiers_accepte ,
              '\n Rejete :', Nb_dossiers_rejete)
        print('Tx de rejete: ', Nb_dossiers_rejete / Nb_dossiers)
        print('Tx de defaut Accepte',
              Nb_dossiers_defaut_accepte / (Nb_dossiers_accepte))
        print('Tx de defaut rejete', Nb_dossiers_defaut_rejete/ Nb_dossiers_rejete)
   
    plt.plot(seuils, liste_accepte)
    plt.plot(seuils, liste_rejete)
    plt.show()
    plt.plot(seuils, liste_tx_rejete)
    plt.plot(seuils, liste_tx_defaut_accepte)
    plt.plot(seuils, liste_tx_defaut_rejete)
    plt.show()





























