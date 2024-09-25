import pandas as pd
import numpy as np
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from joblib import load

columns_df = pd.read_csv('../../columns/columns.csv', header=None)

model_num = int(input("choisissez quel model vous souhaitez utiliser : \n entrer '1' pour le 'decison tree'\n entrer '2' pour le 'K neighbors'\n entrer '3' pour le 'logistic regression (le pire entre eux)'\n entrer '4' pour le 'random forest' \n votre réponse SVP : "))
l = ["decision_tree","k_nearest_neighbors","logistic_regression","random_forest"]
model = load(f'../../models/{l[model_num-1]}.pkl')

RUNNING = True

while RUNNING:
    file_path = input("donne moi le chemin du fichier contenant le log pour l'analyser : ")
    #DATA CLEANING
    
    data_df = pd.read_csv(file_path, sep='\t', comment='#',header=None)
    data_df.columns = columns_df[0].tolist()

    tunnel_parents_column = data_df.iloc[:,-1].apply(lambda x: x.split()[0])

    data_df.drop(columns=["ts","uid","local_resp","local_orig","tunnel_parents"], inplace=True)
    data_df.drop(columns=["id.orig_h","id.resp_h"], inplace=True)

    data_df.replace({'-':np.nan, "(empty)":np.nan}, inplace=True)

    dtype_convert_dict = {
        "duration": float,
        "orig_bytes": float,
        "resp_bytes": float
    }
    data_df = data_df.astype(dtype_convert_dict)

    #DATA PREPROCESSING

    numerical_features = ["id.orig_p", "id.resp_p", "duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes",	"resp_pkts", "resp_ip_bytes"]
    categorical_features = ["proto","service","conn_state","history"]
    min_max_scaler = MinMaxScaler()
    data_df[numerical_features] = min_max_scaler.fit_transform(data_df[numerical_features])
    data_df["history"].value_counts()

    valid_history_values = ["C","S","D","ShAdDaf","Dd","ShAdDaft","ShAdfDr"]
    data_df["history"] = data_df["history"].apply(lambda x: x if x in valid_history_values else "Other")
    data_df["history"].value_counts()
    ohe = load('../../encoder/onehot_encoder.pkl')

    encoded_features = ohe.transform(data_df[categorical_features])
    encoded_features_df = pd.DataFrame(encoded_features.toarray(), columns=ohe.get_feature_names_out())

    data_df = pd.concat([data_df, encoded_features_df], axis=1).drop(categorical_features, axis=1)

    #APPLY THE MODEL
    if model_num in [2,3] :
        data_df =  data_df.fillna(0.0)
    predictions = model.predict(data_df.values)
    for elt in predictions :
        if elt == 0 :
            print("----------normal-----------")
        else :
            print("!!!------malicious------!!!")