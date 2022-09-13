# 1. Library imports
import pandas as pd 
import pickle
import numpy as np
import json
from flask import Flask, render_template, jsonify, request
from flask import Flask

app = Flask(__name__)

# 2. Lecture des données cleaning
x_test_transformed = pd.read_csv("x_test_transformed.csv")
x_test_transformed = x_test_transformed.set_index("SK_ID_CURR")
#lecture du model choisi serilisé
model_log =pickle.load(open("Model_choice.md", "rb"))

# 2. bis lecture des données non traités
app_test = pd.read_csv("application_test.csv")
app_test =  app_test.set_index("SK_ID_CURR")
# 3.  Selction de varaible pertinentes
#-------------------------- liste des vaiables pertinentes----------------------------------
listeInfos = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN',
              'AMT_INCOME_TOTAL',
 'AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',"REG_CITY_NOT_LIVE_CITY",'REG_REGION_NOT_LIVE_REGION',
  'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY','LIVE_REGION_NOT_WORK_REGION','CNT_FAM_MEMBERS',
              'LIVE_CITY_NOT_WORK_CITY',
'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_YEAR']

app_test = app_test[listeInfos]
#----------------------------- quelque transformation pour infos client
app_test['DAYS_BIRTH'] = round(app_test['DAYS_BIRTH']/-365, 0).astype(int)
app_test.loc[app_test['DAYS_EMPLOYED']> 0, "DAYS_EMPLOYED"] =70
app_test.loc[app_test['DAYS_EMPLOYED']< 0, "DAYS_EMPLOYED"] = app_test.loc[app_test['DAYS_EMPLOYED']< 0, "DAYS_EMPLOYED"]/-365
app_test['DAYS_EMPLOYED'] = round(app_test['DAYS_EMPLOYED'], 0).astype(int)





# 5. Stockage des identifiants des client dans un dictionnaire

id_clients = app_test.index.values
id_clients = pd.DataFrame(id_clients)
# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():
    
    return id_clients.to_json(orient='values')

# Chargement d'informations générales


@app.route("/infos_client", methods=["GET"])
def infos_client():
 

    id = request.args.get('id_clients', 100001, type = int)

    data_client = app_test[app_test.index == id]

   # print(data_client)
    dict_infos = {
       "status_famille" : dict(data_client["NAME_FAMILY_STATUS"].items()),
       "nb_enfant" : dict(data_client["CNT_CHILDREN"].items()),
       "age" : dict(data_client["DAYS_BIRTH"].items()),
       "revenus" : dict(data_client["AMT_INCOME_TOTAL"].items()),
       "montant_credit" : dict(data_client["AMT_CREDIT"].items()),
       "annuites" : dict(data_client["AMT_ANNUITY"].items()),
       "montant_bien" : dict(data_client["AMT_GOODS_PRICE"].items())
       }
    
    print(dict_infos)
  
    json_object = json.dumps(dict_infos, indent=2) 


    return json_object

@app.route("/predict", methods=["GET"])
def predict():
    
    id = request.args.get('id_clients', 100001, type = int)

    print(app_test.shape)
    print(app_test[app_test.index== id])

    prediction = model_log[3].predict_proba(np.array(x_test_transformed.loc[id]).reshape(1, -1)).flatten()

    prediction = prediction.tolist()
    #print("Prediction de faillite:")
    print(prediction)

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)