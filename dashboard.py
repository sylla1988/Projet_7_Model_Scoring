# Import des librarie
from dash import Dash, html, dcc, Input, Output, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import lime
from lime import lime_tabular
import pickle
from operator import itemgetter
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objs as go
import base64

# Import de base de données  
x_test_transformed = pd.read_csv("x_test_transformed.csv")
x_test_transformed = x_test_transformed.set_index("SK_ID_CURR")
#lecture du model choisi
pipe_model =pickle.load(open("Model_choice.md", "rb"))

#lecture de la matice de confusiion optimal
conf_mx =pickle.load(open("conf_mx_opt", "rb"))


#-----------------------------------------------------------------------------------
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
#---------------------------------------------------------------------------------------

explainer = lime_tabular.LimeTabularExplainer(x_test_transformed,
                             feature_names=x_test_transformed.columns,
                             class_names=["0", "1"],
                             discretize_continuous=False)

#------------------------------------------------------------------------------------------------------

path = r'C:\Users\sylla\Desktop\Data Sciences\Projet7_ impémenter un model scoring\archive/'

app_test = pd.read_csv(path + "application_test.csv")
app_test = app_test.set_index("SK_ID_CURR")
#----------------------------------------------------------------------------------------------------
df_cout = pd.read_csv("df_cout.csv")
#---------------------------------------------------------------------------------------------------

# Calcul des 5 plus proches voisins
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(x_test_transformed)

#-------------------------- liste des vaiables pertinentes----------------------------------
listeInfos = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN',
              'AMT_INCOME_TOTAL',
 'AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',"REG_CITY_NOT_LIVE_CITY",'REG_REGION_NOT_LIVE_REGION',
  'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY','LIVE_REGION_NOT_WORK_REGION','CNT_FAM_MEMBERS',
              'LIVE_CITY_NOT_WORK_CITY',
'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_YEAR']

#---------------------------- selection de vaeiable à afficher ------------------------------------
app_test = app_test[listeInfos]
#--------------------------------------------------------------------------
app_test['DAYS_BIRTH'] = round(app_test['DAYS_BIRTH']/-365, 0).astype(int)
app_test.loc[app_test['DAYS_EMPLOYED']> 0, "DAYS_EMPLOYED"] =70
app_test.loc[app_test['DAYS_EMPLOYED']< 0, "DAYS_EMPLOYED"] = app_test.loc[app_test['DAYS_EMPLOYED']< 0, "DAYS_EMPLOYED"]/-365
app_test['DAYS_EMPLOYED'] = round(app_test['DAYS_EMPLOYED'], 0).astype(int)
#------------------------------------------------------------------------------------------------------
##Définition des function
def plot_mat_conf(conf_mx):
    
    labels = ["False", "True"]
    
    annotations = go.Annotations()
    for n in range(conf_mx.shape[0]):
        for m in range(conf_mx.shape[1]):
            annotations.append(go.Annotation(text=str(conf_mx[n][m]), x=labels[m], y=labels[n],
                                             showarrow=False))
            
            
    colorscale=[[0.0, 'rgb(255, 255, 153)'], [.2, 'rgb(255, 255, 203)'],
            [.4, 'rgb(153, 255, 204)'], [.6, 'rgb(179, 217, 255)'],
            [.8, 'rgb(240, 179, 255)'],[1.0, 'rgb(255, 77, 148)']]

    trace = go.Heatmap(x=labels,
                       y=labels,
                       z=conf_mx,
                       colorscale=colorscale,
                       showscale=True)

    fig_confusion = go.Figure(data=go.Data([trace]))
    
    fig_confusion['layout'].update(
        annotations=annotations,
        xaxis= dict(title='Classes prédites'), 
        yaxis=dict(title='Classes réelles', dtick=1),
        margin={'b': 30, 'r': 20, 't': 10},
        width=600,
        height=450,
        autosize=False
    )
    
    return fig_confusion # Retourne la figure crée
#-------------------------------------------------------------------------------------------------
def coef_importances(pipe_model) :
     
    coef = pipe_model[3].coef_.flatten()
    indices =[]
    values = []
    for val, ind in sorted(zip(pipe_model[3].coef_.flatten(),
                               x_test_transformed.columns), reverse=True):
        indices.append(ind)
        values.append(val)
        
    data = pd.DataFrame(values, columns=["values"], index=indices)
    data["positive"] = data["values"]>0
    del indices, values

    traces = [go.Bar(x=data["values"], y=data.index,
                    orientation='h',
                    marker_color=list(data.positive.map({True: 'red', False: 'blue'}).values))]
     
    return {
        'data': traces,
        
        'layout': go.Layout(margin=dict(l=300, r=0, t=30, b=100))
           }


def fonction_cout():
    x_axis = df_cout["threshold"]
    y_axis_cout = df_cout["cost"]
    min_cout = df_cout['cost'].min()
    fig = go.Figure()
    traces = go.Scatter(
            x= x_axis,
            y= y_axis_cout,
            mode= 'lines+markers',
            y0 =min_cout,
            line={'shape':'spline','smoothing':1},
            name= "Fonction cout")
  

    return {
        'data': [traces],
        'layout': go.Layout(
            title = "Evolution du coût par rapport aux thresholds",
            xaxis= dict(title = "Thresholds"),
            yaxis= dict(title = "Coût"),
          
            hovermode= 'closest',
            legend_orientation= 'h'

        )
    }

# Choix des variables pertinentes pour information client
#-------------------------------------------------------------------------------------
variables = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',"DAYS_BIRTH","DAYS_EMPLOYED",
                 'CNT_FAM_MEMBERS' ,'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3', "AMT_REQ_CREDIT_BUREAU_YEAR" ]

variable_inf = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',"DAYS_BIRTH","DAYS_EMPLOYED",
                 'CNT_FAM_MEMBERS' ,'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3', "AMT_REQ_CREDIT_BUREAU_YEAR" ]


#----------------------------------------------------------------------------------------------------
# Markdown pour le service clientel
#-----------------------------------------------------------------------------------------------------
markdown_text = '''
### Seuil de probabilté

Le seuil de probalilité obtenu après optimisation de la
fonction coût est de: **0.54123**.

En sous de cette  valeur, **demande acceptée**;


Sinon, **refusée**.
'''

#-------------------------------------------------------------------------
 #initialisation de dash
app = Dash(__name__)
# definition de la colone vertebrale du dashboad avec dcc qui permet de creer des onglets
id_dropdown = dcc.Dropdown( id='id_client',
                           options=x_test_transformed.index,
                            value=x_test_transformed.index[0],
                           multi = True
                          )
## Définition de loyout de app
app.layout = html.Div([
   html.Div( children =[
                html.Div([
            #Mettre un titre au dropdown
            html.H4("Choisissez l'identifiant d'un client:"),
            ## deuxieme composante un dcc
            dcc.Dropdown(
                ##donner un id
                id = 'id_client',
                #on choisit l'option qu'est je vais mettre à l'interieur du dropdown (ici les identifiant des clients
                options = x_test_transformed.index,
                #on on peut même donner une valeur par defaut le client à l'index 0
                value = x_test_transformed.index[0]
            )
        ], style = {
             'background-color': 'rgb(200, 200, 200)',
    
            'width': "25%", # 25 % de la page
            "border": '1px solid #eee',
            'padding': '30px 30px 30px 12px',
            'box-shadow': '0 2px 2px #ccc',
             'display': 'inline_block',
            'textAlign': 'left'}),
#On va definir les onglet à farire apparaitre sur le tableau de bord
html.Div([
        dcc.Tabs(id ='tab', value ='tab-1', # les children sont les enfants de de Tabs 
                children = [
                    # 1 ier onglet: infos client
                    dcc.Tab(label = 'Infos client', children = [
                         html.Div([html.H3('Infos client') 
                             
                         ], style ={'background': 'blue',
                                          "color": "white", 
                                    'textAlign':'center',
                                'padding':'10px 0px 10px 0px'}),

html.Div([
                            dash_table.DataTable(
                                id = "table_infos",style_cell = {'font-family': 'Montserrat'}, 
                    style_data_conditional  =[{
                        
                        'if': {'column_id': 'intitule'},
                        'textAlign': 'left'
                    }] +
                    
                    [{
                        
                        'if': {'row_index': 'odd'}, # si les ligens sont impaire
                        'backgroundColor ': 'rgb(248,248, 248)'
                    }], style_header = { 'backgroundColor ': 'rgb(230,230, 230)',       #l'entête un peu plus foncé
                                        'fontWeight': 'bold'
                        
                    }
                           
                )], style= {'display':'inline-block',
                            'verticalAlign':'top',
                            'width': '45%', 
                            'padding':'40px 0px 0px 10px'} ),
                        
                        
                        html.Div([
                            dcc.Graph(id = 'bar_infos')
                            
                            
                        ], style= {'display':'inline-block', 
                                   'verticalAlign':'top',
                                   'width': '45%',
                                  'padding':'70px 0px 0px 70px',
                                  'box-shadow': '2px 2px 2px  #ccc'}),

 #--------------------liste deroulante de toute les variable de la table  client simil----------------------
                     html.Div([  
            
            html.H4("Choisissez une variable:"),
            ## deuxieme composante un dcc
            dcc.Dropdown(
                ##donner un id
                id = 'variable_info',
                #on choisit l'option qu'est je vais mettre à l'interieur du dropdown (ici les identifiant des clients
                options = variable_inf,
                #on on peut même donner une valeur par defaut le client à l'index 0
                value = variable_inf[0])

                     ], style = {'width': "25%", # 25 % de la page
                                            "border": '1px solid #eee',
                                            'padding': '30px 30px 30px 120px',
                                            'box-shadow': '2px 2px 0px #ccc',
                                             'display': 'inline_block',
                                            'verticalAlign': 'top'}),
   
 #-------------------------------------- Titre -------------------------------------------
                      html.Div([html.H3('Information du client choisi par rapport à la moyenne de la population') 
                             
                         ], style ={'background': 'blue',
                                          "color": "white", 
                                    'textAlign':'center',
                                'padding':'10px 0px 10px 0px'}),

 #_________________________________ graphe infos clien vs moyenne pop________________________________________________  
                        html.Div([
                            dcc.Graph(id = 'bar_moyenne')
                            
                            
                        ], style= {'display':'inline-block', 
                                   'verticalAlign':'center',
                                   'width': '90%',
                                  'padding':'70px 0px 0px 70px',
                                  'box-shadow': '2px 2px 2px  #ccc'}),
  
                                
                    ]),
                 





                ])


        ])

   ])

])