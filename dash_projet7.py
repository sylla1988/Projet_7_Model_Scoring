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
  
           
                                   ]
             
            
        ),
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
    ################################################################################################
                    # 2 ieme onglet: Score de probabilté de faillite
                    dcc.Tab(label = 'Situation du client', children =[
   #****************************************************************************************      
           
                 html.Div([html.H3("Acceptabilité de la demande de prêt du client ")], style ={'background': 'blue',
                                                                                             "color": "white", 'textAlign':'center',
                                                                                           'padding':'10px 0px 10px 0px'}), 
   # prediction 
   html.Div(id='result', children=["Resultat"], style={'background': '',
                                                                       "color": "black",
                                                                      # "height": '50px',
                                                                       'border': '5px dotted red',
                                                                       'padding':'50px 3px 50px 50px'}), 
#------------------------------------------------------------------------------   
                        
                         html.Div([
                                        dcc.Markdown(children=markdown_text)
                                    ],style ={'background': '',
                                                "color": "black", 'textAlign':'center',
                                                 'border': '10px solid #ccc', 
                                                  'padding':'10px 0px 10px 0px'} ),
                  #****************************************************************************************       
                html.Div([html.H3("Eléments justifiant la décision de prêt ou non ")], style ={'background': 'blue',
                                                                                             "color": "white", 'textAlign':'center',
                                                                                    'padding':'10px 0px 10px 0px'}),

                      html.Div([
                    dcc.Graph(id = 'proba_value'),
                            
                   html.P("Position of hline"),
                dcc.Slider(
                id='slider-position', 
                        min=0, max=1, value=0.54123, step=0.54123,
                    marks={0: '0',0.54123:'0.54123', 1: '1'}
                                        ),
                     ], style = {'border': '1px solid #ccc', 
                                #'box-shadow': '0 2px 2px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '45%',
                                 'padding': '0px 0px 0px 0px'
                                }),
                    #____________________________________________________________
                        html.Div([
                    dcc.Graph(id = 'graph_lime')
                     ], style = {#'border': '1px solid #ccc', 
                                #'box-shadow': '0 2px 0px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '45%',
                                'padding': '50px 0px 0px 30px'
                                }), 
                             #________________________client similaires__________________________
                      
             html.Div([html.H3("Tableau des données du client choisi et de ses quatre plus proches voisins ?")], 
                      style ={'background': 'blue',
                               "color": "white", 'textAlign':'center',
                           'padding':'10px 0px 10px 0px'}),
        html.Div([            
                    dash_table.DataTable(
                                            id='client_similaire',
                                            columns=[  {"name": i, "id": i} for i in app_test.reset_index().columns],
                                            
                                           filter_action='custom',
                                            filter_query='',
                                            fixed_rows={'headers': True, 'data': 0 },
                                           style_cell={'width': '200px'},
                                            style_table={'minWidth': '90%'},
                                            style_data_conditional=[
                                                                    {'if': {'row_index': 'odd'},
                                                                    'backgroundColor': 'rgb(248, 248, 248)' 
                                                                    }],
                                            style_header={
                                                            'backgroundColor': 'rgb(230, 230, 230)',
                                                            'fontWeight': 'bold'
                                                        }, 
                                            virtualization=True,
                                         )


                                  ],style ={'width': '90%', "border": '1px solid #eee',
                            'box-shadow': '0 2px 2px #ccc',
                             'display': 'inline_block',
                            'verticalAlign': 'top',
                            'padding':'60px 30px 60px 30px'
                                           } ),

                         #--------------------liste deroulante de choix de variable pour afficher une pie plot----------------------

                     html.Div([  
            
            html.H4("Choisissez une variable:"),
            ## deuxieme composante un dcc
            dcc.Dropdown(
                ##donner un id
                id = 'variable',
                #on choisit l'option qu'est je vais mettre à l'interieur du dropdown (ici les variables)
                options = variables,
                #on  peut même donner une valeur par defaut le client à l'index 0
                value = variables[0])

                     ], style = {'width': "25%", # 25 % de la page
                                            "border": '1px solid #eee',
                                            'padding': '30px 30px 30px 120px',
                                            'box-shadow': '2px 2px 0px #ccc',
                                             'display': 'inline_block',
                                            'verticalAlign': 'top'}),
              #-------------------------------------- Titre -------------------------------------------
                        
                  html.Div([  
            
                               html.Div([html.H3("Comparaison du client choisi par rapport à ses plus proches voisins")],
                                              style ={'background': 'blue',
                                                        "color": "white", 'textAlign':'center',
                                                        'padding':'10px 0px 10px 0px'}),

                                ]),

                      #------------------------------------------------------- Grphe 1---------------------------------------------
                        html.Div([
            dcc.Graph(id = 'graph_comp_1')
                    ], style = {'border': '1px solid #ccc', 
                                'box-shadow': '0 2px 2px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '90%',
                                 'padding': '50px 0px 0px 50px'} ),                                                                                                                                            
                        
                    ]),
                    # fin deuxieme onglet
   #*****************************************************************************************         
                            # 3ieme onglet: Grahique des coef importances local

                         dcc.Tab(label="Performance du model", children=[
 #------------------------------- Matrice de confusion ------------------------
                                  html.Div([
                                      html.H3("Matrice de confusion et coefficients importances"),
                                            
                                           ], style ={'background': 'blue',
                                                    "color": "white", 'textAlign':'center',
                                                    'padding':'10px 0px 10px 0px'},
                                           ),
                     html.Div([
                            # Affiche la matrice de confusion obtenue apres optimisation fonction cout
                                            
                                            html.Div([
                                                html.H3("Matrice de confusion"), 
                                                dcc.Graph(id='matrice_conf',
                                                          figure= plot_mat_conf(conf_mx),
                                                         ),
                                                     ], style = {'border': '1px solid #ccc', 
                                                                'box-shadow': '0 2px 2px #ccc',
                                                                 'display': 'inline-block',
                                                                 'verticalAlign': 'top',
                                                                 'width': '45%',
                                                                 'padding': '50px 0px 0px 50px'
                                                                }),
                                 #------------------------------- coeficicient importances globales------------------------
                        
                                            html.Div([
                                                html.H3("Coeficient importances"), 
                                                dcc.Graph(id='coef_importance',
                                                          figure= coef_importances(pipe_model),
                                                         ),
                                                     ],    style = {'border': '1px solid #ccc', 
                                                                    'box-shadow': '0 2px 2px #ccc',
                                                                     'display': 'inline-block',
                                                                     'verticalAlign': 'top',
                                                                     'width': '45%',
                                                                     'padding': '50px 0px 0px 50px'
                                                                    }),


                             ]),                                                 



                    ]) ,# fin troisieme onglet
                 # 4 ième onglet: Tableau fonction cout
                    dcc.Tab(label = 'Fonction coût', children = [
                                  
        #-------------------------------------------------------petit titre de l'onglet-----------------------
                    html.Div([html.H3('Graphique de la fonction de coût optimisée'),
                             ],
                              style = {'background': 'blue',
                                         "color": "white", 'textAlign':'center',
                                          'padding':'10px 0px 10px 0px'}),       
                           
                        html.Div([
                            dcc.Graph(id = 'cout', 
                                 figure = fonction_cout(),
                                     
                                 ),
                        ], style = {'border': '1px solid #ccc', 
                                      'box-shadow': '2px 2px 2px #ccc',
                                        'display': 'inline-block',
                                        'verticalAlign': 'top',
                                        'width': '80%',
                                            'padding': '50px 0px 0px 100px'})   
                                ],
                           ),  #fin 4 ieme fin             

                ]) #fin des onglet 
        ]) #fin div global

]) #fin div Layout de app dash
# Création d'un système de filtre
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]
def split_filter_part(filter_part):
    # Permet d'avoir un outil de filtrage des données
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3
############################ infos client #######################################################""

@app.callback([Output('table_infos', 'data'),Output('table_infos', 'columns')],
              [Input("id_client", 'value')]) #obliger de la mettre entre crocher meme si c"est un seul elemen
def infos_client(id_client):
    infos_list = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
                  'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',"REG_CITY_NOT_LIVE_CITY",'REG_REGION_NOT_LIVE_REGION',
                'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY','LIVE_REGION_NOT_WORK_REGION','CNT_FAM_MEMBERS',
              'LIVE_CITY_NOT_WORK_CITY', 'AMT_INCOME_TOTAL']
    
    #info_list = [info  for info in col ]
    infos = {'intitule':infos_list,
            'donnee': [app_test[app_test.index==id_client][col].iloc[0] for col in infos_list]
            }
    
    table = pd.DataFrame(infos)
    data = table.to_dict('rows')
    entete = {'id':'intitule', 'name': "Principaux indicateurs"}, {'id': 'donnee',
                                                                   'name': "Id_client: " + str(id_client)}
    
    return data, entete
    #--------------------------------------- grahe pie client----------------------------------------------------#
@app.callback(Output('bar_infos', 'figure'),
              [Input('id_client', 'value')])
def graph_bar_infos_client(id_client):
    
    col_bar = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE']
    data_info = [float(app_test[app_test.index ==id_client][col].iloc[0]) for col in col_bar]
    
    #data_moy_age_pop = round(app_test["DAYS_BIRTH"].mean(), 0).astype(int)
    data_moy_incom_pop = round(app_test["AMT_INCOME_TOTAL"].mean(), 2)
    data_moy_credit_pop = round(app_test["AMT_CREDIT"].mean(),2)
    data_moy_goods_pop = round(app_test["AMT_GOODS_PRICE"].mean(), 2)
    
    data_pop= [data_moy_incom_pop, data_moy_credit_pop,  data_moy_goods_pop]
    
    labels =  col_bar
   
    traces = [
        go.Bar(x= labels, y =data_info, text =data_info , textposition='auto', name = "Client choisi"),
       go.Bar(x= labels, y = data_pop, text =data_pop , textposition='auto', name = "Moyennne Pop"  ),

    ]
    del data_info
    return {
        'data': traces,
        'layout': go.Layout(
           
            title= 'Information client vs moyenne de la population <br> dans la même catégorie',
            legend_orientation= 'h'
        )
    }
############################################## acceptabilité de la demande ##################""
@app.callback(
    Output('result', 'children'),
    [Input('id_client', 'value')])

def update_result(id_client):
    proba = pipe_model[3].predict_proba(np.array(x_test_transformed.loc[id_client]).reshape(1, -1)).flatten()
    proba = proba[1]
    if (proba < 0.541237): 
        return "Demande de prêt accéptée !"
    else:
        return "Demande de prêt refusée !"


############################ probabilté de faillite #######################################################""

@app.callback(Output('proba_value', 'figure'),
             [Input('id_client', 'value')])
def update_proba_bar(id_client):    
    proba = pipe_model[3].predict_proba(np.array(x_test_transformed.loc[id_client]).reshape(1, -1)).flatten()
    prod = np.array([proba[1]])
    df = ["Classe"]
    fig =px.bar(x = df, y= prod,
                labels=dict(x = '...= Seuil', y='Score proba'),
               title='Le score de probabilité  pour prise de décision ', text_auto=True)
    fig.add_hline(y=0.8, line_color="black", line_dash="dot", line_width = 0.001)
    fig.add_hline(y=0.54123, line_color="red", line_dash="dot", line_width = 2)
    return fig
    del proba
#****************************************graph de lime**********************************
@app.callback(Output('graph_lime', 'figure'),
             [Input('id_client', 'value')])
def graphe_lime(id_client) :
     
    exp = explainer.explain_instance(x_test_transformed.loc[id_client],
                                 pipe_model[3].predict_proba)
    
    indices, values = [], []
    

    for ind, val in sorted(exp.as_list(), key=itemgetter(1)):
        indices.append(ind)
        values.append(val)
    data = pd.DataFrame(values, columns=["values"], index=indices)
    data["positive"] = data["values"]>0
    del indices, values
    
    # Retourne le barplot correspondant aux 'feature importances'
    # du client dont l'id est séléctionné sur le dashboard
    traces = [go.Bar(
                    x=data["values"],
                    y=data.index,
                    orientation='h',
                    marker_color=list(data.positive.map({True: 'red', False: 'green'}).values)
        )]
    return {
        
        'data': traces,
        
        'layout': go.Layout(
                            margin=dict(l=300, r=0, t=30, b=100),
            title = "Influences des variables(Vert facilitant Non faillite et rouge  faillite)"
        )  
                           
    }
############################################ client similaire ################################
@app.callback(
    Output('client_similaire', 'data'),
    [Input('client_similaire', "filter_query"),
     Input('id_client', "value")])
def update_table(filter, id_client):
    
    # Déterminer les individus les plus proches du client dont l'id est séléctionné
    indices_similary_clients = nbrs.kneighbors(np.array(x_test_transformed.loc[id_client]).reshape(1, -1))[1].flatten()
     
    filtering_expressions = filter.split(' && ')
    dff = app_test.iloc[indices_similary_clients].reset_index()
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]
    
    return dff.to_dict('records')
   ################################### graphe_comp_1 ###################### 
@app.callback(Output('graph_comp_1', 'figure'),
             [Input('id_client', 'value'), Input('variable', 'value')])
def graph_comparaison_1(id_client,variable):
    indices_similary_clients = nbrs.kneighbors(np.array(x_test_transformed.loc[id_client]).reshape(1, -1))[1].flatten()
    df_client_simil = app_test.iloc[indices_similary_clients]
 
    ind_client_vs = df_client_simil.index.to_list()
    
    data_client_princi = df_client_simil[df_client_simil.index == id_client][variable].iloc[0]
    data_client_1 = df_client_simil[df_client_simil.index  == ind_client_vs[1]][variable].iloc[0]
    data_client_2 = df_client_simil[df_client_simil.index  == ind_client_vs[2]][variable].iloc[0]
    data_client_3 = df_client_simil[df_client_simil.index  == ind_client_vs[3]][variable].iloc[0]
    data_client_4 = df_client_simil[df_client_simil.index  == ind_client_vs[4]][variable].iloc[0]

    labels = ['Client choisi', 'Voisin_1', "Voisin_2", "Voisin_3", "Voisin_4"]
    values = [float(data_client_princi), float(data_client_1),float(data_client_2), float(data_client_3),
              float( data_client_4) ]
    
    total = sum(values)

    traces = [
        go.Pie(labels= labels, values= values, texttemplate = "%{label}: %{value:s} <br>(%{percent})",
    textposition = "inside")
    ]

    return {
        'data': traces,
        'layout': go.Layout(
        title= " Information du client choisi comparée <br> à celle de ses 4 + proches voisins  (Total: " + str(total) + ")",
            #legend_orientation= 'h'
        )
    }
##################################################Graphe moyenne #####################################################
@app.callback(Output('bar_moyenne', 'figure'),
             [Input('id_client', 'value'), Input('variable_info', 'value')])
def graph_bar_moyenne(id_client,variable_info):

    
    data_client_princi = app_test[app_test.index == id_client][variable_info].iloc[0]
    data_moy = round(app_test[variable_info].mean(), 0).astype(int)
 

    labels = ['Client choisi', "Moy_Pop"]
    values = [float(data_client_princi), float(data_moy)]
    
    total = sum(values)

    traces = [
        go.Pie(labels= labels, values= values, hole =0.5, texttemplate = "%{label}: %{value:s} <br>(%{percent})",
    textposition = "inside")
    ]

    return {
        'data': traces,
        'layout': go.Layout(
        title= " Information du client choisi comparée <br> à la moyenne de la population (Total: " + str(total) + ")",
            #legend_orientation= 'h'
        )
    }
server =  app.server  
if __name__ == "__main__":
    app.run_server(debug =True) 