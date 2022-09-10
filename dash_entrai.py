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

#----------------------------------------------------------------------------------
x_test_transformed = pd.read_csv("x_test_transformed.csv")

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

#---------------------------------------------------------------------------------------------------

# Calcul des 5 plus proches voisins
nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(x_test_transformed)

#-------------------------- liste des vaiables pertinentes----------------------------------
listeInfos = ['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN',
              'AMT_INCOME_TOTAL',
 'AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',"REG_CITY_NOT_LIVE_CITY",'REG_REGION_NOT_LIVE_REGION',
  'REG_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_WORK_CITY','LIVE_REGION_NOT_WORK_REGION','CNT_FAM_MEMBERS',
              'LIVE_CITY_NOT_WORK_CITY',
'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_YEAR']

#---------------------------- selection de vaeiable à afficher ------------------------------------
app_test = app_test[listeInfos]
#------------------------------------------------------------------------------------------------------
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


#-------------------------------------- table des client similaire-----------------------------------



#-------------------------------------------------------------------------------------
variables = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
                 'CNT_FAM_MEMBERS' ,'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3', "AMT_REQ_CREDIT_BUREAU_YEAR" ]
################################################################################################################""
########################################################################################################################"
app = Dash(__name__)
# definition de la colone vertebrale du dashboad avzec dcc qui permet de creer des onglers
id_dropdown = dcc.Dropdown( id='id_client',
                           options=x_test_transformed.index,
                            value=x_test_transformed.index[0],
                           multi = True
                          )
## Définition de loyout de app

app.layout = html.Div([
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
            'width': "25%", # 25 % de la page
            "border": '1px solid #eee',
            'padding': '30px 30px 30px 120px',
            'box-shadow': '0 2px 2px #ccc',
             'display': 'inline_block',
            'verticalAlign': 'top'
            
        }), 
#----------------------------------------------------------------------------------------------    
    #On va definir les onglet à farire apparaitre sur le tableau de bord
    html.Div([
        dcc.Tabs(id ='tab', value ='tab-1', # les children sont les enfants de de Tabs 
                children = [
           ###################################################################################################         
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
                            'paddingTop':'50px'} ),
                        
                        
                        html.Div([
                            dcc.Graph(id = 'pie_infos')
                            
                            
                        ], style= {'display':'inline-block', 
                                   'verticalAlign':'top',
                                   'width': '45%',
                                  'paddingTop':'50px'}),
                    
                                                                ],
                           ),
        ################################################################################################
                    # 2 ieme onglet: Score de probabilté de faillite
                    dcc.Tab(label = 'Situation du client', children =[
        #****************************************************************************************       
                html.Div([html.H3("Situation de faillite, variables d'influences et clients similaires")], style ={'background': 'blue',
                                                                                             "color": "white", 'textAlign':'center',
                                                                                    'padding':'10px 0px 10px 0px'}),
                        html.Div([
                    dcc.Graph(id = 'proba_value')
                     ], style = {'border': '1px solid #ccc', 
                                'box-shadow': '0 2px 2px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '45%',
                                 'padding': '50px 0px 0px 50px'
                                }),
                        #____________________________________________________________
                        html.Div([
                    dcc.Graph(id = 'graph_lime')
                     ], style = {'border': '1px solid #ccc', 
                                'box-shadow': '0 2px 2px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '45%',
                                 'padding': '50px 0px 0px 50px'
                                }), 
                        #____________________________________________________________
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
                          
                        
                                  
                  #--------------------liste deroulante de toute les variable de la table  client simil----------------------
                              #non fait 
      #-------------------------------------- Titre -------------------------------------------
                        
                  html.Div([  
            
                                     html.Div([html.H3("Comparaison du client choisi par rapport à ses voisins")],
                                              style ={'background': 'blue',
                                                        "color": "white", 'textAlign':'center',
                                                        'padding':'10px 0px 10px 0px'}),
#------------------------------------------------------- Grphe 1---------------------------------------------

                                ]),
                        html.Div([
            dcc.Graph(id = 'graph_comp_1')
                    ], style = {'border': '1px solid #ccc', 
                                'box-shadow': '0 2px 2px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '30%',
                                 'padding': '50px 0px 0px 50px'} ), 
   #------------------------------------------------------fin graphe ------------------------------------------------------
                        
 #------------------------------------------------------- Grphe 2---------------------------------------------

                        html.Div([
            dcc.Graph(id = 'graph_comp_2')
                    ], style = {'border': '1px solid #ccc', 
                                'box-shadow': '0 2px 2px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '30%',
                                 'padding': '50px 0px 0px 50px'} ), 
   #------------------------------------------------------fin graphe ------------------------------------------------------
 #------------------------------------------------------- Grphe 2---------------------------------------------

                        html.Div([
            dcc.Graph(id = 'graph_comp_3')
                    ], style = {'border': '1px solid #ccc', 
                                'box-shadow': '0 2px 2px #ccc',
                                 'display': 'inline-block',
                                 'verticalAlign': 'top',
                                 'width': '25%',
                                 'padding': '50px 0px 0px 50px'} ), 
   #------------------------------------------------------fin graphe ------------------------------------------------------
                        
                                                           ]),

                                 
   
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

                                                                 ],
                                ),
               
                       
      #********************************************************************************************************************              
                    
                     # 3 ième onglet: Tableau fonction
                    dcc.Tab(label = 'Fonction coût'),
                    #4 iéme onglet : Exploration des données
                    dcc.Tab(label = 'Exploration des données'),
                    
                ])
                
        
    ])
   
])
#*************************************************************************************************************
#******************************************************    *******************************************************
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
#*******************************************       ***************************************************
#**************************************************************************************************************








#--------------------------------------------------------------------------------------
# Definition fonction de rappel callback

#----------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

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
@app.callback(Output('pie_infos', 'figure'),
              [Input('id_client', 'value')])
def graph_pie(id_client):
    col_pie = ['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
                 'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS']
    data_info = [float(app_test[app_test.index ==id_client][col].iloc[0]) for col in col_pie]
    labels =  col_pie
    traces = [
        go.Pie(labels= labels, values= data_info)
    ]
    del data_info
    return {
        'data': traces,
        'layout': go.Layout(
            title= 'Information client',
            legend_orientation= 'h'
        )
    }

############################ probabilté de faillite #######################################################""

@app.callback(Output('proba_value', 'figure'),
             [Input('id_client', 'value')])
def update_proba_bar(id_client):
    proba = pipe_model[3].predict_proba(np.array(x_test_transformed.loc[id_client]).reshape(1, -1)).flatten()
    # Retourne le bar plot mis à jour pour l'id client
    df = ['Faillite', "Non faillite"]
    fig = px.bar(x = df, y= proba,
             color=['Non faillite', 'Faillite'],
             labels=dict(x='Classes', y='Score proba'),
             title='Probabilité de faillite', text_auto=True)
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
                    marker_color=list(data.positive.map({True: 'blue', False: 'red'}).values)
        )]
    return {
        
        'data': traces,
        
        'layout': go.Layout(
                            margin=dict(l=300, r=0, t=30, b=100),
            title = "Influences des variables(Rouge pour Non faillite et bleu pour faillite"
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
             [Input('id_client', 'value')])
def graph_comparaison_1(id_client):
    indices_similary_clients = nbrs.kneighbors(np.array(x_test_transformed.loc[id_client]).reshape(1, -1))[1].flatten()
    df_client_simil = app_test.iloc[indices_similary_clients]
 
    ind_client_vs = indices_similary_clients.tolist()
    
    data_client_princi = df_client_simil[df_client_simil.index == id_client]["AMT_INCOME_TOTAL"].iloc[0]
    data_client_1 = df_client_simil[df_client_simil.index  == ind_client_vs[1]]["AMT_INCOME_TOTAL"].iloc[0]
    data_client_2 = df_client_simil[df_client_simil.index  == ind_client_vs[2]]["AMT_INCOME_TOTAL"].iloc[0]
    data_client_3 = df_client_simil[df_client_simil.index  == ind_client_vs[3]]["AMT_INCOME_TOTAL"].iloc[0]
    data_client_4 = df_client_simil[df_client_simil.index  == ind_client_vs[4]]["AMT_INCOME_TOTAL"].iloc[0]

    labels = ['Client choisi', 'Voisin_1', "Voisin_2", "Voisin_3", "Voisin_4"]
    values = [float(data_client_princi), float(data_client_1),float(data_client_2), float(data_client_3),
              float( data_client_4) ]
    
    total = sum(values)

    traces = [
        go.Pie(labels= labels, values= values)
    ]

    return {
        'data': traces,
        'layout': go.Layout(
        title= "Infos du client choisi comparée <br> à ses +  4 proches voisins  (Total: " + str(total) + ")",
            legend_orientation= 'h'
        )
    }
     ################################### graphe_comp_2 ###################### 
@app.callback(Output('graph_comp_2', 'figure'),
             [Input('id_client', 'value')])
def graph_comparaison_2(id_client):
    indices_similary_clients = nbrs.kneighbors(np.array(x_test_transformed.loc[id_client]).reshape(1, -1))[1].flatten()
    df_client_simil = app_test.iloc[indices_similary_clients]
 
    ind_client_vs = indices_similary_clients.tolist()
    
    data_client_princi = df_client_simil[df_client_simil.index == id_client]["AMT_CREDIT"].iloc[0]
    data_client_1 = df_client_simil[df_client_simil.index  == ind_client_vs[1]]["AMT_CREDIT"].iloc[0]
    data_client_2 = df_client_simil[df_client_simil.index  == ind_client_vs[2]]["AMT_CREDIT"].iloc[0]
    data_client_3 = df_client_simil[df_client_simil.index  == ind_client_vs[3]]["AMT_CREDIT"].iloc[0]
    data_client_4 = df_client_simil[df_client_simil.index  == ind_client_vs[4]]["AMT_CREDIT"].iloc[0]

    labels = ['Client choisi', 'Voisin_1', "Voisin_2", "Voisin_3", "Voisin_4"]
    values = [float(data_client_princi), float(data_client_1),float(data_client_2), float(data_client_3),
              float( data_client_4) ]
    
    total = sum(values)

    traces = [
        go.Pie(labels= labels, values= values)
    ]

    return {
        'data': traces,
        'layout': go.Layout(
        title= "AMT_CREDIT du client choisi  comparé <br> à ses + proches voisins  (Total: " + str(total) + ")",
            legend_orientation= 'h'
        )
    }

     ################################### graphe_comp_3 ###################### 
@app.callback(Output('graph_comp_3', 'figure'),
             [Input('id_client', 'value')])
def graph_comparaison_2(id_client):
    indices_similary_clients = nbrs.kneighbors(np.array(x_test_transformed.loc[id_client]).reshape(1, -1))[1].flatten()
    df_client_simil = app_test.iloc[indices_similary_clients]
 
    ind_client_vs = indices_similary_clients.tolist()
    
    data_client_princi = df_client_simil[df_client_simil.index == id_client]["AMT_GOODS_PRICE"].iloc[0]
    data_client_1 = df_client_simil[df_client_simil.index  == ind_client_vs[1]]["AMT_GOODS_PRICE"].iloc[0]
    data_client_2 = df_client_simil[df_client_simil.index  == ind_client_vs[2]]["AMT_GOODS_PRICE"].iloc[0]
    data_client_3 = df_client_simil[df_client_simil.index  == ind_client_vs[3]]["AMT_GOODS_PRICE"].iloc[0]
    data_client_4 = df_client_simil[df_client_simil.index  == ind_client_vs[4]]["AMT_GOODS_PRICE"].iloc[0]

    labels = ['Client choisi', 'Voisin_1', "Voisin_2", "Voisin_3", "Voisin_4"]
    values = [float(data_client_princi), float(data_client_1),float(data_client_2), float(data_client_3),
              float( data_client_4) ]
    
    total = sum(values)

    traces = [
        go.Pie(labels= labels, values= values)
    ]

    return {
        'data': traces,
        'layout': go.Layout(
        title= "AMT_GOODS_PRICE du client choisi  comparé <br> à ses + proches voisins  (Total: " + str(total) + ")",
            legend_orientation= 'h'
        )
    }

##############################################################################################################################







server =  app.server  
if __name__ == "__main__":
    app.run_server(debug = True) 