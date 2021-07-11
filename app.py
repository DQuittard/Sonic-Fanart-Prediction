# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, ALL

import plotly.express as px
import plotly.graph_objects as go

from lenspy import DynamicPlot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import os
import seaborn as sns

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

import sklearn.metrics as skm

from sklearn.model_selection import learning_curve

from sklearn.model_selection import validation_curve

import platform

import base64

# metrics are used to find accuracy or error 
from sklearn import metrics 

import pickle

#import dash_bootstrap_components as dbc

import time

from data.check import zip_generator
import zipfile

from natsort import natsorted
from varname.helpers import Wrapper
from varname import nameof

import re


###########################################################################
#Style#

#external_stylesheets = ['style.css']
#external_stylesheets = ['https://www.w3schools.com/w3css/4/w3.css']
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,suppress_callback_exceptions=True,prevent_initial_callbacks=True)

server = app.server

app.title = 'Sonic Fanart RandomForest prediction'

###########################################################################
#Regular Python#

#os check
#read the data
if platform.system()=="Windows":
    main_path = os.getcwd()+'\\'
elif platform.system()=="Linux":
    main_path = os.getcwd()+'/'

if platform.system()=="Windows":
    img_path = main_path +'\\img\\'
    data_path = main_path + '\\data\\'
elif platform.system()=="Linux":
    img_path = main_path+'/img/'
    data_path = main_path+'/data/'


#Dataframe
df = pd.read_csv(data_path+"sonic.csv")
df.rename( columns={'Unnamed: 0':'Id'}, inplace=True )

#Figure test
fig = px.scatter(df, x="views", y="comments", color="year")

if platform.system()=="Windows":
    img_path = main_path +'\\img\\'
elif platform.system()=="Linux":
    img_path = main_path+'/img/'


#Image test
#image_filename = main_path+"526729.png" 
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#image_filename2 = img_path+"FanartSonic_1.png" 
#encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
#encoded_image2 = encoded_image


###CODE###

#5) PASSAGE CATEGORIQUE

#fonctionne que pour les features catégoriques 
#pd.get_dummies(df)

#less than 6000, 6000 to 230000 and more than 23000


'''

# using metrics module for accuracy calculation 
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


'''


###########################################################################
#OTHER FUNCTIONS

def generate_thumbnail(df,encoded_image,number):
    return html.Div([
        html.A([
            html.Img(
                src = 'data:image/png;base64,{}'.format(encoded_image.decode()),
                style = {
                    'height': '40%',
                    'width': '40%',
                    'float': 'left',
                    'position': 'relative',
                    'padding-top': 0,
                    'padding-right': 0
                }
            )
        ], href = df['url'][number]),
    ])

'''    
files = os.listdir(img_path)
file_number = len(files)

images_div = []
for i in (0,file_number-1):
    encoded_image = base64.b64encode(open(img_path+files[i], 'rb').read() )
    images_div.append(generate_thumbnail(df,encoded_image,i))
'''

###########################################################################
#APP LAYOUT#

app.layout = html.Div(children=[
    html.Img(src=app.get_asset_url("dash-logo.png"),
             style = {
                                'height': '35%',
                                'width': '35%',
                                'float': 'left',
                                'position': 'relative',
                                'padding-top': 0,
                                'padding-right': 10,
                                'padding-bottom':15
                            }
             ),
    
    html.H3(children='Sonic Fanart prediction'),
    
    
    
 
    html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='⌨️ 1-Inputs', value='tab-1',children=[
            html.Div(
                [
            html.H6("Proportion of data used to train a model (by default 0.7/70%):"),
            html.Div(
            ["Input: ",
              dcc.Input(id='training',value=0.7, type='text')
                  ]
                    ),
            
                ]    
                ),
            
            html.Div([
            html.H6("Select the minimum and maximum number of views required so a fanart is considered popular or not (or leave it to default for the quantiles)."),
            html.H6("Minimal number of views"),
            html.Div(["Input: ",
                      dcc.Input(id='quantile_25', value=df.quantile(0.25)[3], type='text')]),
            html.Br(),
            
            html.H6("Maximum number of views"),
            html.Div(["Input: ",
                      dcc.Input(id='quantile_75', value=df.quantile(0.75)[3], type='text')]),
            html.Br(),
            ]),
            
            html.H6("See how the score changes for a different parameter."),
            
            dcc.Dropdown(
            options=[
                {'label': 'Maximum number of leaf nodes', 'value': 'max_leaf_nodes'},
                {'label': 'Number of trees', 'value': 'n_estimators'},
                {'label': 'Maximum depth of each tree', 'value': 'max_depth'}
            ],
            value='max_leaf_nodes',
            id='param',
            placeholder="Select a parameter to see"
            ),
            
            
            ]
            ),
        dcc.Tab(label='📂 2-Data', value='tab-2',children=[
            html.Div(
                [   
            html.H3("Datatable:"),
            html.Button("Display table", id="btn_table"),
            dcc.Loading(
            id="loading_table",
            type="default",
            children=html.Div(id="table2")
                        ),
             
                ]
                )
            ]
            ),
        dcc.Tab(label='📉 3-Results', value='tab-3',children=[
            html.Div(
                    [
            html.Button("Display curves", id="btn_curves"),
            dcc.Loading(
            id="loading-2",
            type="default",
            children=dcc.Graph(
                id='learning_curve'
                            )
                        ),
    
        dcc.Loading(
                id="loading-3",
                type="default",
                children=dcc.Graph(
                    id='validation_curve'
                    )
                    ),
        html.Br(),
        html.Div(["Accuracy of the model: ",
              html.Div(id='accuracy')]),
        
        html.Br(),
        html.Hr(),
        
        
        html.P("Put the numbers you want to use for the prediction. Assure that both inputs are the same length. Separate them by commas. For example: 200,500,600."),
        html.H6("Input favorites:"),
        dcc.Input(id='select1', type='text',value='0',debounce=True),
        html.Br(),
        html.H6("Input comments:"),
        dcc.Input(id='select2', type='text',value='0',debounce=True),
        html.Br(),
        html.H6("Choose the parameter you want to see:"),
        dcc.Dropdown(
            options=[
                {'label': 'Favorites', 'value': 'favs'},
                {'label': 'Comments', 'value': 'comments'},
            ],
            value='favs',
            id='param2',
            placeholder="Select a parameter to see"
            ),
        
        
        html.Button("Display prediction", id="btn_strip"),
            dcc.Loading(
            id="loading-10",
            type="default",
            children=dcc.Graph(
                id='stripplot'
                            )
                        )
                    ]
                    )
            
            ]
                ),
        dcc.Tab(label='📷 4-Gallery', value='tab-4',children=[
            html.Div([
        html.H6('Enter beginning and ending number. This will download all the fan arts in-between those (~4 seconds to download one picture).'),
        dcc.Input(id='inf_dl', type='text',value='0',debounce=True),
        dcc.Input(id='sup_dl', type='text',value='2',debounce=True),
        #html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),style={'height':'10%', 'width':'10%'}),
        dcc.Loading(
            id="loading_images1",
            type="default",
            children= html.Div(id='text_dl')
                        ),
        html.Button("1) Download images to app", id="btn_dl", n_clicks=0),
        
        #html.Div(images_div)
        html.Br(),
        html.Button("Optional: Download images as zip file", id="btn_image"), 
        html.Br(),
        dcc.Download(id="download-image"),
        html.Button("2) Display images", id="add-filter", n_clicks=0),
        html.Br(),
        html.Div(id='dropdown-container', children=[]),
        html.Br(),
        
        
        #html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()),style={'height':'10%', 'width':'10%'})
        
        ]
        ),
       
        
            ]
            )
    ]
    ),
    
    
    html.Div(id='tabs-example-content')
    ])

    
    
    
    #html.Div(dcc.Input(id='input-on-submit', type='text')),
    
    
    
])

'''
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    '''
             
###########################################################################
#CALLBACKS#

@app.callback(
    Output('table2', 'children'),
    Input("btn_table", "n_clicks"),
    State(component_id='quantile_25', component_property='value'),
    State(component_id='quantile_75', component_property='value')
    
    )

def update_df(n_clicks,quant_25,quant_75):
    
    if os.path.exists(os.path.join(data_path,'dash_table.csv')):
        os.remove(os.path.join(data_path,'dash_table.csv'))
    
    df2 = pd.read_csv(data_path+"sonic.csv",sep=',')
    
    q25 = int(quant_25)
    #q50 = df.quantile(0.5)[3]
    q75 = int(quant_75)

    quant_25 = int(quant_25)
    quant_75 = int(quant_75)

    df['not many views']=0
    df['regular number views']=0
    df['lot of views']=0
    
    #categorical
    index = df[df['views']<=quant_25].index
    print(len(index))
    df.loc[index,'not many views'] = 1
    index = df[(df['views']>quant_25) & (df['views']<=quant_75)].index
    print(len(index))
    df.loc[index,'regular number views'] = 1
    index = df[df['views']>quant_75].index
    print(len(index))
    df.loc[index,'lot of views'] = 1
    
    #print(df)
    
    output = ['not many views','regular number views','lot of views']
    
    df['class']=''
    
    index = df[df['not many views']==1].index
    df.loc[index,'class'] = output[0]
    index = df[df['regular number views']==1].index
    df.loc[index,'class'] = output[1]
    index = df[df['lot of views']==1].index
    df.loc[index,'class'] = output[2]
    
    #print(df)
    
    data=df.to_dict('rows')
    columns =  [{"name": i, "id": i} for i in (df.columns)]
    
    #print(columns)
    #print("------------------------------------------------------\n")
    
    df.to_csv(data_path+'dash_table.csv', index = False)
    
    
    #print("df saved "+ str(n_clicks) + " times.")
    #dash_table.DataTable(data=data, columns=columns)
    
    return dash_table.DataTable(
            #fixed_rows={'headers': True},
            style_table={'height': '400px','overflowY': 'auto', 'overflowX':'auto', 'width':'auto'},
            style_cell_conditional=[
            {'if': {'column_id': 'tags',},
                'display': 'None',}],
            style_header={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'width':'auto',
            },
           style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            #'maxWidth': 0
            },
            columns=columns,
            data=data
            )



@app.callback(Output("loading-output-1", "children"), Input("training", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value


@app.callback(
    Output("learning_curve", "figure"),
    Output("validation_curve", "figure"),
    Output('accuracy','children'),
    Input("btn_curves", "n_clicks"),
    State(component_id='training',component_property='value'),
    State('param','value')
    )

def curves(n_clicks,train_test_size,param):

    #6) RANDOMFOREST CATEGORICAL
    
    #time.sleep(1)
    df = pd.read_csv(data_path+"dash_table.csv",sep=',')
    
    train_test_size = float(train_test_size)
    
    #x = np.array(df[['not many faves','regular number faves','lot of faves','not many comments','regular number comments','lot of comments']])
    x = np.array(df[['faves','comments']])
    
    y = np.array(df['class'])
    
    
    #######
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = train_test_size) 
    
    
    param_name = param#"max_leaf_nodes" #for validation_curve
    
    n_estimators = 350
    max_leaf_nodes = 20
    max_depth = 100
    max_samples = 0.7
    n_jobs = -1
    
    names_param = [nameof(n_estimators),nameof(max_leaf_nodes),nameof(max_depth)]
    vec_param = [n_estimators,max_leaf_nodes*5,max_depth*2]
    
    upper_limit = 0
    
    for i in range(0,len(vec_param)):
        if(names_param[i]==param_name):
            upper_limit = vec_param[i]
    
    # creating a RF classifier 
    clf = RandomForestClassifier(n_estimators = n_estimators, max_leaf_nodes = max_leaf_nodes, max_depth=max_depth,bootstrap=True, max_samples=max_samples,n_jobs=n_jobs)   
    
    # Training the model on the training dataset 
    # fit function is used to train the model using the training sets as parameters 
    clf.fit(x_train, y_train) 
      
    # performing predictions on the test dataset 
    y_pred = clf.predict(x_test) 
    
    #LEARNING CURVE
    train_sizes = np.linspace(0.1, 1, 10) # 0.1, 0.2, 0.3, ..., 0.9, 1.0
    
    train_sizes, train_scores, test_scores = learning_curve(estimator=clf, X = x_train, y = y_train, 
                                                            train_sizes = train_sizes, cv = 10,n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_scores_mean,
        name="Training score"
    ))
    
    
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_scores_mean,
        name="Test score"
    ))
    
    fig.update_layout(
        title="Learning curves",
        xaxis_title="Training set size",
        yaxis_title="Score",
        legend_title="Score",
    )
    
    ###FIGURE 2
    
    param_range=np.arange(2,upper_limit,10)
    
    
    train_scores, test_scores = validation_curve(
        clf, x, y, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    
    
    
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=param_range,
        y=train_scores_mean,
        name="Training score",
        #stackgroup = 'one',
        line=dict(color='rgb(26,150,65)')
    ))
    
    fig2.add_trace(go.Scatter(
        x=param_range,
        y=test_scores_mean,
        name="Cross validation score",
        line=dict(color='rgb(200,50,0)')
        #stackgroup = 'two'
    ))
    
    fig2.add_trace(go.Scatter(
        x=param_range,
        y=train_scores_mean - train_scores_std,
        showlegend=False,
        line=dict(color='rgb(100,200,25)')
    ))
    
    
    fig2.add_trace(go.Scatter(
        x=param_range,
        y=train_scores_mean + train_scores_std,
        #showlegend=False,
        name="Training score +/- std",
        fill='tonexty', #fill to previous y added on the figure,
        fillcolor='rgba(100,200,25,0.2)',
        line=dict(color='rgb(100,200,25)')
    ))
    
    
    fig2.add_trace(go.Scatter(
        x=param_range,
        y=test_scores_mean - test_scores_std,
        showlegend=False,
        line=dict(color='rgb(200,100,0)')
    ))
    
    
    fig2.add_trace(go.Scatter(
        x=param_range,
        y=test_scores_mean + test_scores_std,
        #showlegend=False,
        name="Cross validation score +/- std",
        fill='tonexty', #fill to previous y added on the figure,
        fillcolor='rgba(200,100,0,0.2)',
        line=dict(color='rgb(200,100,0)')
    ))
    
    
    fig2.update_layout(
        title="Validation curve",
        xaxis_title=param_name,
        yaxis_title="Score",
        legend_title="Score",
    )
    
    plot = DynamicPlot(fig)
    plot2 = DynamicPlot(fig2)
    
    model_file = data_path + "model.pkl"
    with open(model_file,'wb') as file:
        pickle.dump(clf,file)
    

    return plot.fig,plot2.fig, [html.H6(children=metrics.accuracy_score(y_test, y_pred,normalize=True))]


@app.callback(
    #[Output("progress", "value"), Output("progress", "children")],
    Output(component_id='text_dl', component_property='children'),
    Input("btn_dl", "n_clicks"),
    #Input("inf_dl", "value"),
    #Input("sup_dl", "value"),
    state=[
    State('inf_dl', 'value'),
    State('sup_dl', 'value')],
    prevent_initial_call=True
)

def downloading_back(n_clicks,inf,sup):
        
    '''if os.path.isfile(img_path+'sonic_images.zip'):
        os.remove(img_path+'sonic_images.zip')
    '''
    
    if (n_clicks > 0):
        zip_generator(df,main_path,int(inf),int(sup))
        number = int(sup)-int(inf)
        text = 'Downloaded ' + str(number) + " fanarts."
        
    n_clicks = 0

    
    return (text)




@app.callback(
    Output('dropdown-container', 'children'),
    Input('add-filter', 'n_clicks'),
    Input("inf_dl", "value"),
    State('dropdown-container', 'children'),
    prevent_initial_call=True
    )
def display_dropdowns(n_clicks, inf, children):
    '''new_dropdown = dcc.Dropdown(
        id={
            'type': 'filter-dropdown',
            'index': n_clicks
        },
        options=[{'label': i, 'value': i} for i in ['NYC', 'MTL', 'LA', 'TOKYO']]
    )'''

    
    
    files = os.listdir(img_path)
    files=natsorted(files)
    file_number = len(files)-1
    
    images = []
    
    
    if(n_clicks > 0):
        for i in range(0,file_number):
            if os.path.exists(os.path.join(img_path,files[i])):
                index_ext = files[i].rfind('.')
                extension = files[i][index_ext:len(files[i])]
                
                if extension != '.mp4':
                    encoded_image = base64.b64encode(open(img_path+files[i], 'rb').read() )
                
                    new_Image = html.Div([
                    html.A([
                        html.Img(
                            id={
                            'index': i
                            },
                            src = 'data:image/png;base64,{}'.format(encoded_image.decode()),
                            style = {
                                'height': '10%',
                                'width': '10%',
                                'float': 'left',
                                'position': 'relative',
                                'padding-top': 0,
                                'padding-right': 0
                            }
                        )
                    ], href = df['url'][int(inf)+i]),
                    ])
                    
    
                    #children.append(new_Image)
                    
                
                else:
                    encoded_video = base64.b64encode(open(img_path+files[i], 'rb').read() )
                    
                    new_Image = html.Div([
                    html.A([
                        html.Video(
                            id={
                            'index': i
                            },
                            src='data:video/mp4;base64,{}'.format(encoded_video.decode()),
                            controls = True,
                            style = {
                                'height': '10%',
                                'width': '10%',
                                'float': 'left',
                                'position': 'relative',
                                'padding-top': 0,
                                'padding-right': 0
                            }
                        )
                    ], href = df['url'][int(inf)+i]),
                    ])
                    
                images.append(new_Image)
                    
        n_clicks=0
        
    return images


@app.callback(
    Output("download-image", "data"),
    Input("btn_image", "n_clicks"),
    prevent_initial_call=True,
)

def downloading_front(n_clicks):
    return dcc.send_file(
        img_path+"sonic_images.zip"
    )


@app.callback(
    Output('stripplot', "figure"),
    Input("btn_strip", "n_clicks"),
    state=[
    State('select1', 'value'),
    State('select2', 'value'),
    State('param2','value')],
    prevent_initial_call=True
)

def display_prediction(n_clicks, favorites, comments, param2):
    
    if (n_clicks > 0):
        print('entering if')
        model_file = data_path + "model.pkl"
        with open(model_file,'rb') as file:
            model = pickle.load(file)
        

        param_name2 = param2#"max_leaf_nodes" #for validation_curve
    
    
        #print("before nameof")
        names_param2 = [nameof(favorites),nameof(comments)]
        #print("after nameof")
        
        for i in range(0,len(names_param2)):
            if(names_param2[i]==param_name2):
                indices = i
            else:
                indices = 0
       
        
        temp1 = re.findall(r'[\d\.\d]+', favorites)
        res1 = list(map(float,temp1))
        res1 = list(map(int,res1))
        temp2 = re.findall(r'[\d\.\d]+', comments)
        res2 = list(map(float,temp2))
        res2 = list(map(int,res2))
        
        test3 = res1 + res2
        n_features = 2
        
        test3 = np.array(test3).reshape( -1 * int(len(test3) / n_features ), int(len(test3) / n_features) )
        test3 = test3.transpose()
        
        result = model.predict(test3)
        
        
        
        fig3 = px.strip(x=result, y=test3[:,indices],color=result,title="Predicting the class according to the number of " + names_param2[indices] ,labels={'x': 'class','y':names_param2[indices]})
        #fig3 = px.strip(x=result, y=test3[:,1],color=result,title="Predicting the class",labels={'x': 'class'})

        plot3 = DynamicPlot(fig3)
        
    
    n_clicks = 0
    
    
    return plot3.fig

###########################################################################
#LAUNCH#


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port='8080',debug=False)