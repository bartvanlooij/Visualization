import pandas as pd
from dash import Dash, html, dcc, Input, Output, State


df = pd.read_csv('data/airbnb_open_data.csv')
app = Dash(__name__)

app.layout = html.Div(id='main', children=[
    html.H1(id='title', children='Air BnB'),
    html.Div(id='graph_container', children=
    [
        dcc.Graph(id='main_graph'),
        dcc.Graph(id='second_graph')
    ], style={'display': 'flex', 'flex-direction': 'row'})
]
)