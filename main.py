from ctypes.wintypes import POINT
from gettext import translation
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import json
import geopandas as gpd
import shapely
from shapely.geometry import Point
import numpy as np
from operator import attrgetter
with open('data/nyc.geojson') as f:
    geo_json = json.load(f)
df = pd.read_csv('data/airbnb_open_data.csv', low_memory=False, index_col=0)
df_plot = pd.read_csv('data/apparments_per_region.csv',
                      low_memory=False, index_col=0)
df_plot['geometry'] = df_plot['geometry'].apply(lambda x: shapely.wkt.loads(x))
df_sub = gpd.read_file('data/Subway Lines.geojson')
df_attraction = gpd.read_file('data/attraction_point.geojson')
app = Dash(__name__)

lats = []
lons = []
names = []

for feature, name in zip(df_sub.geometry, df_sub.name):
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    else:
        continue
    for linestring in linestrings:
        x, y = linestring.xy
        lats = np.append(lats, y)
        lons = np.append(lons, x)
        names = np.append(names, [name]*len(y))
        lats = np.append(lats, None)
        lons = np.append(lons, None)
        names = np.append(names, None)


app.layout = html.Div(id='main', children=[
    html.H1(id='title', children='Air BnB'),
    html.Div(id='graph_container', children=[
        dcc.Graph(id='main_graph', animate=True),
        html.Label(id='option_label', children=['Graph options']),
        html.Div(id='graph_options_container', children=[
            dcc.RangeSlider(id='price_slider', value=[df.price.min(), df.price.max()], min=df.price.min(),
                            max=df.price.max(), tooltip={"placement": "bottom", "always_visible": True}),
            dcc.Checklist(id='graph_options', options=[
                'New York districts', 'Public transit', 'Tourist attractions'],labelStyle={'display': 'block'}, value=['New York districts', 'Public transit', 'Tourist attractions']),
            
            html.Details(id='cancellation_summary', children=[
                html.Summary('Cancellation policy'),
                dcc.Checklist(style={'margin-left' : '15px'},id='cancellation',labelStyle={'display': 'block'}, options=df.cancellation_policy.unique(
            ).tolist(), value=df.cancellation_policy.unique().tolist())
            ])
            ,
            html.Details(
                id='room_type_details', children=[
                    html.Summary('Room type'),
                    dcc.Checklist(style={'margin-left' : '15px'},id='room_type_checklist',labelStyle={'display': 'block'}, options=df['room type'].unique().tolist(), value=df['room type'].unique().tolist())
                ]
            ),
            html.Details(id='num_days_to_book', children=[
                html.Summary('Number of nights you want to book'),
                daq.NumericInput(value=1, id='number_of_days_input', max=1000)
            ]),
            html.Details(id='average_review_details', children=[
                html.Summary('Average review'),
                dcc.Slider(id='review_slider', min=0, max=5, value=0,tooltip={"placement": "bottom", "always_visible": True})
            ]),
            html.Details(id='instant_bookable_details', children=[
                html.Summary('Immediately available'),
                dcc.Checklist(style={'margin-left' : '15px'},id='available_check', labelStyle={'display': 'block'}, options=['Instantly avaible'], value=[])
            ]),
            html.Details(id='service_fee_details', children=[
                html.Summary('Maximum service fee'),
                dcc.Slider(id='service_fee_slider', min=df['service fee'].min(), max=df['service fee'].max(), value=df['service fee'].max(),tooltip={"placement": "bottom", "always_visible": True} )
            ])
            
            



        ]
        )

    ]),
    html.Div(id='sub_graph_container'),
    html.Div(id='debug'),
    html.Div(id='debug2')
]
)


@app.callback(
    Output('main_graph', 'figure'),
    Output('debug', 'children'),
    Input('graph_options', 'value'),
    Input('price_slider', 'value'),
    Input('cancellation', 'value')

)
def update_graph(graph_options, price_range, cancellation):
    go_fig = go.Figure()
    df_appartments_true = df[(df['price'] >= price_range[0]) & (
        df['price'] <= price_range[1]) & (df['cancellation_policy'].isin(cancellation))]

    if 'New York districts' in graph_options:
        df_plot['appartment_count'] = df_plot['id'].map(
            dict(df_appartments_true.region.value_counts()))
        second_option = go.Choropleth(geojson=geo_json, locations=df_plot.id, z=df_plot.appartment_count, colorscale="Viridis",
                                      zmin=df_plot.appartment_count.min(), zmax=df_plot.appartment_count.max(), marker_line_width=0, hoverinfo='none')
        go_fig.add_trace(second_option)
        go_fig.update_geos(fitbounds="locations")
        go_fig.update_layout(mapbox_style="carto-positron",
                             mapbox_zoom=10, mapbox_center={"lat": 40.64964395581123, "lon": -73.828093870371374}, geo_scope='usa')

    if 'Public transit' in graph_options:
        go_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        fig_2 = go.Scattergeo(lat=lats, lon=lons,
                              mode='lines', hoverinfo='skip')
        go_fig.add_trace(fig_2)
        go_fig.update_geos(fitbounds="locations")
        go_fig.update_layout(mapbox_style="carto-positron",
                             mapbox_zoom=10, mapbox_center={"lat": 40.64964395581123, "lon": -73.828093870371374}, geo_scope='usa')

    if 'Tourist attractions' in graph_options:

        fig_3 = go.Scattergeo(lon=df_attraction['geometry'].map(attrgetter(
            'x')), lat=df_attraction['geometry'].map(attrgetter('y')), text=df_attraction['name'],
            marker=dict(color='black'))

        go_fig.add_trace(fig_3)
    go_fig.update_layout(transition={'duration': 500})
    return go_fig, f'{cancellation}, {graph_options}, {price_range[0]}, {price_range[1]}'


@app.callback(
    Output('sub_graph_container', 'children'),
    Output('debug2', 'children'),
    Input('main_graph', 'clickData'),
    prevent_initial_call=True
)
def on_graph_click(clickdata):
    region = clickdata['points'][0]['location']
    go_fig = go.Figure()
    df_subregion = df[df['region'] == region]
    fig_3 = go.Scattergeo(lon=df_subregion['long'], lat=df_subregion['lat'], text=df_subregion['NAME'],
                          marker=dict(color='black'))
    go_fig.add_trace(fig_3)
    go_fig.update_geos(fitbounds="locations")
    go_fig.update_layout(mapbox_style="carto-positron")

    children = [html.H2(region), dcc.Graph(figure=go_fig)]
    return children, region


if __name__ == '__main__':
    app.run_server(debug=True)
