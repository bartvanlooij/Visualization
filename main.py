from ctypes.wintypes import POINT
from gettext import translation
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
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
        dcc.Checklist(id='graph_options', options=[
                      'New York districts', 'Public transit', 'Tourist attractions'], value=['New York districts', 'Public transit', 'Tourist attractions'], inline=True, )
    ]),
    html.Div(id='sub_graph_container')
]
)


@app.callback(
    Output('main_graph', 'figure'),
    Input('graph_options', 'value')

)
def update_graph(graph_options):
    go_fig = go.Figure()
    if 'New York districts' in graph_options:
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
    return go_fig


@app.callback(
    Output('sub_graph_container', 'children'),
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
    return children


if __name__ == '__main__':
    app.run_server(debug=True)
