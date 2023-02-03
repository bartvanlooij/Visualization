from cgitb import text
from ctypes.wintypes import POINT
from gettext import translation
from turtle import color, heading
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import json
import geopandas as gpd
import shapely
import numpy as np
with open('data/nyc.geojson') as f:
    geo_json = json.load(f)
df = pd.read_csv('data/airbnb_open_data.csv', low_memory=False, index_col=0)
df['size'] = 1
list_colors = ['blue', 'red', 'green', 'purple']
mapping_collors = {x: list_colors[index] for index,
                   x in enumerate(df['room type'].unique().tolist())}
df['app_type_color'] = df['room type'].map(mapping_collors)
df_plot = pd.read_csv('data/apparments_per_region.csv',
                      low_memory=False, index_col=0)
df_plot['name'] = df_plot['id'].str.replace("_", " ")
df_plot['geometry'] = df_plot['geometry'].apply(lambda x: shapely.wkt.loads(x))
df_plot['centroid'] = df_plot['geometry'].apply(
    lambda x: list(x.centroid.coords[0]))

df_sub = gpd.read_file('data/Subway Lines.geojson')
df_attraction = pd.read_csv('data/number_of_vistors.csv', index_col=0)
table_columns = ['NAME', 'room type', 'price', 'service fee',
                 'minimum nights', 'number of reviews', 'review rate number', 'region']
prices = ['price', 'service fee']
max_columns = ['price', 'service fee',
               'number of reviews', 'review rate number']

graph_width = 1000
graph_height = 500
hightlight_size = 12
app = Dash(__name__)
token = open("token").read()
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
    html.H1(id='title', children='Air BnB appartments in New York'),
    html.Div(id='main_graph_options_container', children=[
        html.Div(id='graph_container', children=[
            dcc.Graph(id='main_graph', animate=False)
        ]),
        html.Div(id='table_container', children=[
            html.H2("Apparment comparrison"),
            dash_table.DataTable(id='comparison_table', columns=[
            {'name': f'{x[0].upper() + x[1:].lower()}', 'id': f'{x}', 'deletable': False} for x in table_columns], editable=True, row_deletable=True),
            html.Button('Add appartments to New York map',
                        id='add_points_button')], style={'display': 'none'})

    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div(id='sub_graph_container', children=[
        html.Div(children=[
            html.H2(id='sub_graph_header'),
            dcc.Graph(id='sub_graph_figure', animate=False,
                      style={'display': 'none'})
        ]),
        html.Div(id='comparison_container', children=[
            html.Div(children=[
                html.Br(),
                html.Br(),
                html.Div(id='graph_options_container', children=[
                    html.H2("Filter options"),
                    dcc.Checklist(id='graph_options', options=[
                        'New York districts', 'Public transit', 'Tourist attractions'], labelStyle={'display': 'block'}, value=['New York districts', 'Public transit', 'Tourist attractions']),
                    html.Details(id='price_range_details', children=[
                        html.Summary('Price range'),
                        dcc.RangeSlider(id='price_slider', value=[df.price.min(), df.price.max()], min=df.price.min(),
                                        max=df.price.max(), tooltip={"placement": "bottom", "always_visible": True})

                    ]),
                    html.Details(id='cancellation_summary', children=[
                        html.Summary('Cancellation policy'),
                        dcc.Checklist(style={'margin-left': '15px'}, id='cancellation', labelStyle={'display': 'block'}, options=df.cancellation_policy.unique(
                        ).tolist(), value=df.cancellation_policy.unique().tolist())
                    ]),
                    html.Details(
                        id='room_type_details', children=[
                            html.Summary('Room type'),
                            dcc.Checklist(style={'margin-left': '15px'}, id='room_type_checklist', labelStyle={
                                'display': 'block'}, options=df['room type'].unique().tolist(), value=df['room type'].unique().tolist())
                        ]
                    ),
                    html.Details(id='num_days_to_book', children=[
                        html.Summary('Number of nights you want to book'),
                        daq.NumericInput(
                            value=1, id='number_of_days_input', max=1000)
                    ]),
                    html.Details(id='average_review_details', children=[
                        html.Summary('Average review'),
                        dcc.Slider(id='review_slider', min=0, max=5, value=0, tooltip={
                            "placement": "bottom", "always_visible": True})
                    ]),
                    html.Details(id='instant_bookable_details', children=[
                        html.Summary('Immediately available'),
                        dcc.Checklist(style={'margin-left': '15px'}, id='available_check', labelStyle={
                            'display': 'block'}, options=['Instantly avaible'], value=[])
                    ]),
                    html.Details(id='service_fee_details', children=[
                        html.Summary('Maximum service fee'),
                        dcc.Slider(id='service_fee_slider', min=df['service fee'].min(), max=df['service fee'].max(
                        ), value=df['service fee'].max(), tooltip={"placement": "bottom", "always_visible": True})
                    ]),
                    html.Button('Apply', id='apply_button')





                ], style={'width': '500px', 'display': 'none'}
                )
            ])
        ])
    ], style={'display': 'flex', 'flex-direction': 'row'})
]
)


@app.callback(
    Output('main_graph', 'figure'),
    Input('apply_button', 'n_clicks'),
    Input('add_points_button', 'n_clicks'),
    State('main_graph', 'figure'),
    State('graph_options', 'value')

)
def update_graph(apply_button, add_points, figure, graph_options):
    go_fig = go.Figure(layout=go.Layout(height=graph_height, width=graph_width))
    go_fig.update_layout()
    if 'New York districts' in graph_options:
        second_option = go.Choroplethmapbox(geojson=geo_json, locations=df_plot.id, z=df_plot.appartment_count, customdata=df_plot['name'], colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']],
                                            zmin=df_plot.appartment_count.min(), zmax=df_plot.appartment_count.max(), marker_line_width=2, hoverinfo='none', marker=dict(opacity=0.2), showlegend=False, showscale=False, hovertemplate="%{customdata}")
        go_fig.add_trace(second_option)
        go_fig.update_geos(fitbounds="locations")
        go_fig.update_layout(mapbox_style="carto-positron",
                             mapbox_zoom=10, mapbox_center={"lat": 40.64964395581123, "lon": -73.828093870371374}, geo_scope='usa')

    if 'Public transit' in graph_options:
        go_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

        fig_2 = go.Scattermapbox(lat=lats, lon=lons,
                                 mode='lines', hoverinfo='skip', name='Metro lines', showlegend=True)
        go_fig.add_trace(fig_2)
        go_fig.update_geos(fitbounds="locations")
        go_fig.update_layout(mapbox_style="carto-positron",
                             mapbox_zoom=10, mapbox_center={"lat": 40.64964395581123, "lon": -73.828093870371374}, geo_scope='usa')

    if 'Tourist attractions' in graph_options:

        fig_3 = go.Scattermapbox(lon=df_attraction['long'], lat=df_attraction['lat'], text=df_attraction['Name'], customdata=df_attraction['Estimated number of visitors (millions)'],
                                 marker=dict(color='black'), showlegend=False, hovertemplate="%{text}<br>Average yearly visitors: %{customdata} million")

        go_fig.add_trace(fig_3)

    df_table = df[df['size'] == hightlight_size]
    fig_table = go.Scattermapbox(lon=df_table['long'], lat=df_table['lat'], text=df_table.NAME, mode='markers', marker=dict(color='yellow',
                                                                                                                            size=hightlight_size), showlegend=True, hovertemplate="%{text}", name='Selected apparments')
    go_fig.add_trace(fig_table)
    go_fig.update_layout(showlegend=True, mapbox=dict(accesstoken=token))
    go_fig.update_layout(mapbox_style="open-street-map")
    if not figure:
        return go_fig
    return go_fig


@app.callback(
    Output('sub_graph_figure', 'figure'),
    Output("sub_graph_header", 'children'),
    Output('sub_graph_figure', 'style'),
    Output('graph_options_container', 'style'),
    Input('apply_button', 'n_clicks'),
    Input('main_graph', 'clickData'),
    State('room_type_checklist', 'value'),
    State('service_fee_slider', 'value'),
    State('number_of_days_input', 'value'),
    State('price_slider', 'value'),
    State('cancellation', 'value'),
    State('review_slider', 'value'),
    prevent_initial_call=True
)
def on_graph_click(apply_botton, clickdata, room_type, service_fee, number_of_days, price_range, cancellation, min_review):
    region = clickdata['points'][0]['location']
    mask1 = (df['price'] >= price_range[0]) & (df['price'] <= price_range[1]) & (
        df['cancellation_policy'].isin(cancellation)) & (df['minimum nights'] <= number_of_days)
    mask2 = mask1 & (df['service fee'] <= service_fee) & (
        df['room type'].isin(room_type)) & (df['review rate number'] >= min_review)
    df_appartments_true = df[mask2]
    go_fig = go.Figure(layout=go.Layout(height=graph_height, width=graph_width+100))
    df_subregion = df_appartments_true[df_appartments_true['region'] == region]

    # Get unique app_types and create a trace for each app_type
    app_types = df_subregion['room type'].unique()
    traces = []
    for app_type in app_types:
        df_app_type = df_subregion[df_subregion['room type'] == app_type]
        trace = go.Scattermapbox(lon=df_app_type['long'], lat=df_app_type['lat'], text=df_app_type['NAME'],
                                 marker=dict(color='black'), marker_color=df_app_type['app_type_color'], hovertemplate="%{text}",
                                 name=app_type)
        traces.append(trace)

    # Add all traces to the figure and set the showlegend attribute to True
    for trace in traces:
        go_fig.add_trace(trace)
    index = df_plot[df_plot['id'] == region].index.tolist()[0]
    coords = df_plot.loc[index, 'centroid']
    go_fig.update_layout(showlegend=True, mapbox=dict(
        accesstoken=token, center=dict(lon=coords[0], lat=coords[1]), zoom=13))
    go_fig.update_geos(fitbounds="locations")
    go_fig.update_layout(mapbox_style="open-street-map")

    return go_fig, region.replace("_", " "), {"display": "block"}, {'width': '500px', 'display': 'block'}


@app.callback(
    Output('comparison_table', 'data'),
    Output('table_container', 'style'),
    Input('sub_graph_figure', 'clickData'),
    State('comparison_table', 'columns'),
    State('comparison_table', 'data'),
    prevent_initial_call=True
)
def on_sub_graph_click(clickdata, columns, rows):
    long = float(clickdata['points'][0]['lon'])
    lat = float(clickdata['points'][0]['lat'])
    row = df[(df['long'] == long) & (df['lat'] == lat)
             ][table_columns]
    index = row.index[0]
    df.loc[index, 'size'] = hightlight_size
    row = row.values.flatten().tolist()
    max_values = []
    if not rows:
        rows = []
    new_row = {c['id']: (f'${int(row[index])}' if c['id'] in prices else str(row[index]).replace("_", ' '))
               for index, c in enumerate(columns)}
    if not (new_row in rows):
        rows.append(new_row)

    return rows, {"display": "block"}


@app.callback(
    Output('comparison_table', 'style_data_conditional'),
    Input('comparison_table', 'data'),
    prevent_initial_call=False)
def update_table(rows):
    max_values = {key: ('$0' if key in prices else 0) for key in max_columns}
    min_values = {key: ('$0' if key in prices else 0) for key in max_columns}
    if rows:
        max_values = {}
        all_values = {key: [] for key in max_columns}
        for x in max_columns:
            for y in range(len(rows)):
                if x in prices:
                    all_values[x].append(int(rows[y][x][1:]))
                else:
                    all_values[x].append(float(rows[y][x]))
        max_values = {key: (f'${max(all_values[key])}' if key in prices else max(
            all_values[key])) for key in max_columns}
        min_values = {key: (f'${min(all_values[key])}' if key in prices else min(
            all_values[key])) for key in max_columns}
    style_data_conditional = [{
        'if': {
            'filter_query': '{{price}}={}'.format(max_values['price']),
            'column_id': 'price'
        },
        'backgroundColor': '#FB2C00'
    }, {
        'if': {
            'filter_query': '{{price}}={}'.format(min_values['price']),
            'column_id': 'price'
        },
        'backgroundColor': '#2EFB00'
    }, {
        'if': {
            'filter_query': '{{service fee}}={}'.format(max_values['service fee']),
            'column_id': 'service fee'
        },
        'backgroundColor': '#FB2C00'
    }, {
        'if': {
            'filter_query': '{{service fee}}={}'.format(min_values['service fee']),
            'column_id': 'service fee'
        },
        'backgroundColor': '#2EFB00'
    }, {
        'if': {
            'filter_query': '{{review rate number}}={}'.format(min_values['review rate number']),
            'column_id': 'review rate number'
        },
        'backgroundColor': '#FB2C00'
    }, {
        'if': {
            'filter_query': '{{review rate number}}={}'.format(max_values['review rate number']),
            'column_id': 'review rate number'
        },
        'backgroundColor': '#2EFB00'
    }]
    return style_data_conditional


if __name__ == '__main__':
    app.run_server(debug=True)
