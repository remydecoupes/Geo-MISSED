
import pandas as pd
import matplotlib.pyplot as plt
from shapely import wkt 
import geopandas as gpd
import folium
from branca.colormap import LinearColormap
import branca.colormap as cm
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objs as go
import json, requests
import os
import math
from urllib.parse import urlparse

MODEL = "Qwen/Qwen2.5-7B-Instruct"  
model_short_name = "Qwen2.5-7B-Instruct"

number_prediction = 10
NUTS_Level = 3

data_dir = "./output"
eurostat_data_files = [
    {
        "indicator": "income",
        "path_absolute": f"{data_dir}/income_2017_nuts_llm_{model_short_name}_absolute.csv",
        "path_relative": f"{data_dir}/income_2017_nuts_llm_{model_short_name}_relative.csv",
        "complete_year": "2017",
        "unit": "EUR_HAB"
    },
    {
        "indicator": "pop_density",
        "path_absolute": f"{data_dir}/pop_density_2018_nuts_llm_{model_short_name}_absolute.csv",
        "path_relative": f"{data_dir}/pop_density_2018_nuts_llm_{model_short_name}_relative.csv",
        "complete_year": "2018",
        "unit": "PER_KM2",
        "nan": ":", # null value are ":" and not empty cell
    },
    {
        "indicator": "poverty",
        "path_absolute": f"{data_dir}/poverty_2022_nuts_llm_{model_short_name}_absolute.csv",
        "path_relative": f"{data_dir}/poverty_2022_nuts_llm_{model_short_name}_relative.csv",
        "complete_year": "2022",
        "unit": "PC",
        "nan": ":"
    },
    {
        "indicator": "age_index",
        "path_absolute": f"{data_dir}/age_index_2021_nuts_llm_{model_short_name}_absolute.csv",
        "path_relative": f"{data_dir}/age_index_2021_nuts_llm_{model_short_name}_relative.csv",
        "complete_year": "2021",
        "unit": "NR"
    }
]

country_list = ['Austria', 'Albania', 'Belgium', 'Bulgaria', 'Switzerland',
       'Czechia', 'Cyprus', 'Germany', 'Denmark', 'Unknown', 'Estonia',
       'Spain', 'France', 'Finland', 'Croatia', 'Hungary', 'Italy',
       'Ireland', 'Norway', 'Netherlands', 'Montenegro',
       'North Macedonia', 'Lithuania', 'Malta', 'Luxembourg', 'Latvia',
       'Romania', 'Poland', 'Portugal', 'Serbia', 'Türkiye', 'Sweden',
       'Slovakia', 'Slovenia']
country_list = ['Germany', 'Spain', 'France', 'Italy']

style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}
# ------------------- #
# Bivariate map       #
# ------------------- #
def conf_defaults():
    # Define some variables for later use
    conf = {
        'plot_title': 'Bivariate choropleth map using Ploty',  # Title text
        'plot_title_size': 20,  # Font size of the title
        'width': 1000,  # Width of the final map container
        'ratio': 0.8,  # Ratio of height to width
        'center_lat': 0,  # Latitude of the center of the map
        'center_lon': 0,  # Longitude of the center of the map
        'map_zoom': 3,  # Zoom factor of the map
        'hover_x_label': 'Label x variable',  # Label to appear on hover
        'hover_y_label': 'Label y variable',  # Label to appear on hover
        'borders_width': 0.5,  # Width of the geographic entity borders
        'borders_color': '#f8f8f8',  # Color of the geographic entity borders

        # Define settings for the legend
        'top': 1,  # Vertical position of the top right corner (0: bottom, 1: top)
        'right': 1,  # Horizontal position of the top right corner (0: left, 1: right)
        'box_w': 0.04,  # Width of each rectangle
        'box_h': 0.04,  # Height of each rectangle
        'line_color': '#f8f8f8',  # Color of the rectagles' borders
        'line_width': 0,  # Width of the rectagles' borders
        'legend_x_label': 'Higher x value',  # x variable label for the legend
        'legend_y_label': 'Higher y value',  # y variable label for the legend
        'legend_font_size': 9,  # Legend font size
        'legend_font_color': '#333',  # Legend font color
    }

    # Calculate height
    conf['height'] = conf['width'] * conf['ratio']

    return conf

def recalc_vars(new_width, variables, conf=conf_defaults()):

    # Calculate the factor of the changed width
    factor = new_width / 1000

    # Apply factor to all variables that have been passed to th function
    for var in variables:
        if var == 'map_zoom':
            # Calculate the zoom factor
            # Mapbox zoom is based on a log scale. map_zoom needs to be set to value ideal for our map at 1000px.
            # So factor = 2 ^ (zoom - map_zoom) and zoom = log(factor) / log(2) + map_zoom
            conf[var] = math.log(factor) / math.log(2) + conf[var]
        else:
            conf[var] = conf[var] * factor

    return conf

def load_geojson(geojson_url, data_dir='data', local_file=False):

    # Make sure data_dir is a string
    data_dir = str(data_dir)

    # Set name for the file to be saved
    if not local_file:
        # Use original file name if none is specified
        url_parsed = urlparse(geojson_url)
        local_file = os.path.basename(url_parsed.path)

    geojson_file = data_dir + '/' + str(local_file)

    # Create folder for data if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download GeoJSON in case it doesn't exist
    if not os.path.exists(geojson_file):

        # Make http request for remote file data
        geojson_request = requests.get(geojson_url)

        # Save file to local copy
        with open(geojson_file, 'wb') as file:
            file.write(geojson_request.content)

    # Load GeoJSON file
    geojson = json.load(open(geojson_file, 'r'))

    # Return GeoJSON object
    return geojson

def set_interval_value(x, break_1, break_2):
    if x <= break_1:
        return 0
    elif break_1 < x <= break_2:
        return 1
    else:
        return 2

def prepare_df(df, x='x', y='y'):

    # Check if arguments match all requirements
    if df[x].shape[0] != df[y].shape[0]:
        raise ValueError('ERROR: The list of x and y coordinates must have the same length.')

    df[x] = df[x].astype(float)
    df[y] = df[y].astype(float)
    # Normalize income value
    # df[x] = (df[x] - df[x].min()) / (df[x].max() - df[x].min())
    # print(df[x])


    # Calculate break points at percentiles 33 and 66
    x_breaks = np.percentile(df[x], [33, 66])
    y_breaks = np.percentile(df[y], [33, 66])

    # print(f"x_breaks: {x_breaks} | not normalize: {x_breaks * (df[x].max() - df[x].min()) + df[x].min()}")
    print(f"x_breaks: {x_breaks}")
    print(f"y_breaks: {y_breaks}")

    # Assign values of both variables to one of three bins (0, 1, 2)
    x_bins = [set_interval_value(value_x, x_breaks[0], x_breaks[1]) for value_x in df[x]]
    y_bins = [set_interval_value(value_y, y_breaks[0], y_breaks[1]) for value_y in df[y]]

    # Calculate the position of each x/y value pair in the 9-color matrix of bivariate colors
    df['biv_bins'] = [int(value_x + 3 * value_y) for value_x, value_y in zip(x_bins, y_bins)]

    return df

def create_legend(fig, colors, conf=conf_defaults()):

    # Reverse the order of colors
    legend_colors = colors[:]
    legend_colors.reverse()

    # Calculate coordinates for all nine rectangles
    coord = []

    # Adapt height to ratio to get squares
    width = conf['box_w']
    height = conf['box_h']/conf['ratio']

    # Start looping through rows and columns to calculate corners the squares
    for row in range(1, 4):
        for col in range(1, 4):
            coord.append({
                'x0': round(conf['right']-(col-1)*width, 4),
                'y0': round(conf['top']-(row-1)*height, 4),
                'x1': round(conf['right']-col*width, 4),
                'y1': round(conf['top']-row*height, 4)
            })

    # Create shapes (rectangles)
    for i, value in enumerate(coord):
        # Add rectangle
        fig.add_shape(go.layout.Shape(
            type='rect',
            fillcolor=legend_colors[i],
            line=dict(
                color=conf['line_color'],
                width=conf['line_width'],
            ),
            xref='paper',
            yref='paper',
            xanchor='right',
            yanchor='top',
            x0=coord[i]['x0'],
            y0=coord[i]['y0'],
            x1=coord[i]['x1'],
            y1=coord[i]['y1'],
        ))

        # Add text for first variable
        fig.add_annotation(
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='top',
            x=coord[8]['x1'],
            y=coord[8]['y1'],
            showarrow=False,
            text=conf['legend_x_label'] + ' 🠒',
            font=dict(
                color=conf['legend_font_color'],
                size=conf['legend_font_size'],
            ),
            borderpad=0,
        )

        # Add text for second variable
        fig.add_annotation(
            xref='paper',
            yref='paper',
            xanchor='right',
            yanchor='bottom',
            x=coord[8]['x1'],
            y=coord[8]['y1'],
            showarrow=False,
            text=conf['legend_y_label'] + ' 🠒',
            font=dict(
                color=conf['legend_font_color'],
                size=conf['legend_font_size'],
            ),
            textangle=270,
            borderpad=0,
        )

    return fig

def create_bivariate_map(df, colors, geojson, x='x', y='y', ids='id', name='name', conf=conf_defaults()):

    if len(colors) != 9:
        raise ValueError('ERROR: The list of bivariate colors must have a length eaqual to 9.')

    # Recalculate values if width differs from default
    if not conf['width'] == 1000:
        conf = recalc_vars(conf['width'], ['height', 'plot_title_size', 'legend_font_size', 'map_zoom'], conf)

    # Prepare the dataframe with the necessary information for our bivariate map
    df_plot = prepare_df(df, x, y)

    # Create the figure
    fig = go.Figure(go.Choroplethmapbox(
        geojson=geojson,
        locations=df_plot[ids],
        z=df_plot['biv_bins'],
        marker_line_width=.5,
        colorscale=[
            [0/8, colors[0]],
            [1/8, colors[1]],
            [2/8, colors[2]],
            [3/8, colors[3]],
            [4/8, colors[4]],
            [5/8, colors[5]],
            [6/8, colors[6]],
            [7/8, colors[7]],
            [8/8, colors[8]],
        ],
        customdata=df_plot[[name, ids, x, y]],  # Add data to be used in hovertemplate
        hovertemplate='<br>'.join([  # Data to be displayed on hover
            '<b>%{customdata[0]}</b> (ID: %{customdata[1]})',
            conf['hover_x_label'] + ': %{customdata[2]}',
            conf['hover_y_label'] + ': %{customdata[3]}',
            '<extra></extra>',  # Remove secondary information
        ])
    ))

    # Add some more details
    fig.update_layout(
        title=dict(
            text=conf['plot_title'],
            font=dict(
                size=conf['plot_title_size'],
            ),
        ),
        mapbox_style='white-bg',
        width=conf['width'],
        height=conf['height'],
        autosize=True,
        mapbox=dict(
            center=dict(lat=conf['center_lat'], lon=conf['center_lon']),  # Set map center
            zoom=conf['map_zoom']  # Set zoom
        ),
    )

    fig.update_traces(
        marker_line_width=conf['borders_width'],  # Width of the geographic entity borders
        marker_line_color=conf['borders_color'],  # Color of the geographic entity borders
        showscale=False,  # Hide the colorscale
    )

    # Add the legend
    fig = create_legend(fig, colors, conf)

    return fig

for eurostat_dict in eurostat_data_files:
    nuts_level_max = 3
    for expe in ["relative"]:
    # for expe in ["absolute", "relative"]:
        indicator = eurostat_dict["indicator"]
        path = eurostat_dict[f"path_{expe}"]
        year = eurostat_dict["complete_year"]
        if indicator == "poverty":
            nuts_level_max = 2
        for NUTS_Level in range(1, nuts_level_max+1):

            df = pd.read_csv(path)

            try :
                df['geometry'] = df['geometry'].apply(wkt.loads)
            except:
                print("geometry loading wkt: already done")
            gdf = gpd.GeoDataFrame(df, geometry='geometry')
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=3035)  # Ajustez si nécessaire (EPSG initial probable)
            gdf = gdf.to_crs(epsg=4326)

            gdf = gdf[gdf["LEVL_CODE"] == NUTS_Level]
            gdf[f'{indicator}_{expe}_predicted'] = gdf[f'{indicator}_{expe}_predicted'].round(0)
            if expe == "absolute":
                gdf['diff_eurostat_llm'] = ((gdf[f'{indicator}_{expe}_predicted'] - gdf['data'])).round(0)
                gdf['diff_eurostat_llm_normalized'] = abs(gdf['diff_eurostat_llm']) / gdf['data']
            else:
                gdf['diff_eurostat_llm'] = ((gdf[f'{indicator}_{expe}_predicted'] - gdf[f'relative_{indicator}'])).round(0)
                gdf['diff_eurostat_llm_normalized'] = abs(gdf['diff_eurostat_llm']) / gdf[f'average_country_indicator']
            gdf.loc[gdf['diff_eurostat_llm'] > 50000, 'diff_eurostat_llm'] = np.nan

            

            gdf[f"{indicator}_{expe}_deviation"] = gdf[f"{indicator}_{expe}_deviation"].round(0)

            gdf_json = gdf.to_crs(epsg=4326).to_json()

            # ---------- #
            # Error maps #
            # ---------- #
            m = folium.Map(location=[50, 20], zoom_start=5)
            folium.Choropleth(
                geo_data=gdf_json,
                data=gdf,  
                columns=['NUTS_ID', 'diff_eurostat_llm_normalized'], 
                key_on='feature.properties.NUTS_ID',  
                fill_color='RdYlGn_r', 
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f'Difference between of {expe} {indicator} between eurostat {year} and {model_short_name} '
            ).add_to(m)
            hover = folium.features.GeoJson(
                gdf,
                style_function=style_function, 
                control=False,
                highlight_function=highlight_function, 
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['NUTS_NAME', "data", f'{indicator}_{expe}_predicted', "diff_eurostat_llm_normalized", f"{indicator}_{expe}_deviation"],
                    aliases=["Region: ", f"Eurostat {indicator}: ", f"LLM {expe} predicted: ", "normalized diff: ", f"std deviation for {number_prediction} llm predictions: "],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                )
            )
            m.add_child(hover)
            m.keep_in_front(m)
            path_save_html = path.replace(f"{data_dir}/", "")
            path_save_html = path_save_html.replace(f".csv", "")
            m.save(f'./docs/transformers/others_questions/{indicator}/error_maps_{path_save_html}_nuts_{NUTS_Level}.html')


            # --------------------- #
            # Error maps by country #
            # --------------------- #
            for country in country_list:
                gdf_copy = gdf.copy()
                gdf_copy = gdf_copy[gdf_copy["country"] == country]
                if expe == "absolute":
                    gdf_copy['diff_eurostat_llm_normalized'] = abs(gdf_copy['diff_eurostat_llm']) / gdf_copy['data']
                else:
                    gdf_copy['diff_eurostat_llm_normalized'] = abs(gdf_copy['diff_eurostat_llm']) / gdf_copy[f'average_country_indicator']

                gdf_copy_json = gdf_copy.to_crs(epsg=4326).to_json()
                m = folium.Map(location=[50, 20], zoom_start=5)
                folium.Choropleth(
                    geo_data=gdf_copy_json,
                    data=gdf_copy,
                    columns=['NUTS_ID', 'diff_eurostat_llm_normalized'], 
                    key_on='feature.properties.NUTS_ID',
                    fill_color='RdYlGn_r',
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=f'Difference between {expe} {indicator} eurostat {year} and {model_short_name}'
                ).add_to(m)
                hover = folium.features.GeoJson(
                    gdf,
                    style_function=style_function, 
                    control=False,
                    highlight_function=highlight_function, 
                    tooltip=folium.features.GeoJsonTooltip(
                        fields=['NUTS_NAME', str(year), f'{indicator}_{expe}_predicted', "diff_eurostat_llm_normalized", f"{indicator}_{expe}_deviation"],
                        aliases=["Region: ", f"Eurostat {indicator}: ", f"LLM {expe} predicted: ", "normalized diff: ", f"std deviation for {number_prediction} llm predictions: "],
                        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                    )
                )
                m.add_child(hover)
                m.keep_in_front(m)
                m.save(f'./docs/transformers/others_questions/{indicator}/error_maps_{path_save_html}_nuts_{NUTS_Level}_{country}.html')


            # ------------------- #
            # pearson correlation #
            # ------------------- #
            correlation_results = []
            for country, group in gdf.groupby('country'):
                if group['diff_eurostat_llm_normalized'].notna().any() and group[f"{indicator}_{expe}_logprobs"].notna().any():
                    try:
                        corr, _ = pearsonr(
                            group['diff_eurostat_llm_normalized'].dropna(),
                            group[f"{indicator}_{expe}_logprobs"].dropna()
                        )
                        correlation_results.append({
                            'Country': country,
                            'Correlation': corr,
                            f'{indicator}': group["data"].dropna().values[0]
                        })
                    except:
                        print(f"error with {country}")

            corr_df = pd.DataFrame(correlation_results)
            corr_df_sorted = corr_df.sort_values(by=f'{indicator}', ascending=False)
            gdp_min = corr_df_sorted[f'{indicator}'].min()
            gdp_max = corr_df_sorted[f'{indicator}'].max()
            gdp_range_text = f"{indicator} Range: {gdp_min} - {gdp_max}"

            fig = px.bar(
                corr_df_sorted,
                x='Country',
                y='Correlation',
                color='Correlation',
                hover_data={f'{indicator}': True},
                color_continuous_scale=["red", "yellow", "green"],
                title=f'Pearson Correlation between Absolute predict {expe} {indicator} error vs Logprobs - by Country Ordered by Eurostat {indicator}',
                labels={'Correlation': 'correlation', f'Country ordered by average {indicator}': 'Country'},
                template='plotly_white'
            )
            fig.write_html(f'./docs/transformers/others_questions/{indicator}/pearson_correlation_{path_save_html}_nuts_{NUTS_Level}.html')

            # ------------------- #
            #   Map of Logprobs   #
            # ------------------- #
            m = folium.Map(location=[50, 20], zoom_start=5)
            folium.Choropleth(
                geo_data=gdf_json,
                data=gdf,  
                columns=['NUTS_ID', f'{indicator}_{expe}_logprobs'], 
                key_on='feature.properties.NUTS_ID',  
                fill_color='RdYlGn', 
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f'Avergage logprobs for {expe} {indicator} prediction'
            ).add_to(m)
            hover = folium.features.GeoJson(
                gdf,
                style_function=style_function, 
                control=False,
                highlight_function=highlight_function, 
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['NUTS_NAME', str(year), f'{indicator}_{expe}_predicted', "diff_eurostat_llm_normalized", f"{indicator}_{expe}_deviation"],
                    aliases=["Region: ", f"Eurostat {expe} {indicator}: ", "LLM predicted: ", "normalized diff: ", f"std deviation for {number_prediction} llm predictions: "],
                    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
                )
            )
            m.add_child(hover)
            m.keep_in_front(m)
            m.save(f'./docs/transformers/others_questions/{indicator}/average_logprobs_{path_save_html}_nuts_{NUTS_Level}.html')

            # ------------------- #
            # Barplot of Logprobs #
            # ------------------- #
            gdf_long = gdf.melt(
                id_vars=['country', 'data'], 
                value_vars=['diff_eurostat_llm_normalized', f'{indicator}_{expe}_logprobs'], 
                var_name='Metric', 
                value_name='Value'
            )
            gdf_long = pd.merge(
                gdf_long, 
                gdf[['country', 'NUTS_NAME']],  # Keep only necessary columns for the merge
                on='country',
                how='left'
            )
            gdf_long = gdf_long.sort_values(by='data', ascending=False)
            fig = px.bar(
                gdf_long,
                x='country',
                y='Value',
                color='Metric',
                barmode='group',  # Ensures bars are grouped side by side
                # text='Value',  # Adds value annotations
                title='Diff Income with logprobs',
                color_discrete_map={
                    'diff_eurostat_llm_normalized': 'yellow', 
                    f'{indicator}_{expe}_logprobs': 'cyan'
                },
                template='plotly_white',
                hover_data={'NUTS_NAME': True, 'data': True}
            )
            fig.update_layout(
                xaxis_title='Country',
                yaxis_title='Value',
                legend_title='Metric',
                title_font_size=20,
            )
            fig.write_html(f'./docs/transformers/others_questions/{indicator}/barplot_logprobs_{path_save_html}_nuts_{NUTS_Level}.html')

            # ------------------- #
            # Bivariate map       #
            # ------------------- #
            color_sets = {
                'pink-blue':   ['#e8e8e8', '#ace4e4', '#5ac8c8', '#dfb0d6', '#a5add3', '#5698b9', '#be64ac', '#8c62aa', '#3b4994'],
                'blue-pink': ['#3b4994', '#8c62aa', '#be64ac', '#5698b9', '#a5add3', '#dfb0d6', '#5ac8c8', '#ace4e4', '#e8e8e8'],
                'teal-red':    ['#e8e8e8', '#e4acac', '#c85a5a', '#b0d5df', '#ad9ea5', '#985356', '#64acbe', '#627f8c', '#574249'],
                'red-teal': ['#574249', '#627f8c', '#64acbe', '#985356', '#ad9ea5', '#b0d5df', '#c85a5a', '#e4acac', '#e8e8e8'],
                'blue-organe': ['#fef1e4', '#fab186', '#f3742d',  '#97d0e7', '#b0988c', '#ab5f37', '#18aee5', '#407b8f', '#5c473d']
            }

            gdf_copy = gdf.copy()
            gdf_copy["prediction"] = gdf_copy[f"{indicator}_{expe}_predicted"]
            gdf_copy["error"] = abs(gdf_copy["prediction"] - gdf_copy[f"{expe}_{indicator}"] / gdf_copy[f"{expe}_{indicator}"])
            gdf_copy["logprobs"] = gdf_copy[f"{indicator}_{expe}_logprobs"]
            gdf_copy["logprobs"] = gdf_copy["logprobs"] * (-1)
            gdf_copy = gdf_copy.dropna(subset=["error"])
            # Define URL of the GeoJSON file for boundaries
            geojson_url = 'https://github.com/yotkadata/covid-waves/raw/main/data/NUTS_RG_10M_2016_4326.geojson'
            geojson = load_geojson(geojson_url)

            # Prepare bivariate map metadata
            conf = conf_defaults()
            conf['plot_title'] = f'Bivariate map between LLM prediction ({model_short_name}) Error prediction for {expe} {indicator} AND its confidence (logprobs * (-1))'
            conf['hover_x_label'] = f'error'  # Label to appear on hover
            conf['hover_y_label'] = 'Logprob'  # Label to appear on hover
            conf['width'] = 1000
            conf['center_lat'] = 48  # Latitude of the center of the map
            conf['center_lon'] = 9  # Longitude of the center of the map
            conf['map_zoom'] = 3.7  # Zoom factor of the map
            conf['borders_width'] = 0  # Width of the geographic entity borders
            conf['top'] = 0.5  # Vertical position of the top right corner (0: bottom, 1: top)
            conf['right'] = 0.1  # Horizontal position of the top right corner (0: left, 1: right)
            conf['line_width'] = 0  # Width of the rectagles' borders
            conf['legend_x_label'] = f'Higher error'  # x variable label for the legend
            conf['legend_y_label'] = 'Smaller confidence'  # y variable label for the legend

            fig = create_bivariate_map(gdf_copy, color_sets['teal-red'], geojson, x='error', y='logprobs', ids='NUTS_ID', name='NUTS_NAME', conf=conf)
            fig.update_layout(
                mapbox_layers = [{
                    'source': geojson,
                    'type': 'line',
                    'line': {'width': 0.5}
                },],
            )
            fig.write_html(f'./docs/transformers/others_questions/{indicator}/bivariate_maps_{path_save_html}_nuts_{NUTS_Level}.html')


