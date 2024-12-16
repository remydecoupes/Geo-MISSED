
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

MODEL = "Qwen/Qwen2.5-7B-Instruct"  
model_short_name = "Qwen2.5-7B-Instruct"

year = 2017
NUTS_Level = 1
country_list = ['Austria', 'Albania', 'Belgium', 'Bulgaria', 'Switzerland',
       'Czechia', 'Cyprus', 'Germany', 'Denmark', 'Unknown', 'Estonia',
       'Spain', 'France', 'Finland', 'Croatia', 'Hungary', 'Italy',
       'Ireland', 'Norway', 'Netherlands', 'Montenegro',
       'North Macedonia', 'Lithuania', 'Malta', 'Luxembourg', 'Latvia',
       'Romania', 'Poland', 'Portugal', 'Serbia', 'Türkiye', 'Sweden',
       'Slovakia', 'Slovenia']
country_list = ['Germany', 'Spain', 'France', 'Italy']

# df = pd.read_csv(f'./output/gdp_{year}_nuts_{NUTS_Level}_llm_{model_short_name}.csv')
df = pd.read_csv(f'./output/gdp_{year}_nuts_llm_{model_short_name}.csv')

try :
    df['geometry'] = df['geometry'].apply(wkt.loads)
except:
    print("geometry loading wkt: already done")
gdf = gpd.GeoDataFrame(df, geometry='geometry')
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=3035)  # Ajustez si nécessaire (EPSG initial probable)
gdf = gdf.to_crs(epsg=4326)

gdf = gdf[gdf["LEVL_CODE"] == NUTS_Level]
gdf['2017_predicted'] = gdf['2017_predicted'].round(0)
gdf['diff_eurostat_llm'] = ((gdf['2017_predicted'] - gdf['2017'])).round(0)
gdf.loc[gdf['diff_eurostat_llm'] > 50000, 'diff_eurostat_llm'] = np.nan
# gdf['diff_eurostat_llm_normalized'] = gdf['diff_eurostat_llm'] / gdf['2017']
# Z-score:
# mean_diff = gdf['diff_eurostat_llm'].mean()
# std_diff = gdf['diff_eurostat_llm'].std()
# gdf['diff_eurostat_llm_normalized'] = abs((gdf['diff_eurostat_llm'] - mean_diff) / std_diff).round(3)
# MAPE
gdf['diff_eurostat_llm_normalized'] = abs(gdf['diff_eurostat_llm']) / gdf['2017']

gdf[f"{year}_deviation"] = gdf[f"{year}_deviation"].round(0)

gdf_json = gdf.to_crs(epsg=4326).to_json()

# Absolute
m = folium.Map(location=[50, 20], zoom_start=5)
folium.Choropleth(
    geo_data=gdf_json,
    data=gdf,  
    # columns=['NUTS_ID', 'diff_eurostat_llm'], 
    columns=['NUTS_ID', 'diff_eurostat_llm_normalized'], 
    key_on='feature.properties.NUTS_ID',  
    fill_color='RdYlGn_r', 
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f'Difference between eurostat {year} and {model_short_name}'
).add_to(m)

style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

hover = folium.features.GeoJson(
    gdf,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['NUTS_NAME', str(year), f"{year}_predicted", "diff_eurostat_llm_normalized", f"{year}_deviation"],
        aliases=["Region: ", "Eurostat GDP: ", "LLM predicted: ", "normalized diff: ", "std deviation for 3 llm prediction: "],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
m.add_child(hover)
m.keep_in_front(m)

#colormap = LinearColormap(['green', 'yellow', 'red'], vmin=-1000, vmax=1000).to_step(5)
# colormap.add_to(m)


m.save(f'./output/gdp_{year}_nuts_{NUTS_Level}_llm_{model_short_name}.html')
m.save(f'./docs/transformers/gdp_{year}_nuts_{NUTS_Level}_llm_{model_short_name}.html')

# relative
gdf['diff_eurostat_llm_relative'] = (gdf['2017_relative_predicted'] - gdf['relative_income']).round(0)
gdf['diff_eurostat_llm_relative_normalized'] = abs(gdf['diff_eurostat_llm_relative']) / gdf[f"{year}"]


m = folium.Map(location=[50, 20], zoom_start=5)
folium.Choropleth(
    geo_data=gdf_json,
    data=gdf,
    # columns=['NUTS_ID', 'diff_eurostat_llm_relative'], 
    columns=['NUTS_ID', 'diff_eurostat_llm_relative_normalized'],
    key_on='feature.properties.NUTS_ID',
    fill_color='RdYlGn_r',
    # fill_color='viridis',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f'Normalized relative income "I" (Iregion - Icountry): Comparison between groundthruth data eurostat {year} and prediction for {model_short_name}'
).add_to(m)


style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

hover = folium.features.GeoJson(
    gdf,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['NUTS_NAME', f"{year}", f"{year}_predicted", "country_income", 'relative_income', f"{year}_relative_predicted", "diff_eurostat_llm_relative_normalized", f"{year}_relative_deviation", f"{year}_relative_logprobs", f"{year}_relative_logprobs_deviation"],
        aliases=["Region: ", f'eurostat Iregion: ', "llm predicted Iregion: ", "Eurostat Icountry: ", "Eurostat Iregion - Icountry: ", "RI_llm: ", "normalized diff: ", "std deviation for 3 llm prediction: ", "Average logprobs: ", "std deviation for logprobs: "],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
m.add_child(hover)
m.keep_in_front(m)

m.save(f'./docs/transformers/relative_gdp_{year}_nuts_{NUTS_Level}_llm_{model_short_name}.html')

for country in country_list:
    gdf_copy = gdf.copy()
    gdf_copy = gdf_copy[gdf_copy["country"] == country]
    gdf_copy['diff_eurostat_llm_normalized'] = abs(gdf_copy['diff_eurostat_llm']) / gdf_copy['2017']

    gdf_copy_json = gdf_copy.to_crs(epsg=4326).to_json()

    # Absolute
    m = folium.Map(location=[50, 20], zoom_start=5)
    folium.Choropleth(
        geo_data=gdf_copy_json,
        data=gdf_copy,  
        # columns=['NUTS_ID', 'diff_eurostat_llm'], 
        columns=['NUTS_ID', 'diff_eurostat_llm_normalized'], 
        key_on='feature.properties.NUTS_ID',  
        fill_color='RdYlGn_r', 
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'Difference between eurostat {year} and {model_short_name}'
    ).add_to(m)

    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}

    hover = folium.features.GeoJson(
        gdf,
        style_function=style_function, 
        control=False,
        highlight_function=highlight_function, 
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NUTS_NAME', str(year), f"{year}_predicted", "diff_eurostat_llm_normalized", f"{year}_deviation"],
            aliases=["Region: ", "Eurostat GDP: ", "LLM predicted: ", "Mean Absolute Percentage Error: ", "std deviation for 3 llm prediction: "],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
        )
    )
    m.add_child(hover)
    m.keep_in_front(m)
    m.save(f'./docs/transformers/gdp_{year}_nuts_{NUTS_Level}_{country}_llm_{model_short_name}.html')

    # Relative
    m = folium.Map(location=[50, 20], zoom_start=5)
    folium.Choropleth(
        geo_data=gdf_copy_json,
        data=gdf_copy,
        columns=['NUTS_ID', 'diff_eurostat_llm_relative'], 
        # columns=['NUTS_ID', 'diff_eurostat_llm_relative_normalized'],
        key_on='feature.properties.NUTS_ID',
        fill_color='RdYlGn_r',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'Relative income Iregion - Icountry: Comparison between groundthruth data eurostat {year} and prediction for {model_short_name}'
    ).add_to(m)


    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}

    hover = folium.features.GeoJson(
        gdf,
        style_function=style_function, 
        control=False,
        highlight_function=highlight_function, 
        tooltip=folium.features.GeoJsonTooltip(
            fields=['NUTS_NAME', f"{year}", f"{year}_predicted", "country_income", 'relative_income', f"{year}_relative_predicted", "diff_eurostat_llm_relative_normalized", f"{year}_relative_deviation", f"{year}_relative_logprobs", f"{year}_relative_logprobs_deviation"],
            aliases=["Region: ", f'eurostat Iregion: ', "llm predicted Iregion: ", "Eurostat Icountry: ", "Eurostat Iregion - Icountry: ", "RI_llm: ", "normalized diff: ", "std deviation for 3 llm prediction: ", "Average logprobs: ", "std deviation for logprobs: "],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
        )
    )
    m.add_child(hover)
    m.keep_in_front(m)
    m.save(f'./docs/transformers/relative_gdp_{year}_nuts_{NUTS_Level}_{country}_llm_{model_short_name}.html')

# pearson correlation
## Absolute
correlation_results = []
for country, group in gdf.groupby('CNTR_NAME'):
    if group['diff_eurostat_llm_normalized'].notna().any() and group['2017_relative_logprobs'].notna().any():
    # if group['diff_eurostat_llm_relative'].notna().any() and group['2017_relative_logprobs'].notna().any():
        try:
            corr, _ = pearsonr(
                group['diff_eurostat_llm_normalized'].dropna(),
                # group['diff_eurostat_llm_relative'].dropna(),
                group['2017_relative_logprobs'].dropna()
            )
            correlation_results.append({
                'Country': country,
                'Correlation': corr,
                'GDP': group['2017'].dropna().values[0]
            })
        except:
            print(f"error with {country}")

# Créer un DataFrame des résultats
corr_df = pd.DataFrame(correlation_results)
corr_df_sorted = corr_df.sort_values(by='GDP', ascending=False)
gdp_min = corr_df_sorted['GDP'].min()
gdp_max = corr_df_sorted['GDP'].max()
gdp_range_text = f"GDP Range: {gdp_min} - {gdp_max}"

# Créer un barplot avec une color bar
fig = px.bar(
    corr_df_sorted,
    x='Country',
    y='Correlation',
    color='Correlation',
    hover_data={'GDP': True},
    color_continuous_scale=["red", "yellow", "green"],
    title='Pearson Correlation between Absolute predict Income error vs Logprobs - by Country Ordered by GDP',
    labels={'Correlation': 'correlation', 'Country ordered by average income': 'Country'},
    template='plotly_white'
)
# Affichage du graphique
fig.write_html(f'./docs/transformers/pearson_correlation_absolute_gdp_{year}_nuts_{NUTS_Level}_llm_{model_short_name}.html')

## Relative
correlation_results = []
for country, group in gdf.groupby('CNTR_NAME'):
    if group['diff_eurostat_llm_relative_normalized'].notna().any() and group['2017_logprobs'].notna().any():
        # if group['2017_logprobs'].std() == 0:
            # group['2017_logprobs'] += np.random.normal(0, 1e-4, size=len(group))
        try:
            corr, _ = pearsonr(
                group['diff_eurostat_llm_relative_normalized'].dropna(),
                group['2017_logprobs'].dropna()
            )
            correlation_results.append({
                'Country': country,
                'Correlation': corr,
                'GDP': group['2017'].dropna().values[0]
            })
        except:
            print(f"error with {country}")
            # print(f"Relative: {group['diff_eurostat_llm_relative_normalized']}")
            # print(f"logprobs: {group['2017_logprobs']}")

corr_df = pd.DataFrame(correlation_results)
corr_df_sorted = corr_df.sort_values(by='GDP', ascending=False)
gdp_min = corr_df_sorted['GDP'].min()
gdp_max = corr_df_sorted['GDP'].max()
gdp_range_text = f"GDP Range: {gdp_min} - {gdp_max}"

fig = px.bar(
    corr_df_sorted,
    x='Country',
    y='Correlation',
    color='Correlation',
    hover_data={'GDP': True},
    color_continuous_scale=["red", "yellow", "green"],
    title='Pearson Correlation between relative predict Income error vs Logprobs - by Country Ordered by GDP',
    labels={'Correlation': 'correlation', 'Country  ordered by average income': 'Country'},
    template='plotly_white'
)
fig.write_html(f'./docs/transformers/pearson_correlation_relative_gdp_{year}_nuts_{NUTS_Level}_llm_{model_short_name}.html')
gdf.to_csv(f"./output/3_final_results.csv")


m = folium.Map(location=[50, 20], zoom_start=5)
folium.Choropleth(
    geo_data=gdf_json,
    data=gdf,  
    # columns=['NUTS_ID', 'diff_eurostat_llm'], 
    columns=['NUTS_ID', '2017_logprobs'], 
    key_on='feature.properties.NUTS_ID',  
    fill_color='RdYlGn', 
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f'Avergage logprobs for absolute Income prediction'
).add_to(m)


hover = folium.features.GeoJson(
    gdf,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['NUTS_NAME', str(year), f"{year}_predicted", "diff_eurostat_llm_normalized", f"{year}_deviation"],
        aliases=["Region: ", "Eurostat GDP: ", "LLM predicted: ", "normalized diff: ", "std deviation for 3 llm prediction: "],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
m.add_child(hover)
m.keep_in_front(m)

#colormap = LinearColormap(['green', 'yellow', 'red'], vmin=-1000, vmax=1000).to_step(5)
# colormap.add_to(m)

m.save(f'./docs/transformers/{model_short_name}_income_average_logprobs.html')


m = folium.Map(location=[50, 20], zoom_start=5)
folium.Choropleth(
    geo_data=gdf_json,
    data=gdf,  
    # columns=['NUTS_ID', 'diff_eurostat_llm'], 
    columns=['NUTS_ID', '2017_relative_logprobs'], 
    key_on='feature.properties.NUTS_ID',  
    fill_color='RdYlGn', 
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f'Avergage logprobs for RI prediction'
).add_to(m)


hover = folium.features.GeoJson(
    gdf,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['NUTS_NAME', str(year), f"{year}_predicted", "diff_eurostat_llm_normalized", f"{year}_deviation"],
        aliases=["Region: ", "Eurostat GDP: ", "LLM predicted: ", "normalized diff: ", "std deviation for 3 llm prediction: "],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
m.add_child(hover)
m.keep_in_front(m)

#colormap = LinearColormap(['green', 'yellow', 'red'], vmin=-1000, vmax=1000).to_step(5)
# colormap.add_to(m)

m.save(f'./docs/transformers/{model_short_name}_RI_average_logprobs.html')