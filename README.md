# Explore LLMs geographic bias for European Countries

The aim of this work is to map geographic bias normalized by geo indicator.

**LLMs' prediction of average income per inhabitant:**

The aim is to predict the annual income for each subregion of European countries (at the NUTS_2 level). The ground truth is based on Eurostat data.

The pipeline is divided into 3 steps:

1. Preprocessing and extracting Eurostat data.
2. Running LLMs to predict the average income per inhabitant.
3. Displaying on maps the difference (**MAPE**) between the predicted income and Eurostat data.

See the maps: https://remydecoupes.github.io/normalized_geobias_llm/


## Data

| Title | link | metadata | name file |
|---|---|---|---|
| GDP at current market prices by NUTS 3 regions| [link](https://ec.europa.eu/eurostat/web/main/data/database) | [metadata](https://ec.europa.eu/eurostat/cache/metadata/en/reg_eco10_esms.htm) | [estat_nama_10r_3gdp.tsv](./data/estat_nama_10r_3gdp.tsv) |
| NUTS3 region| | | [NUTS_RG_01M_2024_3035.geojson](./data/NUTS_RG_01M_2024_3035.geojson)|

## Install environment

**Code**:

```bash
conda create -n geobias python=3.10 pip ipython
conda activate geobias 
pip install geopandas pandas folium langchain langchain_community langchain_core timeout_decorator langchain_openai matplotlib pycountry
```

**Data**: 

You have to donwload the data files into data folder

-------
<img align="left" src="https://www.umr-tetis.fr/images/logo-header-tetis.png">


|           |
|----------------------|
| Rémy Decoupes        |