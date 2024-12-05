# Explore LLMs geographic bias for European Country

The aim of this work is to map geographic bias normalized by GDP indicator 

From now, those models have been studied:

- Codestral-22B
- Llama3.1-8b
- Qwen2.5-7b

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