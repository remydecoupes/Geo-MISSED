import pandas as pd 
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def contains_geo_info(text):
    ner_list = nlp(text)
    if not ner_list or not isinstance(ner_list, (list, tuple)):
        return False
    try:
        return any("LOC" in entity["entity"] for entity in ner_list)
    except:
        print(text)


hellaswag = load_dataset("hellaswag") # https://github.com/rowanz/hellaswag/tree/master/data
hellaswag_dict = {
    "dataset": "hellaswag",
    "split_considered": "train",
    "feature_considered": "ctx_a",
    "size_considered": len(hellaswag['train']['ctx_a'])
}

openbookqa = load_dataset("openbookqa") # https://huggingface.co/datasets/allenai/openbookqa
openbookqa_dict = {
    "dataset": "openbookqa",
    "split_considered": "validation",
    "feature_considered": "question_stem",
    "size_considered": len(openbookqa['validation']['question_stem'])
}

truthfulqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
truthfulqa_dict = {
    "dataset": "truthfulqa",
    "split_considered": "validation",
    "feature_considered": "question_stem",
    "size_considered": len(truthfulqa['validation']['question'])
}
benchmarks = [hellaswag_dict, openbookqa_dict, truthfulqa_dict]

# MMLU
# https://huggingface.co/datasets/cais/mmlu
mmlu_subjects = [
    "high_school_geography",
    "high_school_world_history",
    "high_school_us_history",
    "high_school_european_history",
    "international_law",
    "global_facts",
    "prehistory"
]

mmlu_subject = {}
for subject in mmlu_subjects:
    # print(f"Loading dataset for subject: {subject}")
    mmlu_subject[subject] = load_dataset("cais/mmlu", subject)
    mmlu_dict = {
        "dataset": f"mmlu_{subject}",
        "split_considered": "test",
        "feature_considered": "question",
        "size_considered": len(mmlu_subject[subject]['test']['question'])  
    }
    benchmarks.append(mmlu_dict)
df_benchmarks = pd.DataFrame(benchmarks)


df_geo = pd.DataFrame()

filtered_dataset = hellaswag['train'].filter(lambda batch: contains_geo_info(batch['ctx_a']))
# filtered_dataset.save_to_disk(f"output/geo_info_hellaswag")
df = filtered_dataset.to_pandas()
df["text"] = df["ctx_a"]
df["dataset"] = "hellaswag"

df_geo = pd.concat([df_geo, df[["text", "dataset"]]], ignore_index=True)

filtered_dataset = openbookqa['validation'].filter(lambda batch: contains_geo_info(batch['question_stem']))
df = filtered_dataset.to_pandas()
df["text"] = df["question_stem"]
df["dataset"] = "openbookqa"

df_geo = pd.concat([df_geo, df[["text", "dataset"]]], ignore_index=True)
# filtered_dataset.save_to_disk(f"output/geo_info_openbookqa")


filtered_dataset = truthfulqa['validation'].filter(lambda batch: contains_geo_info(batch['question']))
# truthfulqa.save_to_disk(f"output/geo_info_truthfulqa")
df = filtered_dataset.to_pandas()
df["text"] = df["question"]
df["dataset"] = "truthfulqa"

df_geo = pd.concat([df_geo, df[["text", "dataset"]]], ignore_index=True)
df_geo.to_csv(f"output/benchmarks_datasets_geo-info.csv")

filtered_mmlu_subject = {}

for subject, dataset in mmlu_subject.items():
    print(f"Filtering dataset for subject: {subject}")
    filtered_mmlu_subject[subject] = dataset['test'].filter(lambda batch: contains_geo_info(batch['question']))
    df = filtered_mmlu_subject[subject].to_pandas()
    df["text"] = df["question"]
    df["dataset"] = subject
    df_geo = pd.concat([df_geo, df[["text", "dataset"]]], ignore_index=True)

# filtered_mmlu_subject.save_to_disk(f"output/geo_info_mmlu")

df_geo.to_csv(f"output/benchmarks_datasets_geo-info.csv")
df_benchmarks["nb_geo_info"] = 0

for dataset in df_benchmarks["dataset"]:
    nb_geo_info = len(df_geo[df_geo["dataset"] == dataset.replace("mmlu_", "")])
    df_benchmarks.loc[df_benchmarks["dataset"] == dataset, "nb_geo_info"] = nb_geo_info

mmlu_rows = df_benchmarks[df_benchmarks["dataset"].str.startswith("mmlu")]

# Aggregate information for "mmlu"
mmlu_aggregated = {
    "dataset": "mmlu",
    "split_considered": "aggregated",  # Indicating aggregation
    "feature_considered": "question",  # Assuming the common feature
    "size_considered": mmlu_rows["size_considered"].sum(),
    "nb_geo_info": mmlu_rows["nb_geo_info"].sum()
}

# Add the aggregated row to the original DataFrame
df_benchmarks = pd.concat([df_benchmarks, pd.DataFrame([mmlu_aggregated])], ignore_index=True)


df_benchmarks["geo_info_percentage"] = (
    df_benchmarks["nb_geo_info"] / df_benchmarks["size_considered"] * 100
)

datasets_to_plot = ["hellaswag", "openbookqa", "truthfulqa", "mmlu"]
df_filtered = df_benchmarks[df_benchmarks["dataset"].isin(datasets_to_plot)]

colors = ["blue", "orange", "green", "red"]
plt.figure(figsize=(8, 6))
sns.barplot(
    x="dataset",
    y="geo_info_percentage",
    data=df_filtered,
    palette=colors
)
plt.title("Geo Info Percentage by Dataset", fontsize=16)
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("Geo Info Percentage (%)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("output/geo-info-distribution-common-benchmark.png")

nlp_aggragate_max = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
nlp_aggragate_first = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
nlp_aggragate_simple = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_list_locations(text):
    ner_list = nlp_aggragate_max(text)
    if not ner_list or not isinstance(ner_list, (list, tuple)):
        ner_list = nlp_aggragate_simple(text)
        if not ner_list or not isinstance(ner_list, (list, tuple)):
            ner_list = nlp_aggragate_first(text)
            if not ner_list or not isinstance(ner_list, (list, tuple)):
                print(f"Error: could not find spatial entity in: {text}")
                print(ner_list)
                return False
    location_list = [entity["word"] for entity in ner_list if "LOC" in entity["entity_group"]]
    return location_list

df_geo["locations"] = df_geo["text"].apply(extract_list_locations)

endpoint = "https://photon.komoot.io/api"

def geocode_with_photon(location_list):
    list_centroids = []
    list_osm_value = []
    for location in location_list:
        params = {
            'q': location,
            'limit': 1
        }
        response = requests.get(endpoint, params=params)
        data = response.json()
        if len(data) != 0:
            try:
                centroid = data['features'][0]['geometry']['coordinates']
                list_centroids.append(centroid)
                osm_value = data['features'][0]['properties']['osm_value']
                list_osm_value.append(osm_value)
            except:
                print(f"Could not geocode: {location}")
    return list_centroids, list_osm_value 

tqdm.pandas()

df_geo["geocode_results"] = df_geo["locations"].progress_apply(geocode_with_photon)
df_geo["centroids"] = df_geo["geocode_results"].apply(lambda x: x[0])
df_geo["osm_values"] = df_geo["geocode_results"].apply(lambda x: x[1])
df_geo.drop(columns=["geocode_results"], inplace=True)

df_geo.to_csv("output/df_geo_centroids.csv")





