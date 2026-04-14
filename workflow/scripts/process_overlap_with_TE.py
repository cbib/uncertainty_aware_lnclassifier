# Load data from snakemake
gtf = snakemake.input.left
bed = snakemake.input.bed

# Load Dfam TE family names
import json

import pandas as pd

DFAM_VERSION = "3.9"
root = "/mnt/cbib/LNClassifier/lnc-datasets/"

family_files = [
    f"{root}/raw/dfam/{DFAM_VERSION}/families?clade=Homo%20sapiens?clade_relatives=ancestors&start=0&limit=10000",
    f"{root}/raw/dfam/{DFAM_VERSION}/families?clade=Homo%20sapiens?clade_relatives=ancestors&start=10000&limit=10000",
    f"{root}/raw/dfam/{DFAM_VERSION}/families?clade=Homo%20sapiens?clade_relatives=ancestors&start=20000&limit=10000",
]

all_data = []
for file in family_files:
    with open(file, "r") as f:
        data = json.load(f)
        data = data["results"]
        all_data.extend(data)

print(f"Number of families: {len(all_data)}")
families_df = pd.DataFrame(all_data)
transposable_elements = set(
    families_df[families_df["second_order"] == "Transposable_Element"]["name"].tolist()
)
transposable_elements


# Load GTF, extract transcript IDs from GTF attributes column
gtf_df = pd.read_csv(gtf, sep="\t", header=None, comment="#")
gtf_df = gtf_df[gtf_df[2] == "transcript"]
gtf_df["transcript_id"] = gtf_df[8].str.split('"').str[3]

# Load BED with TE overlap results
bed_df = pd.read_csv(bed, sep="\t", header=None)
bed_df = bed_df[
    [2, 8, 17]
]  # Keep only relevant columns: feature type, attributes, and overlapping TE attributes
bed_df = bed_df[bed_df[2] == "transcript"]  # Keep only transcripts

bed_df = (
    bed_df.groupby(2)
    .apply(lambda x: list(set(x[17].str.split('"').str[1])))
    .reset_index()
)  # Group by transcript ID and make list of unique TE classes
bed_df["transcript_id"] = bed_df[8].str.split('"').str[3]
bed_df["has_family_match"] = bed_df[0].apply(
    lambda x: any(item in transposable_elements for item in x)
)
