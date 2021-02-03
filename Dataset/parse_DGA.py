import pandas as pd
import glob
from tqdm import tqdm

DGA_paths = glob.glob("traffic_DGA/raw_DGA/*/list/5000.txt")
normal_paths = glob.glob("traffic_Alexa_Majestic/raw_Alexa_Majestic/list/1000000.txt")

print("Parsing DGA traffic")
print("-"*60)
for path in tqdm(DGA_paths):
    subpath = path.split("/")
    family = subpath[2]
    size = subpath[4].split(".")[0]
    df = pd.read_csv(path, names=["qname"])
    df.insert(0, "label", 1)
    df.insert(2, "family", family)
    df.to_csv(f"traffic_DGA/parsed_DGA/{family}/{size}.csv", sep=",", index=False)

print("Parsing Alexa & Majestic traffic")
print("-"*60)
for path in tqdm(normal_paths):
    print(path)
    subpath = path.split("/")
    family = "Alexa_Majestic"
    size = subpath[3].split(".")[0]
    df = pd.read_csv(path, names=["qname"])
    df.insert(0, "label", 0)
    df.insert(2, "family", family)
    df.to_csv(f"traffic_Alexa_Majestic/parsed_Alexa_Majestic/{size}.csv", sep=",", index=False)