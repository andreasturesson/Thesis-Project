from functools import reduce
import pandas as pd

# Inspired by https://github.com/netrack/learn/blob/master/dns/make_multilabel_sets.py

TUNNEL_PATHS = [
    "Iodine/subdomain.csv",
]


NORMAL_PATHS = [
    "normal-traffic-campus/qname20160424_195423.csv",
    "normal-traffic-campus/qname_20160423_235403.csv",
    "normal-traffic-campus/qname_20160424_005404.csv",
    "normal-traffic-campus/qname_20160424_015405.csv",
    "normal-traffic-campus/qname_20160424_025406.csv",
    "normal-traffic-campus/qname_20160424_035407.csv",
    "normal-traffic-campus/qname_20160424_045408.csv",
    "normal-traffic-campus/qname_20160424_055409.csv",
    "normal-traffic-campus/qname_20160424_065410.csv",
    "normal-traffic-campus/qname_20160424_075411.csv",
    "normal-traffic-campus/qname_20160424_085412.csv",
    "normal-traffic-campus/qname_20160424_105414.csv",
    "normal-traffic-campus/qname_20160424_115415.csv",
    "normal-traffic-campus/qname_20160424_125416.csv",
    "normal-traffic-campus/qname_20160424_135417.csv",
    "normal-traffic-campus/qname_20160424_145418.csv",
    # "normal-traffic-campus/qname_20160424_155419.csv",
    # "normal-traffic-campus/qname_20160424_165420.csv",
    "normal-traffic-campus/qname_20160424_175421.csv",
    "normal-traffic-campus/qname_20160424_185422.csv",
    "normal-traffic-campus/qname_20160424_205424.csv",
    "normal-traffic-campus/qname_20160424_215425.csv",
    "normal-traffic-campus/qname_20160424_225426.csv",
    "normal-traffic-campus/qname_20160424_235427.csv",
    "normal-traffic-campus/qname_20160425_005428.csv",
    "normal-traffic-campus/qname_20160425_015429.csv",
    "normal-traffic-campus/qname_20160425_025430.csv",
    "normal-traffic-campus/qname_20160425_035431.csv",
    "normal-traffic-campus/qname_20160425_045432.csv",
    # "normal-traffic-campus/qname_20160425_055433.csv",
    "normal-traffic-campus/qname_20160425_065434.csv",
    "normal-traffic-campus/qname_20160425_215449.csv",
]


# retrive data from all tunnelling csv
dataframes = [pd.read_csv(path) for path in TUNNEL_PATHS]
# retrive data from all normal csv
dataframes_normal = [pd.read_csv(path) for path in NORMAL_PATHS]

# from the normal csv retrive traffic that matches with n top domain from
# majestic csv, this is done to ensure that traffic is benign. Top domains
# are less likley to be tunneling
searchfor = pd.read_csv("Majestic-top-million-domains/majestic_million.csv", nrows=100)
searchfor = searchfor["Domain"].tolist()
for idx, df in enumerate(dataframes_normal):
	dataframes_normal[idx] = df[df.qname.str.contains('|'.join(searchfor))]

print(f"Only retrive normal traffic that contains top {len(searchfor)} domains from Majestic.")
print("Example of top 10 domains:")
print("-"*60)
print(searchfor[:10])
print("-"*60)

# retrive the count(size) from tunneling csv lowest entries
mincount = min([df.shape[0] for df in dataframes])
print(f"Tunneling csv with least entries (mincount): {mincount}.")
# retrive the count(size) from normal csv lowest entries
mincount_normal = min([df.shape[0] for df in dataframes_normal])
print(f"Normal csv with least entries (mincount_normal): {mincount_normal}.")

# take samples of data equal to count of lowest entries(mincount)
dataframes = [df.sample(mincount) for df in dataframes]
# take samples of data equal to count of lowest entries(mincount_normal) <- alternative to full size
# dataframes_normal = [df.sample(mincount_normal) for df in dataframes_normal] <- alternative to full size 
# take full size samples from entries, i.e., all data from all normal traffic csv.
dataframes_normal = [df.sample(frac=1) for df in dataframes_normal]
# combine normal and tunnel traffic samples
dataframes.append(dataframes_normal)
# ...
names_df = reduce(lambda x, y: x.append(y), dataframes)
# remove duplicates from data
names_df = names_df.drop_duplicates(subset=['qname'])

# retrive the count of tunnel traffic in data without duplicates
tunnel_count = len(names_df[names_df["label"] == 1])
print(f"Tunneling count is: {tunnel_count}.")
# retrive the count of normal traffic in data without duplicates
normal_count = len(names_df[names_df["label"] == 0])
print(f"Normal count is: {normal_count}.")
# set lowest_count to the count of the traffic type with lowest freqency
if tunnel_count < normal_count:
    lowest_count = tunnel_count
    print(f"Tunneling count is the lowest: {tunnel_count}.")
else:
    lowest_count = normal_count
    print(f"Normal count is the lowest: {normal_count}.")
    
# take balanced sample, i.e., freqency of normal and tunnel traffic same.
# sample size = lowest_count
train_df = names_df.groupby("label").apply(lambda x: x.sample(
    n=lowest_count, random_state=1)).reset_index(drop=True)
# shuffles the data
train_df = train_df.sample(frac=1) 
final_tunnel_count = len(train_df[train_df["label"] == 1])
final_normal_count = len(train_df[train_df["label"] == 0])
print(f"Final tunneling count: {final_tunnel_count}.")
print(f"Final normal count: {final_normal_count}.")
print(f"Complete count: {final_normal_count+final_tunnel_count}.")

# Remove the part that matches with the selected majestic top domains. 
# For example if microsoft.com is part of the selected top domain then:
# geover-prod.do.dsp.mp.microsoft.com. -> geover-prod.do.dsp.mp.
# This is done to prevent the model/agent recognizing that none tunnled traffic
# by the top domains. 
searchfor = [item + '.' for item in searchfor]
train_df['qname'] = train_df['qname'].str.replace('|'.join(searchfor), '')

# write data to csv file
train_df.to_csv("training_data.csv", index=False, header=True)
