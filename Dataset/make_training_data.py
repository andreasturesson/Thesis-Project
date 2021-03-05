from functools import reduce
from sklearn.model_selection import train_test_split
import pandas as pd
import glob

# Inspired by https://github.com/netrack/learn/blob/master/dns/make_multilabel_sets.py


def combine_tunneling_traffic(tunnel_paths, normal_paths):
    # Retrive data from all tunnelling csv
    df_tunnel = [pd.read_csv(path) for path in tunnel_paths]
    # Retrive data from all normal csv
    df_normal = [pd.read_csv(path) for path in normal_paths]

    # From the normal csv retrive traffic that matches with n top domain from
    # majestic csv. This is done to ensure that traffic is benign. Top domains
    # are less likley to be tunneling
    searchfor = pd.read_csv("Majestic-top-million-domains/majestic_million.csv", nrows=1000)
    searchfor = searchfor["Domain"].tolist()
    for idx, df in enumerate(df_normal):
        df_normal[idx] = df[df.qname.str.contains('|'.join(searchfor))]

    print(f"Only retrive normal traffic that contains top {len(searchfor)} domains from Majestic.")
    print("Example of top 10 domains:")
    print("-"*60)
    print(searchfor[:10])
    print("-"*60)

    # Retrive the count(size) of tunneling csv with lowest entries
    mincount = min([df.shape[0] for df in df_tunnel])
    print(f"Tunneling csv with least entries (mincount): {mincount}.")

    # Retrive the count(size) from normal csv lowest number of entries
    mincount_normal = min([df.shape[0] for df in df_normal])
    print(f"Normal csv with least entries (mincount_normal): {mincount_normal}.")

    # Take samples of data equal to count of lowest entries(mincount)
    df_tunnel = [df.sample(mincount, random_state=1) for df in df_tunnel]
    # Take samples of data equal to count of lowest entries(mincount_normal) <- alternative to full size
    # df_normal = [df.sample(mincount_normal, random_state=2) for df in df_normal] 
    # Take full size samples from entries, i.e., all data from all normal traffic csv.
    df_normal = [df.sample(frac=1) for df in df_normal]
    # Combine normal and tunnel traffic samples
    df_tunnel.append(df_normal)
    # ...
    names_df = reduce(lambda x, y: x.append(y), df_tunnel)
    # Remove duplicates from data
    names_df = names_df.drop_duplicates(subset=['qname'])

    # Retrive the count of tunnel traffic in data without duplicates
    tunnel_count = len(names_df[names_df["label"] == 1])
    print(f"Tunneling count is: {tunnel_count}.")
    # Retrive the count of normal traffic in data without duplicates
    normal_count = len(names_df[names_df["label"] == 0])
    print(f"Normal count is: {normal_count}.")
    # Set lowest_count to the count of the traffic type with lowest freqency
    if tunnel_count < normal_count:
        lowest_count = tunnel_count
        print(f"Tunneling count is the lowest: {tunnel_count}.")
    else:
        lowest_count = normal_count
        print(f"Normal count is the lowest: {normal_count}.")
    print(names_df.shape)

    # Group the dataframe by label and then take lowest_count samples of data from 
    # each label. This results in a balanced sample df_train.
    grouped = names_df.groupby("label")
    df_train = grouped.apply(lambda x: x.sample(n=lowest_count, replace=False, random_state=3)).reset_index(drop=True)

    # Shuffles the data
    df_train = df_train.sample(frac=1)

    # Print out final count
    final_tunnel_count = len(df_train[df_train["label"] == 1])
    final_normal_count = len(df_train[df_train["label"] == 0])
    print(f"Final tunneling count: {final_tunnel_count}.")
    print(f"Final normal count: {final_normal_count}.")
    print(f"Complete count: {final_normal_count+final_tunnel_count}.")

    # Remove the part that matches with the selected majestic top domains. 
    # For example if microsoft.com is part of the selected top domain then:
    # geover-prod.do.dsp.mp.microsoft.com. -> geover-prod.do.dsp.mp.
    # This is done to prevent the model/agent recognizing that none tunnled traffic
    # by the top domains. Only subdomain remains.
    searchfor = ['.' + item + '.' for item in searchfor]
    df_train['qname'] = df_train['qname'].str.replace('|'.join(searchfor), '')

    # Remove .maliciousDomain.com if exist in tunneling qnames
    df_train['qname'] = df_train['qname'].str.replace('.maliciousDomain.com', '')

    # write data to csv file
    df_train.to_csv("training_data.csv", index=False, header=True)


def combine_DGA_traffic(DGA_paths, normal_paths):
    # Retrive data from all DGA csv
    df_DGA = [pd.read_csv(path) for path in DGA_paths]

    # Retrive data from all normal csv
    df_normal = [pd.read_csv(path) for path in normal_paths]

    # Retrive the count(size) of tunneling csv with lowest entries
    mincount = min([df.shape[0] for df in df_DGA])

    # Sample with mincount of DGA entries
    df_DGA = [df.sample(5000, random_state=1) for df in df_DGA]

    # Retrive the count(size) of tunneling csv with lowest entries
    normal_mincount = min([df.shape[0] for df in df_normal])

    # Sample with mincount of DGA entries
    df_normal = [df.sample(normal_mincount, random_state=1) for df in df_normal]

    # Print csv with lowest count
    print("\n Csv with lowest count:")
    print("-"*60)
    print(f"DQA csv with least entries (mincount): {mincount}.")
    print(f"normal csv with least entries (mincount): {normal_mincount}. \n")

    # Combine dataset
    df_DGA.append(df_normal)
    df_combined = reduce(lambda x, y: x.append(y), df_DGA)
    df_combined = df_combined.assign(label_multiclass=(df_combined["family"]).astype("category").cat.codes)

    # Print the combined count
    print("\n Combined csv count:")
    print("-"*60)
    # Retrive the count of DGA traffic in data 
    DGA_count = len(df_combined[df_combined["label"] == 1])
    print(f"DGA count is: {DGA_count}.")
    # Retrive the count of normal traffic in data 
    normal_count = len(df_combined[df_combined["label"] == 0])
    print(f"Normal count is: {normal_count}.")
    # Set lowest_count to the count of the traffic type with lowest freqency
    if DGA_count < normal_count:
        lowest_count = DGA_count
        print(f"DGA count is the lowest: {DGA_count}.")
    else:
        lowest_count = normal_count
        print(f"Normal count is the lowest: {normal_count}.")

    # Group the dataframe by label and then take lowest_count samples of data from 
    # each label. This results in a balanced sample df_train.
    grouped = df_combined.groupby("label")
    df_train = grouped.apply(lambda x: x.sample(n=lowest_count, replace=False, random_state=2)).reset_index(drop=True)

    # Print count of each family
    print(f"\n {df_train['family'].nunique()} total family count:")
    print("-"*60)
    print(f"{df_train.family.value_counts()} \n")

    # Print out final count
    final_DGA_count = len(df_train[df_train["label"] == 1])
    final_normal_count = len(df_train[df_train["label"] == 0])
    print(f"Total DGA count: {final_DGA_count}.")
    print(f"Total normal count: {final_normal_count}.")
    print(f"Total count: {final_normal_count+final_DGA_count}. \n")

    # Split data into train at test split, shuffles the data with seeding
    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=3)

    # write training data to csv file
    df_train.to_csv("train_data.csv", index=False, header=True)

    # write testing data to csv file
    df_test.to_csv("test_data.csv", index=False, header=True)

    # Print train count of each family
    print(f"\n {df_train['family'].nunique()} train family count:")
    print("-"*60)
    print(f"{df_train.family.value_counts()} \n")

    # Print out train count
    final_DGA_count = len(df_train[df_train["label"] == 1])
    final_normal_count = len(df_train[df_train["label"] == 0])
    print(f"Train DGA count: {final_DGA_count}.")
    print(f"Train normal count: {final_normal_count}.")
    print(f"Train count: {final_normal_count+final_DGA_count}. \n")

    # Print test count of each family
    print(f"\n {df_test['family'].nunique()} test family count:")
    print("-"*60)
    print(f"{df_test.family.value_counts()} \n")

    # Print out test count
    final_DGA_count = len(df_test[df_test["label"] == 1])
    final_normal_count = len(df_test[df_test["label"] == 0])
    print(f"Test DGA count: {final_DGA_count}.")
    print(f"Test normal count: {final_normal_count}.")
    print(f"Test count: {final_normal_count+final_DGA_count}.")


def main():
    # benign_dir = "traffic_campus"
    # malicious_dir = "traffic_nozzel"

    benign_dir = "traffic_Alexa_Majestic"
    malicious_dir = "traffic_DGA"

    if(malicious_dir != "traffic_DGA"):
        normal_paths = glob.glob(f"{benign_dir}/qname*.csv")
        tunnel_paths = glob.glob(f"{malicious_dir}/ExfiltrationAttackFQDNs.csv")
        combine_tunneling_traffic(tunnel_paths, normal_paths)
    else:
        DGA_paths = glob.glob(f"{malicious_dir}/parsed_DGA/*/10000.csv")
        normal_paths = glob.glob(f"{benign_dir}/parsed_Alexa_Majestic/1000000.csv")
        combine_DGA_traffic(DGA_paths, normal_paths)


if __name__ == "__main__":
    main()