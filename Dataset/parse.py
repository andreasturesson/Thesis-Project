import os
from tqdm import tqdm
import argparse
import csv
import scapy.layers.dns
import scapy.sendrecv

PARSE_DATA = [
    {"pcap": "Iodine/iodine.pcap",
    "out": "Iodine/",
    "label": "1",
    "data_type": "1"},

]


def write_qname(writer, label, data_type):
    csvwriter = csv.writer(writer)
    csvwriter.writerow(["label", "qname"])

    def _prn(pkt):
        dns = pkt.lastlayer()
        if dns.qr == 0 and data_type == "0":  # dns.qr: query (0), or a response (1)
            csvwriter.writerow([label, dns.qd.qname.decode(errors="ignore")])
        elif dns.qr == 0 and data_type == "1":
            if "dns-tunnel" in dns.qd.qname.decode(errors="ignore"):
                csvwriter.writerow([label, dns.qd.qname.decode(errors="ignore")])
    return _prn


def write_subdomain(writer, label, data_type):
    csvwriter = csv.writer(writer)
    csvwriter.writerow(["label", "subdomain"])

    def _prn(pkt):
        dns = pkt.lastlayer()
        if dns.qr == 0 and data_type == "0":  # dns.qr: query (0), or a response (1) 
            csvwriter.writerow([label, dns.qd.qname.decode(errors="ignore").split(".")[0]])
        elif dns.qr == 0 and data_type == "1":
            if "dns-tunnel" in dns.qd.qname.decode(errors="ignore"):
                csvwriter.writerow([label, dns.qd.qname.decode(errors="ignore").split(".")[0]])
    return _prn


def main():
    #parser = argparse.ArgumentParser(description="Parse DNS packets")

    #parser.add_argument("parse", metavar="PARSE", type=str,
    #                    help="Choose parsing out, choices: qname or subdomain")
    #parser.add_argument("pcap", metavar="PCAP", type=str,
    #                    help="Traffic dump. Example: YourPath/YourCapturedPackets.pcap")
    #parser.add_argument("out", metavar="OUT", type=str,
    #                    help="Output attributes. Example: YourPath/GivenName.cvs")
    #parser.add_argument("label", metavar="label", type=int,
    #                    help="The class label. Example: 0")
    #parser.add_argument("data_type", metavar="DATA_TYPE", type=str,
    #                    help="normal 0, tunnel 1")

    #args = parser.parse_args()

    for data in tqdm(PARSE_DATA):
        with open(os.path.join(data["out"], "qname.csv"), "w+") as csvfile:
            scapy.sendrecv.sniff(
                offline=data["pcap"],
                store=False,
                lfilter=lambda p: p.haslayer(scapy.layers.dns.DNS),
                prn=write_qname(csvfile, data["label"], data["data_type"]))
        with open(os.path.join(data["out"], "subdomain.csv"), "w+") as csvfile:
            scapy.sendrecv.sniff(
                offline=data["pcap"],
                store=False,
                lfilter=lambda p: p.haslayer(scapy.layers.dns.DNS),
                prn=write_subdomain(csvfile,  data["label"], data["data_type"]))


if __name__ == "__main__":
    main()
