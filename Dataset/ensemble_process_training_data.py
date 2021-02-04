import pandas as pd
import csv
import os
import tldextract
import re


def suffixCheck(suffix):
    suffix_ASCII = [ord(c) for c in suffix]
    suffix = int("".join(map(str, suffix_ASCII))) 
    return suffix


def letterNumberCheck(string):
    counter = 0
    
    for idx, char in enumerate(string):
        if (int(ord(char)) >= 97 and int(ord(char)) <= 122):
            if idx != len(string)-1:
                if (int(ord(string[idx+1])) >= 48 and int(ord(string[idx+1])) <= 57):
                    counter += 1
    return counter


def numberLetterCheck(string):
    counter = 0
    
    for idx, char in enumerate(string):
        if (int(ord(char)) >= 48 and int(ord(char)) <= 57):
            if idx != len(string)-1:
                if (int(ord(string[idx+1])) >= 97 and int(ord(string[idx+1])) <= 122):
                    counter += 1
    return counter


def alphabetPairCheck(string):
    AH_counter, HO_counter, OV_counter, VZ_counter = 0, 0, 0, 0

    for idx, char in enumerate(string):
        if (int(ord(char)) >= 97 and int(ord(char)) <= 122):
            if idx != len(string)-1:
                if (int(ord(string[idx+1])) >= 97 and int(ord(string[idx+1])) <= 122):
                    if bool(re.search('[a-h]', char)) and bool(re.search('[a-h]', string[idx+1])):
                        AH_counter += 1
                    if bool(re.search('[h-o]', char)) and bool(re.search('[h-o]', string[idx+1])):
                        HO_counter += 1
                    if bool(re.search('[o-v]', char)) and bool(re.search('[o-v]', string[idx+1])):
                        OV_counter += 1
                    if bool(re.search('[v-z]', char)) and bool(re.search('[v-z]', string[idx+1])):
                        VZ_counter += 1
    return AH_counter, HO_counter, OV_counter, VZ_counter


def alphabetCheck(string):
    AH_counter, HO_counter, OV_counter, VZ_counter = 0, 0, 0, 0
    
    for char in string:
        if (int(ord(char)) >= 97 and int(ord(char)) <= 122):
            if bool(re.search('[a-h]', char)):
                AH_counter += 1
            if bool(re.search('[h-o]', char)):
                HO_counter += 1
            if bool(re.search('[o-v]', char)):
                OV_counter += 1
            if bool(re.search('[v-z]', char)):
                VZ_counter += 1
    return AH_counter, HO_counter, OV_counter, VZ_counter
        

def charCheck(input_char):
    # print(input_char)
    # lowercase_letters, uppercase_letters, digits, special_characters, dots = 0, 0, 0, 0, 0     

    # CHECKING FOR ALPHABET LOWERCASE
    if (int(ord(input_char)) >= 97 and int(ord(input_char)) <= 122):
        # lowercase_letters += 1
        return 1
    # CHECKING FOR ALPHABET UPPERCASE
    if ((int(ord(input_char)) >= 65 and int(ord(input_char)) <= 90)):
        # uppercase_letters += 1
        return 2
    # CHECKING FOR DIGITS  
    if (int(ord(input_char)) >= 48 and int(ord(input_char)) <= 57):
        # digits += 1
        return 3
    # CHECKING FOR dot
    if (int(ord(input_char)) == 46):
        # dots += 1
        return 4

    # OTHERWISE SPECIAL CHARACTER  
    if not((((int(ord(input_char)) >= 97 and int(ord(input_char)) <= 122)) or (((int(ord(input_char)) >= 65 and int(ord(input_char)) <= 90)) or ((int(ord(input_char)) >= 48 and int(ord(input_char)) <= 57)) or (int(ord(input_char)) == 46)))):
        # special_characters += 1
        return 5

    # print(lowercase_letters, uppercase_letters, digits, dots, special_characters)


dataframe = pd.read_csv("training_data.csv")

if 'family' in dataframe.columns:
    dataframe.to_csv("dataset_ensemble/family.csv", sep=",", index=False, columns=["family"])

with open(os.path.join("dataset_ensemble/", "dataset.csv"), "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["total_character",
                        "letters", "numbers", "special_characters", "letters_number",
                        "domain_total", "domain_letters", "domain_numbers", "domain_special",
                        "domain_letterNumber", "domain_numberLetter",
                        "domain_AH", "domain_HO", "domain_OV", "domain_VZ",
                        "domain_AH_pair", "domain_HO_pair", "domain_OV_pair", "domain_VZ_pair",
                        "subdomain_total", "subdomain_letters", "subdomain_numbers", "subdomain_special",
                        "subdomain_letterNumber", "subdomain_numberLetter", 
                        "subdomain_AH", "subdomain_HO", "subdomain_OV", "subdomain_VZ",
                        "subdomain_AH_pair", "subdomain_HO_pair", "subdomain_OV_pair", "subdomain_VZ_pair", 
                        "suffix", 
                        "label"])

    for row in dataframe.itertuples():
        label, qname = "", ""
        subdomain, domain, fqdn, suffix = "", "", "", ""
        total_character, letters, numbers, special_characters = 0, 0, 0, 0

        domain_total, domain_letters, domain_numbers, domain_special = 0, 0, 0, 0 
        domain_letterNumber, domain_numberLetter = 0, 0
        domain_AH, domain_HO, domain_OV, domain_VZ = 0, 0, 0, 0
        domain_AH_pair, domain_HO_pair, domain_OV_pair, domain_VZ_pair = 0, 0, 0, 0

        subdomain_total, subdomain_letters, subdomain_numbers, subdomain_special = 0, 0, 0, 0
        subdomain_letterNumber, subdomain_numberLetter = 0, 0
        subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ = 0, 0, 0, 0
        subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair = 0, 0, 0, 0

        suffix = 0

        label = row.label
        qname = row.qname

        ext = tldextract.extract(qname)
        fqdn = ext.fqdn
        domain = ext.registered_domain
        suffix = ext.suffix
        subdomain = ext.subdomain

        domain_letterNumber = letterNumberCheck(domain)
        domain_numberLetter = numberLetterCheck(domain)
        domain_AH, domain_HO, domain_OV, domain_VZ  = alphabetCheck(domain)
        domain_AH_pair, domain_HO_pair, domain_OV_pair, domain_VZ_pair = alphabetPairCheck(domain)
        domain_total = len(domain)

        subdomain_letterNumber = letterNumberCheck(subdomain)
        subdomain_numberLetter = numberLetterCheck(subdomain)
        subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ  = alphabetCheck(subdomain)
        subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair = alphabetPairCheck(subdomain)
        subdomain_total = len(subdomain)

        suffix = suffixCheck(suffix)

        if type(fqdn) != float:
            total_character = len(fqdn)
        else:
            continue

        for char in fqdn:
            char_type = charCheck(char)
            if char_type == 1:
                letters += 1
            elif char_type == 2:
                # Uppecase does not exist, keep this just in case for later
                letters += 1
            elif char_type == 3:
                numbers += 1
            elif char_type == 5:
                special_characters += 1

        for char in domain:
            char_type = charCheck(char)
            if char_type == 1:
                domain_letters += 1
            elif char_type == 2:
                # Uppecase does not exist, keep this just in case for later
                domain_letters += 1
            elif char_type == 3:
                domain_numbers += 1
            elif char_type == 5:
                domain_special += 1

        for char in subdomain:
            char_type = charCheck(char)
            if char_type == 1:
                subdomain_letters += 1
            if char_type == 2:
                # Uppecase does not exist, keep this just in case for later
                subdomain_letters += 1
            if char_type == 3:
                subdomain_numbers += 1
            if char_type == 4:
                subdomain_special += 1

        csvwriter.writerow([total_character,
                            letters,
                            numbers, special_characters,
                            letters+numbers,
                            domain_total, domain_letters, domain_numbers, domain_special,
                            domain_letterNumber, domain_numberLetter,
                            domain_AH, domain_HO, domain_OV, domain_VZ,
                            domain_AH_pair, domain_HO_pair, domain_OV_pair, domain_VZ_pair,
                            subdomain_total, subdomain_letters, subdomain_numbers, subdomain_special,
                            subdomain_letterNumber, subdomain_numberLetter,
                            subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ,
                            subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair,
                            suffix,
                            label])

