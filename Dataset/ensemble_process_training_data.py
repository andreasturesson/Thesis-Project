import pandas as pd
import csv
import os
import tldextract
import re

# https://www.tutorialspoint.com/find-longest-consecutive-letter-and-digit-substring-in-python 


def number_tokens(s):
    tokens = s.split("-")
    total_tokens = 0

    for token in tokens:
        if token != '':
            total_tokens += 1

    return total_tokens


def suffix_hashing(s):
    suffix_hash = hash(s)

    return suffix_hash


def unique_numbers(s):
    total_unique = ''.join(set(s))
    total_unique = list(set(s))
    total_unique_numbers = 0

    for uniquie in total_unique:
        if uniquie.isdigit():
            total_unique_numbers += 1

    return total_unique_numbers


def unique_chars(s):
    total_unique = list(set(s))
    total_unique_chars = 0

    for uniquie in total_unique:
        if uniquie.isalpha():
            total_unique_chars += 1

    return total_unique_chars


def hex_ratio(subdomain, domain):
    subdomain_domain = (subdomain, domain)
    subdomain_domain = '.'.join(subdomain_domain)
    total_hex = 0

    hex_matches = re.findall(r"[0123456789abcdef]+", subdomain_domain) 

    for hex in hex_matches:
        total_hex += len(hex)
    
    hex_ratio = total_hex/len(subdomain_domain)

    return hex_ratio


def numbers_ratio(subdomain, domain):
    subdomain_domain = (subdomain, domain)
    subdomain_domain = '.'.join(subdomain_domain)

    total_numbers = sum(c.isdigit() for c in subdomain_domain)

    number_ratio = total_numbers/len(subdomain_domain)

    return number_ratio


def constant_vowel_ratio(subdomain, domain):
    subdomain_domain = (subdomain, domain)
    subdomain_domain = '.'.join(subdomain_domain)
    total_vowels = 0
    total_constants = 0

    constant_matches = re.findall(r"[bcdfghjklmnpqrstvwxyz]+", subdomain_domain) # Finds all chunks of constants letters

    vowel_matches = re.findall(r"[aeiou]+", subdomain_domain) # Finds all chunks of vowel letters

    for contants in constant_matches:
        total_constants += len(contants)

    for vowels in vowel_matches:
        total_vowels += len(vowels)

    constant_ratio = total_constants/len(subdomain_domain)    
    vowels_ratio = total_vowels/len(subdomain_domain)

    return vowels_ratio, constant_ratio


def longest_vowel_constant_sequence(s): 
    constants_matches = re.findall(r"[bcdfghjklmnpqrstvwxyz]+", s) # Finds all chunks of constants letters
    longest_constantSeq = max(constants_matches, key=len, default=0)

    vowel_matches = re.findall(r"[aeiou]+", s) # Finds all chunks of vowel letters
    longest_vowelSeq = max(vowel_matches, key=len, default=0)

    if longest_vowelSeq != 0:
        longest_vowelSeq = len(longest_vowelSeq)

    if longest_constantSeq != 0:
        longest_constantSeq = len(longest_constantSeq)

    return longest_vowelSeq, longest_constantSeq


def longest_number_sequence(s): 
    longest_letterSeq = '' 
    longest_digitSeq = '' 
    i = 0

    while(i<len(s)): 

        curr_letterSeq = '' 
        curr_digitSeq = '' 

        # For letter substring  
        while(i<len(s) and s[i].isalpha()): 
            curr_letterSeq += s[i] 
            i += 1

        # For digit substring 
        while(i<len(s) and s[i].isdigit()): 
            curr_digitSeq += s[i] 
            i += 1

        # Case handling if the character  
        # is neither letter nor digit     
        if(i< len(s) and not(s[i].isdigit()) and not(s[i].isalpha())):
            i += 1

        if(len(curr_letterSeq) > len(longest_letterSeq)):
            longest_letterSeq = curr_letterSeq 

        if(len(curr_digitSeq) > len(longest_digitSeq)):
            longest_digitSeq = curr_digitSeq 

    return len(longest_digitSeq)


def pair_letter_number(string):
    counter = 0

    for idx, char in enumerate(string):
        if (int(ord(char)) >= 97 and int(ord(char)) <= 122):
            if idx != len(string)-1:
                if (int(ord(string[idx+1])) >= 48 and int(ord(string[idx+1])) <= 57):
                    counter += 1
    return counter


def pair_number_letter(string):
    counter = 0

    for idx, char in enumerate(string):
        if (int(ord(char)) >= 48 and int(ord(char)) <= 57):
            if idx != len(string)-1:
                if (int(ord(string[idx+1])) >= 97 and int(ord(string[idx+1])) <= 122):
                    counter += 1
    return counter


""" def alphabetPairCheck(string):
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
    return AH_counter, HO_counter, OV_counter, VZ_counter """


def pair_alphabet_small_gap(string):
    domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY_pair, domain_YZ_pair = 0, 0, 0, 0, 0, 0, 0, 0, 0

    for idx, char in enumerate(string):
        if (int(ord(char)) >= 97 and int(ord(char)) <= 122):
            if idx != len(string)-1:
                if (int(ord(string[idx+1])) >= 97 and int(ord(string[idx+1])) <= 122):
                    if bool(re.search('[a-d]', char)) and bool(re.search('[a-d]', string[idx+1])):
                        domain_AD_pair += 1
                    if bool(re.search('[d-g]', char)) and bool(re.search('[d-g]', string[idx+1])):
                        domain_DG_pair += 1
                    if bool(re.search('[g-j]', char)) and bool(re.search('[g-j]', string[idx+1])):
                        domain_GJ_pair += 1
                    if bool(re.search('[j-m]', char)) and bool(re.search('[j-m]', string[idx+1])):
                        domain_JM_pair += 1
                    if bool(re.search('[m-p]', char)) and bool(re.search('[m-p]', string[idx+1])):
                        domain_MP_pair += 1
                    if bool(re.search('[p-s]', char)) and bool(re.search('[p-s]', string[idx+1])):
                        domain_PS_pair += 1
                    if bool(re.search('[s-v]', char)) and bool(re.search('[s-v]', string[idx+1])):
                        domain_SV_pair += 1
                    if bool(re.search('[v-y]', char)) and bool(re.search('[v-y]', string[idx+1])):
                        domain_VY_pair += 1
                    if bool(re.search('[v-z]', char)) and bool(re.search('[v-z]', string[idx+1])):
                        domain_YZ_pair += 1
     
    return domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY_pair, domain_YZ_pair


def alphabet_large_gap(string):
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


def alphabet_small_gap(string):
    domain_AD, domain_DG, domain_GJ, domain_JM, domain_MP, domain_PS, domain_SV, domain_VY, domain_YZ = 0, 0, 0, 0, 0, 0, 0, 0, 0

    for char in string:
        if (int(ord(char)) >= 97 and int(ord(char)) <= 122):
            if bool(re.search('[a-d]', char)):
                domain_AD += 1
            if bool(re.search('[d-g]', char)):
                domain_DG += 1
            if bool(re.search('[g-j]', char)):
                domain_GJ += 1
            if bool(re.search('[j-m]', char)):
                domain_JM += 1
            if bool(re.search('[m-p]', char)):
                domain_MP += 1
            if bool(re.search('[p-s]', char)):
                domain_PS += 1
            if bool(re.search('[s-v]', char)):
                domain_SV += 1
            if bool(re.search('[v-y]', char)):
                domain_VY += 1
            if bool(re.search('[y-z]', char)):
                domain_YZ += 1
    return domain_AD, domain_DG, domain_GJ, domain_JM, domain_MP, domain_PS, domain_SV, domain_VY, domain_YZ


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
        if input_char != "-" or input_char != ".":
            return 5

    # print(lowercase_letters, uppercase_letters, digits, dots, special_characters)


dataframe = pd.read_csv("training_data.csv")

if 'family' in dataframe.columns:
    dataframe.to_csv("dataset_ensemble/family.csv", sep=",", index=False, columns=["family"])

with open(os.path.join("dataset_ensemble/", "dataset.csv"), "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["domain_total", "domain_tokens", "domain_letters", "domain_numbers", "dots", "domain_special", "domain_letters_number_sum",
                        "domain_letterNumber", "domain_numberLetter",
                        "vowel_ratio", "constant_ratio", "number_ratio", "hexadecimal_ratio",  
                        #"domain_AH", "domain_HO", "domain_OV", "domain_VZ",
                        "domain_AD", "domain_DG", "domain_GJ", "domain_JM", "domain_MP", "domain_PS", "domain_SV", "domain_VY", "domain_YZ",
                        #"domain_AH_pair", "domain_HO_pair", "domain_OV_pair", "domain_VZ_pair",
                        "domain_AD_pair", "domain_DG_pair", "domain_GJ_pair", "domain_JM_pair", "domain_MP_pair", "domain_PS_pair", "domain_SV_pair", "domain_VY_pair", "domain_YZ_pair",
                        "domain_lng_numb_seq", "domain_lng_vowel_seq", "domain_lng_constant_seq",
                        "domain_uniq_chars", "domain_uniq_numbers",
                        "subdomain_total", "subdomain_tokens", "subdomain_letters", #"subdomain_numbers", "subdomain_special",
                        "subdomain_uniq_chars", "subdomain_uniq_numbers",
                        #"subdomain_letterNumber", "subdomain_numberLetter", 
                        #"subdomain_AH", "subdomain_HO", "subdomain_OV", "subdomain_VZ",
                        #"subdomain_AM", "subdomain_MZ",
                        #"subdomain_AH_pair", "subdomain_HO_pair", "subdomain_OV_pair", "subdomain_VZ_pair", 
                        "suffix_total", "suffix_hash", "suffix_AH", "suffix_HO", "suffix_OV", "suffix_VZ", 
                        "label"])

    for row in dataframe.itertuples():
        label, qname = "", ""
        subdomain, domain, fqdn, suffix = "", "", "", ""

        # Domain + subdomain values
        total_character, letters, numbers, dots, special_characters = 0, 0, 0, 0, 0
        vowel_ratio, constant_ratio, number_ratio, hexadecimal_ratio, special_ratio = 0, 0, 0, 0, 0

        # Domain values
        domain_total, domain_tokens, domain_letters, domain_numbers, domain_special = 0, 0, 0, 0, 0
        domain_lng_vowel_seq, domain_lng_constant_seq, domain_lng_numb_seq = 0, 0, 0
        domain_letterNumber, domain_numberLetter = 0, 0
        domain_uniq_chars, domain_uniq_numbers = 0, 0
        domain_AD, domain_DG, domain_GJ, domain_JM, domain_MP, domain_PS, domain_SV, domain_VY, domain_YZ = 0, 0, 0, 0, 0, 0, 0, 0, 0
        domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY, domain_YZ = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # Subdomain values
        subdomain_total, subdomain_tokens, subdomain_letters, subdomain_numbers, subdomain_special = 0, 0, 0, 0, 0
        subdomain_uniq_chars, subdomain_uniq_numbers = 0, 0
        #subdomain_letterNumber, subdomain_numberLetter = 0, 0
        #subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ = 0, 0, 0, 0
        #subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair = 0, 0, 0, 0

        # Suffix values
        suffix_total, suffix_hash, suffix_AH, suffix_HO, suffix_OV, suffix_VZ = 0, 0, 0, 0, 0, 0

        label = row.label
        qname = row.qname

        ext = tldextract.extract(qname)
        fqdn = ext.fqdn
        domain = ext.domain
        suffix = ext.suffix
        subdomain = ext.subdomain
        subdomain_domain = (subdomain, domain)
        subdomain_domain = '.'.join(subdomain_domain)

        vowel_ratio, constant_ratio = constant_vowel_ratio(domain, subdomain)
        number_ratio = numbers_ratio(domain, subdomain)
        hexadecimal_ratio = hex_ratio(domain, subdomain)

        domain_total = len(domain)
        domain_letterNumber = pair_letter_number(domain)
        domain_numberLetter = pair_number_letter(domain)
        domain_AD, domain_DG, domain_GJ, domain_JM, domain_MP, domain_PS, domain_SV, domain_VY, domain_YZ = alphabet_small_gap(domain)
        domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY_pair, domain_YZ_pair = pair_alphabet_small_gap(domain)
        domain_lng_numb_seq = longest_number_sequence(domain)
        domain_lng_vowel_seq, domain_lng_constant_seq = longest_vowel_constant_sequence(domain)
        domain_uniq_chars = unique_chars(domain)
        domain_uniq_numbers = unique_numbers(domain)
        domain_tokens = number_tokens(domain)
        
        #subdomain_letterNumber = letterNumberCheck(subdomain)
        #subdomain_numberLetter = numberLetterCheck(subdomain)
        #subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ  = alphabetCheck(subdomain)
        #subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair = alphabetPairCheck(subdomain)
        subdomain_total = len(subdomain)
        subdomain_uniq_chars = unique_chars(subdomain)
        subdomain_uniq_numbers = unique_numbers(subdomain)
        subdomain_tokens = number_tokens(subdomain)
        
        suffix_total = len(suffix)
        suffix_AH, suffix_HO, suffix_OV, suffix_VZ = alphabet_large_gap(suffix)
        suffix_hash = suffix_hashing(suffix)

        if type(fqdn) != float:
            total_character = len(subdomain_domain)
        else:
            continue

        for char in subdomain_domain:
            char_type = charCheck(char)
            if char_type == 1:
                letters += 1
            elif char_type == 2:
                # Uppecase does not exist, keep this just in case for later
                letters += 1
            elif char_type == 3:
                numbers += 1
            elif char_type == 4:
                dots += 1
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

        csvwriter.writerow([domain_total, domain_tokens, domain_letters, domain_numbers, dots, domain_special, domain_letters+domain_numbers,
                            domain_letterNumber, domain_numberLetter,
                            vowel_ratio, constant_ratio, number_ratio, hexadecimal_ratio,
                            #domain_AH, domain_HO, domain_OV, domain_VZ,
                            domain_AD, domain_DG, domain_GJ, domain_JM, domain_MP, domain_PS, domain_SV, domain_VY, domain_YZ,
                            #domain_AH_pair, domain_HO_pair, domain_OV_pair, domain_VZ_pair,
                            domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY_pair, domain_YZ_pair,
                            domain_lng_numb_seq, domain_lng_vowel_seq, domain_lng_constant_seq,
                            domain_uniq_chars, domain_uniq_numbers,
                            subdomain_total, subdomain_tokens, subdomain_letters, #subdomain_numbers, subdomain_special,
                            subdomain_uniq_chars, subdomain_uniq_numbers,
                            #subdomain_letterNumber, subdomain_numberLetter,
                            #subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ,
                            #subdomain_AM, subdomain_MZ,
                            #subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair,
                            suffix_total, suffix_hash, suffix_AH, suffix_HO, suffix_OV, suffix_VZ,
                            label])

