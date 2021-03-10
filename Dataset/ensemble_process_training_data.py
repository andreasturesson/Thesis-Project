from typing import Counter
import pandas as pd
import csv
import os
import statistics
import tldextract
import collections
import re
import enchant
from string import punctuation as pt
from nltk import ngrams
from math import log2
from tqdm import tqdm


class Ngram(object):
    """
    Class to handle all ngram feature extraction.
    Based on Top 15 features from: https://github.com/jselvi/docker-r-masked-ngrams
    Paper: https://www-sciencedirect-com.libraryproxy.his.se/science/article/pii/S0957417419300648
    """
    def __init__(self, s):
        self.domain_name = s
        self.masked_domain_name = self.__get_masked_domain_name()

    def __get_masked_domain_name(self):
        aux1_domain_name = self.__multi_replace(self.domain_name, ['b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'], "c" )
        aux2_domain_name = self.__multi_replace(aux1_domain_name, ['a', 'e', 'i', 'o', 'u'], "v" )
        aux3_domain_name = self.__multi_replace(aux2_domain_name, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], "n" )
        masked_domain_name = self.__multi_replace(aux3_domain_name, ['-'], "s")
        return masked_domain_name

    def __multi_replace(self, source, chars_out, char_in):
        for c in chars_out:
            source = source.replace(c, char_in)
        return source

    def __get_ngram_stats(self, grams):
        counter = collections.Counter(grams)
        vector_values = counter.values()
        m = statistics.mean(vector_values)
        v = statistics.variance(vector_values)
        s = statistics.stdev(vector_values)
        return m, v, s

    def get_one_gram(self):
        n = 1
        one_grams = list(ngrams(self.domain_name, n))
        m, v, s = self.__get_ngram_stats(one_grams)
        return m, v, s

    def get_two_gram(self):
        n = 2
        two_grams = list(ngrams(self.domain_name, n))
        m, v, s = self.__get_ngram_stats(two_grams)
        return m, v, s

    def get_two_gram_circular(self):
        n = 2
        two_grams = list(ngrams(self.domain_name+self.domain_name, n))
        m, v, s = self.__get_ngram_stats(two_grams)
        return m, v, s

    def get_three_gram_circular(self):
        n = 2
        two_grams = list(ngrams(self.domain_name+self.domain_name, n))
        m, v, s = self.__get_ngram_stats(two_grams)
        return m, v, s

    def get_occurance_ngram(self):
        ngram_features = ["ccc", "cvc", "vcc", "vcv", "cv", "vc", "cc", "c", "v"]
        ngram_dict = {}
        for i in ngram_features:
            ngram_dict[i] = 0
        for i in [1, 2, 3]:
            ng = i
            v = list(ngrams(self.masked_domain_name, ng))
            for n in v:
                n = ''.join(n)
                if n in ngram_features:
                    ngram_dict[n] += 1
        return ngram_dict


def get_entropy(s: str):
    """
    Returns the metric entropy (Shannon's entropy divided by string length)
    Taken from: https://bit.ly/3t2eqSm
    """
    return -sum(i/len(s) * log2(i/len(s)) for i in Counter(s).values())


def number_tokens(s):
    """
    Returns the number of tokens in a string, tokens are seperated by "-".
    For example in 'a31sdf-dwa12' it would return '2'.
    """
    tokens = s.split("-")
    total_tokens = 0
    for token in tokens:
        if token != '':
            total_tokens += 1
    return total_tokens


def has_digits_or_punctuation(s):
    """
    Check if a string has any digit or symbols
    Taken from: https://github.com/alejandro-g-m/DetExt/tree/91a24bd174f599c6d9551f246f81519b274093af
    """
    return any(char.isdigit() or char in pt for char in s)


def get_all_substrings(s):
    """
    Return all the contiguous substrings in a string
    Taken from: https://github.com/alejandro-g-m/DetExt/tree/91a24bd174f599c6d9551f246f81519b274093af
    """
    substrings = []
    for i in range(len(s)):
        for j in range(i, len(s)):
            substrings.append(s[i:j+1])
    return substrings


def get_longest_meaningful_word(s):
    """
    Return the longest substring that belongs to the English dictionary
    has_digits_or_punctuation is needed because enchant understands digit
    strings and some symbols as valid words.
    Taken from: https://github.com/alejandro-g-m/DetExt/tree/91a24bd174f599c6d9551f246f81519b274093af
    """
    dictionary = enchant.Dict('en_US')
    substrings = set(get_all_substrings(s))
    longest_meaningful_word = ''
    for substring in substrings:
        if (not has_digits_or_punctuation(substring) and
            dictionary.check(substring.lower()) and
            len(substring) > len(longest_meaningful_word)):
            longest_meaningful_word = substring
    return longest_meaningful_word


def get_longest_meaningful_word_ratio(s):
    """
    Wrapper for get_longest_meaningful_word
    It returns the ratio compared to the total length
    Taken from: https://github.com/alejandro-g-m/DetExt/tree/91a24bd174f599c6d9551f246f81519b274093af
    """
    if len(s) > 0:
        return len(get_longest_meaningful_word(s)) / len(s)
    return 0


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


def hex_ratio(s):
    total_hex = 0
    hex_matches = re.findall(r"[0123456789abcdef]+", s) 

    for hex in hex_matches:
        total_hex += len(hex)
    hex_ratio = total_hex/len(s)
    return hex_ratio


def numbers_ratio(s):
    total_numbers = sum(c.isdigit() for c in s)
    number_ratio = total_numbers/len(s)
    return number_ratio


def constant_vowel_ratio(s):
    total_vowels = 0
    total_constants = 0

    constant_matches = re.findall(r"[bcdfghjklmnpqrstvwxyz]+", s)

    vowel_matches = re.findall(r"[aeiou]+", s)

    for contants in constant_matches:
        total_constants += len(contants)

    for vowels in vowel_matches:
        total_vowels += len(vowels)

    constant_ratio = total_constants/len(s) 
    vowels_ratio = total_vowels/len(s)
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
    """
    It returns longest number sequence in a string.
    For example 'cool123freezers928418' would return '928414'.
    Taken from: https://github.com/alejandro-g-m/DetExt/tree/91a24bd174f599c6d9551f246f81519b274093af
    """
    longest_letterSeq = ''
    longest_digitSeq = ''
    i = 0

    while(i<len(s)):
        curr_letterSeq = ''
        curr_digitSeq = ''

        while(i<len(s) and s[i].isalpha()): 
            curr_letterSeq += s[i] 
            i += 1

        while(i<len(s) and s[i].isdigit()): 
            curr_digitSeq += s[i] 
            i += 1
   
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


def pair_alphabet_small_gap(s):
    domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY_pair, domain_YZ_pair = 0, 0, 0, 0, 0, 0, 0, 0, 0

    for idx, char in enumerate(s):
        if (int(ord(char)) >= 97 and int(ord(char)) <= 122):
            if idx != len(s)-1:
                if (int(ord(s[idx+1])) >= 97 and int(ord(s[idx+1])) <= 122):
                    if bool(re.search('[a-d]', char)) and bool(re.search('[a-d]', s[idx+1])):
                        domain_AD_pair += 1
                    if bool(re.search('[d-g]', char)) and bool(re.search('[d-g]', s[idx+1])):
                        domain_DG_pair += 1
                    if bool(re.search('[g-j]', char)) and bool(re.search('[g-j]', s[idx+1])):
                        domain_GJ_pair += 1
                    if bool(re.search('[j-m]', char)) and bool(re.search('[j-m]', s[idx+1])):
                        domain_JM_pair += 1
                    if bool(re.search('[m-p]', char)) and bool(re.search('[m-p]', s[idx+1])):
                        domain_MP_pair += 1
                    if bool(re.search('[p-s]', char)) and bool(re.search('[p-s]', s[idx+1])):
                        domain_PS_pair += 1
                    if bool(re.search('[s-v]', char)) and bool(re.search('[s-v]', s[idx+1])):
                        domain_SV_pair += 1
                    if bool(re.search('[v-y]', char)) and bool(re.search('[v-y]', s[idx+1])):
                        domain_VY_pair += 1
                    if bool(re.search('[v-z]', char)) and bool(re.search('[v-z]', s[idx+1])):
                        domain_YZ_pair += 1
    return domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY_pair, domain_YZ_pair


def alphabet_large_gap(s):
    AH_counter, HO_counter, OV_counter, VZ_counter = 0, 0, 0, 0

    for char in s:
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


def alphabet_small_gap(s):
    domain_AD, domain_DG, domain_GJ, domain_JM, domain_MP, domain_PS, domain_SV, domain_VY, domain_YZ = 0, 0, 0, 0, 0, 0, 0, 0, 0

    for char in s:
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


def charCheck(s):
    # CHECKING FOR ALPHABET LOWERCASE
    if (int(ord(s)) >= 97 and int(ord(s)) <= 122):
        # lowercase_letters += 1
        return 1
    # CHECKING FOR ALPHABET UPPERCASE
    if ((int(ord(s)) >= 65 and int(ord(s)) <= 90)):
        # uppercase_letters += 1
        return 2
    # CHECKING FOR DIGITS  
    if (int(ord(s)) >= 48 and int(ord(s)) <= 57):
        # digits += 1
        return 3
    # CHECKING FOR dot
    if (int(ord(s)) == 46):
        # dots += 1
        return 4
    # OTHERWISE SPECIAL CHARACTER  
    if not((((int(ord(s)) >= 97 and int(ord(s)) <= 122)) or (((int(ord(s)) >= 65 and int(ord(s)) <= 90)) or ((int(ord(s)) >= 48 and int(ord(s)) <= 57)) or (int(ord(s)) == 46)))):
        # special_characters += 1
        if s != "-" or s != ".":
            return 5


def process_data(file):
    dataframe = pd.read_csv(file) 

    file_type = file.split("_")[0]

    if 'family' in dataframe.columns:
        dataframe.to_csv(f"dataset_ensemble/{file_type}_family.csv", sep=",", index=False, columns=["family", "label_multiclass"])

    with open(os.path.join("dataset_ensemble/", f"{file_type}_dataset.csv"), "w+") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["sld_total", "sld_tokens", "sld_letters", "sld_special", "sld_letters_number_sum",
                            "sld_letterNumber", "sld_numberLetter",
                            "vowel_ratio", "constant_ratio", "hexadecimal_ratio", "lng_meaningful_word_ratio", 
                            "one_gram_mean", "one_gram_var", "one_gram_std",
                            "two_gram_mean", "two_gram_var", "two_gram_std",
                            "two_gram_circular_mean", #"two_gram_circular_var", "two_gram_circular_std",
                            "three_gram_circular_mean",
                            "shannon_entropy",
                            "ngram_ccc", "ngram_cvc", "ngram_vcc", "ngram_vcv", "ngram_cv", "ngram_vc", "ngram_cc", "ngram_c", "ngram_v",
                            #"domain_AH", "domain_HO", "domain_OV", "domain_VZ",
                            "sld_AD", "sld_DG", "sld_GJ", "sld_JM", "sld_MP", "sld_PS", "sld_SV", "sld_VY", "sld_YZ",
                            #"domain_AH_pair", "domain_HO_pair", "domain_OV_pair", "domain_VZ_pair",
                            #"domain_AD_pair", "domain_DG_pair", "domain_GJ_pair", "domain_JM_pair", "domain_MP_pair", "domain_PS_pair", "domain_SV_pair", "domain_VY_pair", "domain_YZ_pair",
                            #"domain_lng_numb_seq", 
                            "sld_lng_vowel_seq", "sld_lng_constant_seq",
                            "sld_uniq_chars", "sld_uniq_numbers",
                            #"subdomain_total", "subdomain_tokens", "subdomain_letters", #"subdomain_numbers", "subdomain_special",
                            #"subdomain_uniq_chars", "subdomain_uniq_numbers",
                            #"subdomain_letterNumber", "subdomain_numberLetter", 
                            #"subdomain_AH", "subdomain_HO", "subdomain_OV", "subdomain_VZ",
                            #"subdomain_AM", "subdomain_MZ",
                            #"subdomain_AH_pair", "subdomain_HO_pair", "subdomain_OV_pair", "subdomain_VZ_pair",
                            "tld_total", "tld_hash", "tld_AH", "tld_HO", "tld_OV", "tld_VZ",
                            "label"])
        
        with tqdm(total=len(dataframe.qname)) as progress_bar:
            for row in dataframe.itertuples():
                label, qname = "", ""
                domain_name, sld, subdomain, tld = "", "", "", ""

                # Domain_name features
                total_character, letters, numbers, dots, special_characters = 0, 0, 0, 0, 0
                lng_meaningful_word_ratio = 0, 
                shannon_entropy = 0
                one_gram_mean, one_gram_var, one_gram_std = 0, 0, 0
                two_gram_mean, two_gram_var, two_gram_std = 0, 0, 0
                ngram_ccc, ngram_cvc, ngram_vcc, ngram_vcv, ngram_cv, ngram_vc, ngram_cc, ngram_c, ngram_v = 0, 0, 0, 0, 0, 0, 0, 0, 0
                
                # sld = subdomain features
                vowel_ratio, constant_ratio, number_ratio, hexadecimal_ratio, special_ratio = 0, 0, 0, 0, 0
                two_gram_circular_mean, two_gram_circular_var, two_gram_circular_std = 0, 0, 0
                three_gram_circular_mean, three_gram_circular_var, three_gram_circular_std = 0, 0, 0
                
                # sld features
                sld_total, sld_tokens, sld_letters, sld_numbers, sld_special = 0, 0, 0, 0, 0
                sld_lng_vowel_seq, sld_lng_constant_seq, sld_lng_numb_seq = 0, 0, 0
                sld_letterNumber, sld_numberLetter = 0, 0
                sld_uniq_chars, sld_uniq_numbers = 0, 0
                sld_AD, sld_DG, sld_GJ, sld_JM, sld_MP, sld_PS, sld_SV, sld_VY, sld_YZ = 0, 0, 0, 0, 0, 0, 0, 0, 0
                sld_AD_pair, sld_DG_pair, sld_GJ_pair, sld_JM_pair, sld_MP_pair, sld_PS_pair, sld_SV_pair, sld_VY_pair, sld_YZ_pair = 0, 0, 0, 0, 0, 0, 0, 0, 0

                # subdomain features
                subdomain_total, subdomain_tokens, subdomain_letters, subdomain_numbers, subdomain_special = 0, 0, 0, 0, 0
                subdomain_uniq_chars, subdomain_uniq_numbers = 0, 0
                #subdomain_letterNumber, subdomain_numberLetter = 0, 0
                #subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ = 0, 0, 0, 0
                #subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair = 0, 0, 0, 0

                # suffix features
                suffix_total, suffix_hash, suffix_AH, suffix_HO, suffix_OV, suffix_VZ = 0, 0, 0, 0, 0, 0

                label = row.label
                qname = row.qname
                
                ext = tldextract.extract(qname)
                domain_name = qname
                #domain = ext.domain
                sld = ext.domain
                tld = ext.suffix
                subdomain = ext.subdomain
                #sld_subdomain = (subdomain, sld)
                #sld_subdomain = '.'.join(sld_subdomain)
                sld_subdomain = (subdomain, sld)
                sld_subdomain = '.'.join(sld_subdomain)
                
                
                #------------------------------ domain_name ------------------------------ #
                
                lng_meaningful_word_ratio = get_longest_meaningful_word_ratio(domain_name)
                shannon_entropy =  get_entropy(domain_name)
                
                ngram = Ngram(domain_name)
                one_gram_mean, one_gram_var, one_gram_std = ngram.get_one_gram()
                two_gram_mean, two_gram_var, two_gram_std = ngram.get_two_gram()
                ngram_dict = ngram.get_occurance_ngram()
                ngram_ccc = ngram_dict["ccc"]
                ngram_cvc = ngram_dict["cvc"]
                ngram_vcc = ngram_dict["vcc"]
                ngram_vcv = ngram_dict["vcv"]
                ngram_cv = ngram_dict["cv"]
                ngram_vc = ngram_dict["vc"] 
                ngram_cc = ngram_dict["cc"]
                ngram_c = ngram_dict["c"]
                ngram_v = ngram_dict["v"]
                del ngram
                
                #------------------------------ sld + subdomain ------------------------------ #

                vowel_ratio, constant_ratio = constant_vowel_ratio(sld_subdomain)
                number_ratio = numbers_ratio(sld_subdomain)
                hexadecimal_ratio = hex_ratio(sld_subdomain)
                
                
                if(len(sld+subdomain) > 4):
                    c = Counter(sld+subdomain)
                    if len(c) != 1:
                        ngram = Ngram(sld+subdomain)
                        two_gram_circular_mean, two_gram_circular_var, two_gram_circular_std = ngram.get_two_gram_circular()
                        three_gram_circular_mean, three_gram_circular_var, three_gram_circular_std = ngram.get_three_gram_circular()
                        del ngram 

                #------------------------------ sld ------------------------------ #

                sld_total = len(sld)
                sld_letterNumber = pair_letter_number(sld)
                sld_numberLetter = pair_number_letter(sld)
                sld_AD, sld_DG, sld_GJ, sld_JM, sld_MP, sld_PS, sld_SV, sld_VY, sld_YZ = alphabet_small_gap(sld)
                sld_AD_pair, sld_DG_pair, sld_GJ_pair, sld_JM_pair, sld_MP_pair, sld_PS_pair, sld_SV_pair, sld_VY_pair, sld_YZ_pair = pair_alphabet_small_gap(sld)
                sld_lng_numb_seq = longest_number_sequence(sld)
                sld_lng_vowel_seq, sld_lng_constant_seq = longest_vowel_constant_sequence(sld)
                sld_uniq_chars = unique_chars(sld)
                sld_uniq_numbers = unique_numbers(sld)
                sld_tokens = number_tokens(sld)

                #------------------------------ subdomain ------------------------------ #
                
                #subdomain_letterNumber = letterNumberCheck(subdomain)
                #subdomain_numberLetter = numberLetterCheck(subdomain)
                #subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ  = alphabetCheck(subdomain)
                #subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair = alphabetPairCheck(subdomain)
                subdomain_total = len(subdomain)
                subdomain_uniq_chars = unique_chars(subdomain)
                subdomain_uniq_numbers = unique_numbers(subdomain)
                subdomain_tokens = number_tokens(subdomain)
                
                #------------------------------ tld ------------------------------ #
                
                tld_total = len(tld)
                tld_AH, tld_HO, tld_OV, tld_VZ = alphabet_large_gap(tld)
                tld_hash = suffix_hashing(tld)

                #------------------------------ mix ------------------------------ #

                for char in domain_name:
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

                for char in sld:
                    char_type = charCheck(char)
                    if char_type == 1:
                        sld_letters += 1
                    elif char_type == 2:
                        # Uppecase does not exist, keep this just in case for later
                        sld_letters += 1
                    elif char_type == 3:
                        sld_numbers += 1
                    elif char_type == 5:
                        sld_special += 1

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
                        
                progress_bar.update(1)
     
                
                csvwriter.writerow([sld_total, sld_tokens, sld_letters, sld_special, sld_letters+sld_numbers,
                                    sld_letterNumber, sld_numberLetter,
                                    vowel_ratio, constant_ratio, hexadecimal_ratio, lng_meaningful_word_ratio,
                                    one_gram_mean, one_gram_var, one_gram_std,
                                    two_gram_mean, two_gram_var, two_gram_std,
                                    two_gram_circular_mean, #two_gram_circular_var, two_gram_circular_std,
                                    three_gram_circular_mean,
                                    shannon_entropy,
                                    ngram_ccc, ngram_cvc, ngram_vcc, ngram_vcv, ngram_cv, ngram_vc, ngram_cc, ngram_c, ngram_v,
                                    #domain_AH, domain_HO, domain_OV, domain_VZ,
                                    sld_AD, sld_DG, sld_GJ, sld_JM, sld_MP, sld_PS, sld_SV, sld_VY, sld_YZ,
                                    #domain_AH_pair, domain_HO_pair, domain_OV_pair, domain_VZ_pair,
                                    #domain_AD_pair, domain_DG_pair, domain_GJ_pair, domain_JM_pair, domain_MP_pair, domain_PS_pair, domain_SV_pair, domain_VY_pair, domain_YZ_pair,
                                    #domain_lng_numb_seq, 
                                    sld_lng_vowel_seq, sld_lng_constant_seq,
                                    sld_uniq_chars, sld_uniq_numbers,
                                    #subdomain_total, subdomain_tokens, subdomain_letters, #subdomain_numbers, subdomain_special,
                                    #subdomain_uniq_chars, subdomain_uniq_numbers,
                                    #subdomain_letterNumber, subdomain_numberLetter,
                                    #subdomain_AH, subdomain_HO, subdomain_OV, subdomain_VZ,
                                    #subdomain_AM, subdomain_MZ,
                                    #subdomain_AH_pair, subdomain_HO_pair, subdomain_OV_pair, subdomain_VZ_pair,
                                    tld_total, tld_hash, tld_AH, tld_HO, tld_OV, tld_VZ,
                                    label])
                


def main():
    train_file = "train_data.csv"
    test_file = "test_data.csv"
    
    process_data(train_file) 
    process_data(test_file)

    pd_family_train = pd.read_csv("dataset_ensemble/train_family.csv", usecols=["label_multiclass"]) 
    pd_family_test = pd.read_csv("dataset_ensemble/test_family.csv", usecols=["label_multiclass"])
    pd_train = pd.read_csv("dataset_ensemble/train_dataset.csv")
    pd_test = pd.read_csv("dataset_ensemble/test_dataset.csv") 
    
    pd_train = pd_train[pd_train.columns[:-1]]
    pd_test = pd_test[pd_test.columns[:-1]]
    
    pd_train.join(pd_family_train)
    pd_multiclass_train = pd_train.join(pd_family_train)
    pd_multiclass_test = pd_test.join(pd_family_test)
    
    pd_multiclass_train.to_csv("dataset_ensemble/train_multiclass.csv", sep=",", index=False)
    pd_multiclass_test.to_csv("dataset_ensemble/test_multiclass.csv", sep=",", index=False)
    


if __name__ == "__main__":
    main()
