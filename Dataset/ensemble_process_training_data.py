import pandas as pd
import csv
import os


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

with open(os.path.join("dataset_ensemble/", "dataset.csv"), "w+") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["label", "total_character",
                        "letters", "numbers",
                        "special_characters", "letters_number",
                        "dots", "lowercase_letters",
                        "uppercase_letters", "subdomains_count",
                        "subdomain1", "subdomain1_letters",
                        "subdomain1_numbers", "subdomain1_special",
                        "subdomain2", "subdomain2_letters",
                        "subdomain2_numbers", "subdomain2_special",
                        "subdomain3", "subdomain3_letters",
                        "subdomain3_numbers", "subdomain3_special",
                        "subdomain4", "subdomain4_letters",
                        "subdomain4_numbers", "subdomain4_special"])

    for row in dataframe.itertuples():
        label, qname = "", "",
        total_character, numbers, special_characters = 0, 0, 0
        lowercase_letters, uppercase_letters, dots, subdomain_count = 0, 0, 0, 0
        subdomain1, subdomain1_letters, subdomain1_numbers, subdomain1_special = 0, 0, 0, 0
        subdomain2, subdomain2_letters, subdomain2_numbers, subdomain2_special = 0, 0, 0, 0
        subdomain3, subdomain3_letters, subdomain3_numbers, subdomain3_special = 0, 0, 0, 0
        subdomain4, subdomain4_letters, subdomain4_numbers, subdomain4_special = 0, 0, 0, 0

        label = row.label
        qname = row.qname

        if not type(qname) == float:
            total_character = len(qname)
        else:
            continue

        splits = qname.split(".")

        if qname.endswith("."):
            subdomain_count = len(splits)-1
            splits.pop()
        else:
            subdomain_count = len(splits)

        for char in qname:
            char_type = charCheck(char)
            if char_type == 1:
                lowercase_letters += 1
            elif char_type == 2:
                uppercase_letters += 1
            elif char_type == 3:
                numbers += 1
            elif char_type == 4:
                dots += 1
            elif char_type == 5:
                special_characters += 1

        for idx, subdomain in enumerate(splits):
            idx += 1
            for char in subdomain:
                char_type = charCheck(char)
                if char_type == 1:
                    if(idx == 1):
                        subdomain1_letters += 1
                    if(idx == 2):
                        subdomain2_letters += 1
                    if(idx == 3):
                        subdomain3_letters += 1
                    if(idx == 4):
                        subdomain4_letters += 1
                elif char_type == 2:
                    if(idx == 1):
                        subdomain1_letters += 1
                    if(idx == 2):
                        subdomain2_letters += 1
                    if(idx == 3):
                        subdomain3_letters += 1
                    if(idx == 4):
                        subdomain4_letters += 1
                elif char_type == 3:
                    if(idx == 1):
                        subdomain1_numbers += 1
                    if(idx == 2):
                        subdomain2_numbers += 1
                    if(idx == 3):
                        subdomain3_numbers += 1
                    if(idx == 4):
                        subdomain4_numbers += 1
                elif char_type == 5:
                    if(idx == 1):
                        subdomain1_special += 1
                    if(idx == 2):
                        subdomain2_special += 1
                    if(idx == 3):
                        subdomain3_special += 1
                    if(idx == 4):
                        subdomain4_special += 1

        for idx, subdomain in enumerate(splits):
            idx += 1 
            if(idx == 1):
                subdomain1 = len(subdomain)
            if(idx == 2):
                subdomain2 = len(subdomain)
            if(idx == 3):
                subdomain3 = len(subdomain)
            if(idx == 4):
                subdomain4 = len(subdomain)

        csvwriter.writerow([label, total_character,
                            lowercase_letters+uppercase_letters,
                            numbers, special_characters,
                            lowercase_letters+uppercase_letters+numbers,
                            dots, lowercase_letters, uppercase_letters,
                            subdomain_count, subdomain1,
                            subdomain1_letters, subdomain1_numbers,
                            subdomain1_special, subdomain2,
                            subdomain2_letters, subdomain2_numbers,
                            subdomain2_special, subdomain3, subdomain3_letters,
                            subdomain3_numbers, subdomain3_special,
                            subdomain4, subdomain4_letters, subdomain4_numbers,
                            subdomain4_special])
