import os
import pandas as pd

numbered_folder_path = "fishes/images/numbered"
file_path = "fishes/final_all_index.txt"

columns = ["species_id", "species_name", "image_type", "file_name", "index_number"]
df = pd.read_csv(file_path, sep="=", names=columns)

df["index_number"] = df["index_number"].astype(str).str.strip() + ".png"

if os.path.exists(numbered_folder_path):
    numbered_files = set(os.listdir(numbered_folder_path))

    # Sprawdzenie dopasowania plików w numbered do index_number
    df["in_numbered_folder"] = df["index_number"].isin(numbered_files)

    # Liczenie zgodnych i brakujących plików
    matched_files = df["in_numbered_folder"].sum()
    missing_files = len(df) - matched_files

    print("Liczba dopasowanych plików: {}".format(matched_files))
    print("Liczba brakujących plików: {}".format(missing_files))
else:
    print("Nie znaleziono katalogu:", numbered_folder_path)

