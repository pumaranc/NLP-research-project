import os
import pandas as pd

files_path = 'sfarim'
def generate_csv_from_txt(csv_file_path, folder_path = files_path):
    data = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith(".txt"):
                with open(os.path.join(dirpath, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                # Get the directory name and append it to the filename
                directory_name = os.path.basename(dirpath)
                directory_parent_path = os.path.dirname(dirpath)
                directory_parent_name = os.path.basename(directory_parent_path)
                filename_without_extension, _ = os.path.splitext(filename)
                new_filename = filename_without_extension + "_" + directory_name + "_" + directory_parent_name
                data.append((new_filename, content))
    df = pd.DataFrame(data, columns=['name', 'content'])
    df.to_csv(csv_file_path, index=False)
