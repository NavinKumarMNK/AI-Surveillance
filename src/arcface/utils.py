# author : @NavinKumarMNK

import os

def find_most_recent_subfolder(folder_path):
    subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    if not subdirectories:
        return None  

    subdirectories.sort(
        key=lambda x: os.path.getctime(os.path.join(folder_path, x)), 
        reverse=True)
    return os.path.join(folder_path, subdirectories[0])

if __name__ == '__main__':
    folder_path = '/path/to/your/folder'  # Replace with the path to your folder
    most_recent_subfolder = find_most_recent_subfolder(folder_path)

    if most_recent_subfolder:
        print(f"The most recently created subfolder is: {most_recent_subfolder}")
    else:
        print("No subfolders found in the specified folder.")
