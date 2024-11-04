import os
import zipfile
from tqdm import tqdm

for item in tqdm(os.listdir('./drive_download')):
    zip_path = './drive_download/' + str(item)
    f = zipfile.ZipFile(zip_path,'r')
    # print(file.namelist())
    # if os.path.exists('./extract_dataset/' + str(item)):
    for file in f.namelist():
        f.extract(file, "./extract_dataset/")
    
    f.close()