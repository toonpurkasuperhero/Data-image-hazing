import urllib.request
import zipfile
import os
import shutil
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://github.com/IDKiro/DehazeFormer/archive/refs/heads/main.zip"
zip_path = "DehazeFormer.zip"

print("Downloading DehazeFormer...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")

if os.path.exists("DehazeFormer"):
    shutil.rmtree("DehazeFormer")

os.rename("DehazeFormer-main", "DehazeFormer")
os.remove(zip_path)

print("Done")
