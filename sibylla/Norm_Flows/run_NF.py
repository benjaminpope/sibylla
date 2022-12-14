"""
Code to demonstrate some simple things about normalising flows and show the
refactoring has been done correctly

"""

import os

# Use pretrained:
import urllib.request
from urllib.error import HTTPError


checkpoint_path = "./saved_models/simple_example"

# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial11/"
# Files to download
pretrained_files = ["MNISTFlow_simple.ckpt", "MNISTFlow_vardeq.ckpt", "MNISTFlow_multiscale.ckpt",
                    "MNISTFlow_simple_results.json", "MNISTFlow_vardeq_results.json",
                    "MNISTFlow_multiscale_results.json"]
# Create checkpoint path if it doesn't exist yet
os.makedirs(checkpoint_path, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(checkpoint_path, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please contact the author with the full output including the"
                  "following error:\n", e)
