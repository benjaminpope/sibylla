"""
Code to demonstrate some simple things about normalising flows and show the
refactoring has been done correctly

"""

import os

import urllib.request
from urllib.error import HTTPError
import flows
from DataLoader import DataLoader
from TrainerModule import TrainerModule
from ModelEvaluator import ModelEvaluator
# from flows import MultiScaleImageFlow

if __name__ == "__main__":
    checkpoint_path = "./saved_models/simple_example"

    # Use pretrained model
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/JAX/tutorial11/"
    # Files to download
    pretrained_files = ["MNISTFlow_simple.ckpt", "MNISTFlow_vardeq.ckpt", "MNISTFlow_multiscale.ckpt",
                        "MNISTFlow_simple_results.json", "MNISTFlow_vardeq_results.json",
                        "MNISTFlow_multiscale_results.json"]
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(checkpoint_path, exist_ok=True)

    print("Checking pretrained files...", end='')
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
    print("Done!")

    print("Loading dataset...", end='')
    train_set, val_set, test_set = DataLoader.load_MNIST()
    train_exmp_loader, train_data_loader, \
        val_loader, test_loader = DataLoader.generate_data_loaders(train_set, val_set, test_set)
    print("Done!")

    # show_imgs(np.stack([train_set[i][0] for i in range(8)], axis=0))

    print("Creating flow...", end='')
    flow_dict = {"simple": {}, "vardeq": {}, "multiscale": {}}
    flow = flows.FlowFactory.create_multiscale_flow()
    flow_dict["multiscale"]["model"], \
        flow_dict["multiscale"]["result"] = TrainerModule.train_flow(flow,
                                                                     checkpoint_path,
                                                                     model_name="MNISTFlow_multiscale")
    print("Done!")

    print("Evalutating flow...", end='')
    evaluator = ModelEvaluator(flow_dict["multiscale"]["model"])
    evaluator.random_sample()
    print("Done!")
