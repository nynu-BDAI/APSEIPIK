# 开发时间 2025/4/17 11:10
# 开发人员:牧良逢
import torch
from data import MscocoDataModule, F30kDataModule
from tqdm import tqdm

config = {
    "data_root": "PATH TO DATASET",
    "datasets": "coco",  # Will be overridden for Flickr
    "num_workers": 0,
    "per_gpu_batchsize": 256,
    "max_text_len":77
}
def generate_and_save_descriptions(data_module, dataset_name):
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    print(f"Generating descriptions for {dataset_name} training set...")
    for batch in tqdm(train_loader, desc=f"Processing {dataset_name}"):
        # The __getitem__ method of the dataset will handle description generation and caching
        pass
    print(f"Completed generating descriptions for {dataset_name}. Descriptions are saved in the cache directory.")

def main():
    # Process COCO dataset
    config["datasets"] = "coco"
    coco_data_module = MscocoDataModule(config, dist=False)
    generate_and_save_descriptions(coco_data_module, "COCO")

    #Process Flickr dataset
    # config["datasets"] = "f30k"
    # flickr_data_module = F30kDataModule(config, dist=False)
    # generate_and_save_descriptions(flickr_data_module, "Flickr")

if __name__ == "__main__":
    main()