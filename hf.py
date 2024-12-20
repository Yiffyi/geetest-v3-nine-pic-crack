from huggingface_hub import HfApi, hf_hub_download
import os
import tarfile

from data_collector import HINTS_DIR, ANNOTED_DIR, RAW_DIR, UNKNOWN_DIR, RAW_IMAGE_METADATA_PATH

def make_dataset_tar():
    with tarfile.open('./dataset/dataset.tar.gz', "w:gz") as tar:
        for p in [HINTS_DIR, ANNOTED_DIR, RAW_DIR, UNKNOWN_DIR, RAW_IMAGE_METADATA_PATH]:
            tar.add(p, arcname=os.path.basename(p))


def extract_dataset_tar(tar_path: str, local_dir: str):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(local_dir)


def upload_stuff():
    api = HfApi()
    api.upload_large_folder(
        folder_path="./dataset",
        repo_id="Yiffyi/dataset-geetest-v3-nine-pic",
        repo_type="dataset"
    )
    api.upload_folder(
        folder_path="./model",
        repo_id="Yiffyi/resnet-geetest-v3-nine-pic",
        repo_type="model"
    )

def download_model(model_name: str, local_dir: str = None) -> str:
    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(
        repo_id="Yiffyi/resnet-geetest-v3-nine-pic",
        filename=model_name,
        repo_type="model",
        local_dir=local_dir
    )

def download_dataset(local_dir: str = './dataset') -> str:
    # return hf_hub_download(
    #     repo_id="Yiffyi/dataset-geetest-v3-nine-pic",
    #     repo_type="dataset",
    #     local_dir=local_dir
    # )
    os.makedirs(local_dir, exist_ok=True)
    tar = hf_hub_download(
        repo_id="Yiffyi/dataset-geetest-v3-nine-pic",
        filename='dataset.tar.gz',
        repo_type="dataset"
    )
    extract_dataset_tar(tar, local_dir)

if __name__ == "__main__":
    os.makedirs("./model", exist_ok=True)
    os.makedirs("./dataset", exist_ok=True)
    download_model(model_name="model.onnx", local_dir="./model")
    download_dataset(local_dir="./dataset")
