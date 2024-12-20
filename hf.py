from huggingface_hub import HfApi, hf_hub_download
import os
import tarfile

from data_collector import HINTS_DIR, ANNOTED_DIR, RAW_DIR, UNKNOWN_DIR, RAW_IMAGE_METADATA_PATH

def make_dataset_tar():
    with tarfile.open('./dataset/dataset.tar.gz', "w:gz") as tar:
        for p in [HINTS_DIR, ANNOTED_DIR, RAW_DIR, UNKNOWN_DIR, RAW_IMAGE_METADATA_PATH]:
            tar.add(p, arcname=os.path.basename(p))

    with tarfile.open('./dataset/only-hints.tar.gz', "w:gz") as tar:
        tar.add(HINTS_DIR, arcname=os.path.basename(HINTS_DIR))


def extract_dataset_tar(tar_path: str, local_dir: str):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(local_dir)


def upload_stuff():
    make_dataset_tar()
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
    if local_dir is not None:
        os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(
        repo_id="Yiffyi/resnet-geetest-v3-nine-pic",
        filename=model_name,
        repo_type="model",
        local_dir=local_dir
    )

def download_dataset(local_dir: str = './dataset', only_hint_images: bool = False) -> str:
    # return hf_hub_download(
    #     repo_id="Yiffyi/dataset-geetest-v3-nine-pic",
    #     repo_type="dataset",
    #     local_dir=local_dir
    # )
    os.makedirs(local_dir, exist_ok=True)
    tar = hf_hub_download(
        repo_id="Yiffyi/dataset-geetest-v3-nine-pic",
        filename='only-hints.tar.gz' if only_hint_images else 'dataset.tar.gz',
        repo_type="dataset"
    )
    extract_dataset_tar(tar, local_dir)


if __name__ == "__main__":
    os.makedirs("./model", exist_ok=True)
    os.makedirs("./dataset", exist_ok=True)
    download_model(model_name="model.onnx", local_dir="./model")
    download_dataset(local_dir="./dataset")
