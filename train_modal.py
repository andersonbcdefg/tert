from modal import Image, App, gpu, Volume, Secret
from src.bitbert.data import (
    download_dclm_data,
    download_fineweb_data,
    download_wiki_data,
)

# DATA_DIR = "/data"
MOUNT_PATH = "/training_outputs"
DATA_MOUNT_PATH = "/data_vol"
DATA_DIR = DATA_MOUNT_PATH


# def download_dclm():
#     download_dclm_data(DATA_DIR)


# def download_fineweb():
#     download_fineweb_data(DATA_DIR)


# def download_wiki():
#     print("hellooo")
#     download_wiki_data(DATA_DIR)


vol = Volume.from_name("bitbert-outputs", create_if_missing=True)
data_vol = Volume.from_name("bitbert-data", create_if_missing=True)

def save_callback():
    vol.commit()


image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "pip install torch==2.6.0.dev20241119+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124"
    )
    .pip_install(
        "datasets==3.1.0",
        "tiktoken",
        "blobfile",
        "transformers",
        "awkward",
        "einops",
        "datasets",
        "tqdm",
        "liger-kernel",
        "pyarrow",
        "zstandard",
        "pydantic>=2.0",
        "adam-mini"
    )
    .pip_install(
        "torchao@git+https://github.com/pytorch/ao.git",
        "muon@git+https://github.com/KellerJordan/Muon.git",
        "cut-cross-entropy@git+https://github.com/andersonbcdefg/ml-cross-entropy.git"
    )
    # .run_function(download_wiki, secrets=[Secret.from_name("HF-SECRET")])
    # .run_function(download_dclm, secrets=[Secret.from_name("HF-SECRET")])
    # .run_function(download_fineweb, secrets=[Secret.from_name("HF-SECRET")])
    # .env({"TORCH_TRACE": "/training_outputs/trace"})
)

app = App("train-bitbert")


@app.function(
    image=image,
    gpu=gpu.H100(),
    timeout=60 * 60 * 24,
    volumes={
        MOUNT_PATH: vol,
        DATA_MOUNT_PATH: data_vol
    },
    secrets=[Secret.from_name("HF-SECRET")],
)
def train(job_id: str):
    vol.reload()
    from src.bitbert.train import train, TrainArgs

    train(job_id, TrainArgs(), MOUNT_PATH, DATA_DIR, save_callback)
