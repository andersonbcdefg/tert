from modal import Image, App, gpu, Volume, Secret
from src.bitbert.data import (
    download_dclm_data,
    download_fineweb_data,
    download_wiki_data,
)

DATA_DIR = "/data"
MOUNT_PATH = "/training_outputs"


def download_dclm():
    download_dclm_data(DATA_DIR)


def download_fineweb():
    download_fineweb_data(DATA_DIR)


def download_wiki():
    download_wiki_data(DATA_DIR)


vol = Volume.from_name("bitbert-outputs", create_if_missing=True)


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
        "heavyball",
        "pydantic>=2.0",
    )
    .pip_install(
        "torchao@git+https://github.com/pytorch/ao.git",
        "muon@git+https://github.com/KellerJordan/Muon.git",
    )
    .run_function(download_dclm, secrets=[Secret.from_name("HF-SECRET")])
    .run_function(download_fineweb, secrets=[Secret.from_name("HF-SECRET")])
    .run_function(download_wiki, secrets=[Secret.from_name("HF-SECRET")])
    .run_commands("pip uninstall -y liger-kernel")
    .pip_install(
        "liger-kernel@git+https://github.com/andersonbcdefg/Liger-Kernel.git@a874bfe"
    )
    .pip_install(
        "cut-cross-entropy@git+https://github.com/andersonbcdefg/ml-cross-entropy.git"
    )
    .pip_install("adam-mini")
    .env({"TORCH_TRACE": "/training_outputs/trace"})
)

app = App("train-bitbert")


@app.function(
    image=image,
    gpu=gpu.H100(),
    timeout=60 * 60 * 24,
    volumes={MOUNT_PATH: vol},
    secrets=[Secret.from_name("HF-SECRET")],
)
def train(job_id: str):
    vol.reload()
    from src.bitbert.train import train, TrainArgs

    train(job_id, TrainArgs(), MOUNT_PATH, DATA_DIR, save_callback)
