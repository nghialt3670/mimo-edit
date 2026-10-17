
import hashlib

from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO = "fal/flux-2-klein-4B-object-remove-lora"
WEIGHT = (
    "kDEkt5q7tDLKOpQJIVMPx_"
    "pytorch_lora_weights_comfy_converted.safetensors"
)

path = hf_hub_download(
    repo_id=REPO,
    filename=WEIGHT,
)

with open(path, "rb") as file:
    sha256 = hashlib.file_digest(file, "sha256").hexdigest()

print("SHA-256:", sha256)

with safe_open(path, framework="pt", device="cpu") as file:
    keys = list(file.keys())
    print("Checkpoint is valid")
    print("Tensor count:", len(keys))
    print("First 10 keys:")
    for key in keys[:10]:
        print(" ", key)
