from torch.utils.data import Dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig

from .dataloader.od_dataset import ODDataset
from .dataloader.transforms import build_transforms


# import os
# import sys
# import subprocess

# def _maybe_wrap_with_nsys():
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     if local_rank != 1:
#         return  # Only rank 0 should use nsys

#     if os.environ.get("NSYS_ACTIVE", "0") == "1":
#         return  # Already inside nsys

#     # Relaunch this script under nsys
#     nsys_cmd = [
#         "nsys", "profile",
#         "--output=outputs/cradio-v3-test/nsys_profile",
#         "--trace=cuda,nvtx,osrt",
#         "--force-overwrite=true",
#         "--capture-range=cudaProfilerApi",
#         "--capture-range-end=stop",
#         sys.executable, "-m", "cosmos_rl.policy.model.c_radio_v3.launcher"
#     ]
#     env = os.environ.copy()
#     env["NSYS_ACTIVE"] = "1"  # Prevent infinite recursion
#     subprocess.run(nsys_cmd, env=env)
#     sys.exit()  # Exit parent process

# _maybe_wrap_with_nsys()

if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        # FIXME - make this a config
        data_source = {
            "image_dir": "/lustre/fsw/portfolios/edgeai/users/scha/data/coco/raw-data/train2017",
            "json_file": "/lustre/fsw/portfolios/edgeai/users/scha/data/coco/raw-data/annotations/instances_train2017.json",
        }
        return ODDataset(
            dataset_dir=data_source["image_dir"],
            json_file=data_source["json_file"],
            transforms=build_transforms(),
        )

    launch_worker(dataset=get_dataset)
