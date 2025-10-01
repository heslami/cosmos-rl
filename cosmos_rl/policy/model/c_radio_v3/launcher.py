from torch.utils.data import Dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig

from .dataloader.od_dataset import ODDataset
from .dataloader.transforms import build_transforms


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
