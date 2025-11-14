from torch.utils.data import Dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig

from .model.deformable_detr.dataloader.od_dataset import ODDataset
from .model.deformable_detr.dataloader.transforms import build_transforms
from .utils import get_experiment_config


if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        experiment_config = get_experiment_config()
        data_source = experiment_config["dataset"]["train_data_sources"][0]
        return ODDataset(
            dataset_dir=data_source["image_dir"],
            json_file=data_source["json_file"],
            transforms=build_transforms(experiment_config["dataset"]["augmentation"]),
        )

    launch_worker(dataset=get_dataset)
