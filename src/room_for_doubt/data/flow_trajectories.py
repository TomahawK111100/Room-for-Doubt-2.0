"""CFM shares the validated conditioned-trajectory dataset."""

from room_for_doubt.data.conditioned_trajectories import CondTrajDataset as _Dataset


class CondTrajDataset(_Dataset):
    def __init__(
        self,
        dataset_name="cifar100",
        root_dir="./data",
        traj_file="",
        transform=None,
        seed=42,
        val_ratio=0.1,
        start_epoch=0,
        missing_epoch_policy="drop",
    ):
        if dataset_name != "cifar100":
            raise ValueError("The current CFM command supports cifar100 only")
        super().__init__(
            dataset_name,
            root_dir,
            traj_file,
            transform,
            seed,
            val_ratio,
            start_epoch,
            missing_epoch_policy,
        )
