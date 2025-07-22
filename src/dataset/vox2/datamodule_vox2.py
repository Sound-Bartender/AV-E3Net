import lightning.pytorch as pl
from torch.utils.data import DataLoader
from dataset_vox2 import VoxCeleb2Dataset
import logging
from typing import List, Tuple
from torch import Tensor


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_root: str = '/development/dataset/preprocess_vox2_ave3',
                 batch_size: int = 1,
                 num_workers: int = 12,
                 noise_condition: int = 2):
        """
        Args:
            data_root: Root directory of preprocessed VoxCeleb2 data
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            noise_condition: Noise condition to use (1, 2, or 3)
        """
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.NOTSET)

        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_condition = noise_condition

    def setup(self, stage: str):
        if stage == "fit":
            self.train = VoxCeleb2Dataset(
                data_root=self.data_root,
                split='train',
                noise_condition=self.noise_condition
            )
            self.valid = VoxCeleb2Dataset(
                data_root=self.data_root,
                split='test',  # Using test as validation
                noise_condition=self.noise_condition
            )
        if stage == 'test':
            self.test = VoxCeleb2Dataset(
                data_root=self.data_root,
                split='test',
                noise_condition=self.noise_condition
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True
        )

    def collate_fn(self, batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        '''
        Because train/test items are of different length, they need to be regrouped
        before passing to the model to avoid different sizes in a batch.

        Dataset returns items of type: (vframes, noisy_waveform, clean_waveform)

        Args:
            batch: list (of size batch_size) of tuples returned by dataset

        Returns:
            Regrouped tuple of lists
        '''
        self.logger.debug(
            f'collate input: batch[0][0].shape {batch[0][0].shape}, '
            f'batch[0][1].shape {batch[0][1].shape}, '
            f'batch[0][2].shape {batch[0][2].shape}'
        )

        video = [item[0] for item in batch]
        noisy = [item[1] for item in batch]
        clean = [item[2] for item in batch]

        self.logger.debug(
            f'collate output: video[0].shape={video[0].shape}, '
            f'noisy[0].shape={noisy[0].shape}, '
            f'clean[0].shape={clean[0].shape}'
        )

        return video, noisy, clean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='preprocess_vox2_with_noise')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--noise_condition', type=int, default=2)
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and test datamodule
    datamodule = VoxCeleb2DataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        noise_condition=args.noise_condition
    )

    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()

    # Get one batch
    i = iter(train_dataloader)
    batch: Tuple[List[Tensor], List[Tensor], List[Tensor]] = next(i)
    vframes_list, noisy_list, clean_list = batch

    print(f"\nBatch info:")
    print(f"Batch size: {len(vframes_list)}")
    print(f"First video shape: {vframes_list[0].shape}")
    print(f"First noisy audio shape: {noisy_list[0].shape}")
    print(f"First clean audio shape: {clean_list[0].shape}")

    # Show different lengths in batch
    if len(vframes_list) > 1:
        print(f"\nDifferent lengths in batch:")
        for i in range(min(len(vframes_list), 3)):
            print(f"  Item {i}: video={vframes_list[i].shape[0]} frames, "
                  f"audio={noisy_list[i].shape[1]} samples")