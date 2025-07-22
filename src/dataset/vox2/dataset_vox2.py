import torch
import torchaudio
import torchvision
import logging
from pathlib import Path
from typing import Tuple
import json


class VoxCeleb2Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 noise_condition: int = 2):
        """
        Args:
            data_root: Root directory containing preprocessed VoxCeleb2 data
            split: 'train' or 'test'
            noise_condition: Which noise condition to use (1, 2, or 3)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.NOTSET)

        self.ROOT = Path(data_root)
        self.split = split
        self.noise_condition = noise_condition

        # Load metadata to get file paths
        self.data = self._load_file_list()

        self.logger.debug(f'Loaded {len(self.data)} files for {split}')
        self.logger.debug(f'self.data[0]: {self.data[0]}')

    def _load_file_list(self):
        """Load file list from metadata"""
        data = []
        metadata_dir = self.ROOT / 'metadata'

        for meta_file in metadata_dir.glob('*.json'):
            with open(meta_file) as f:
                metadata = json.load(f)

            # Check if this is for the correct split
            if self.split in metadata['processed_video']:
                # Build file paths
                video_path = metadata['processed_video']
                clean_path = metadata['processed_audio_clean']
                noisy_path = metadata[f'processed_audio_noisy_{self.noise_condition}']

                # Verify files exist
                if (self.ROOT / video_path).exists() and \
                        (self.ROOT / clean_path).exists() and \
                        (self.ROOT / noisy_path).exists():
                    data.append({
                        "id": metadata['unique_id'],
                        "video": video_path,
                        "clean": clean_path,
                        "noisy": noisy_path
                    })

        return data

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            vframes: Video frames tensor [T, C, H, W]
            noisy_waveform: Noisy audio tensor [1, samples]
            clean_waveform: Clean audio tensor [1, samples]
        """
        video_path = self.ROOT / self.data[i]['video']
        noisy_path = self.ROOT / self.data[i]['noisy']
        clean_path = self.ROOT / self.data[i]['clean']

        # Load video
        vframes, aframes, info = torchvision.io.read_video(
            str(video_path), pts_unit='sec', output_format='TCHW'
        )

        # Load audio
        clean_waveform, sr = torchaudio.load(clean_path)
        noisy_waveform, sr = torchaudio.load(noisy_path)

        self.logger.debug(f'vframes.shape: {vframes.shape}')
        self.logger.debug(f'clean_waveform.shape: {clean_waveform.shape}')
        self.logger.debug(f'noisy_waveform.shape: {noisy_waveform.shape}')

        # Normalize video frames to [0, 1]
        vframes = vframes / 255.0

        return vframes, noisy_waveform, clean_waveform

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # Test the dataset
    dataset = VoxCeleb2Dataset(
        data_root='preprocess_vox2_with_noise',
        split='train',
        noise_condition=2
    )

    if len(dataset) > 0:
        vframes, noisy, clean = dataset[0]
        print(f"Video shape: {vframes.shape}")
        print(f"Noisy audio shape: {noisy.shape}")
        print(f"Clean audio shape: {clean.shape}")
        print(f"Sample ID: {dataset.data[0]['id']}")