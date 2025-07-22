import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
import numpy as np
from tqdm import tqdm
import cv2
import librosa
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import torchaudio
from utils.verify_media import verify_video_file, verify_wav_file


class VoxCeleb2PreprocessorWithNoise:
    """Preprocess VoxCeleb2 dataset with noise generation for RT-LA-VocE training"""

    def __init__(self,
                 voxceleb2_root: str,
                 output_root: str,
                 sample_rate: int = 16000,
                 video_fps: int = 25,
                 n_workers: int = 8,
                 n_interference: int = 10,
                 noise_conditions: List[int] = [1, 2, 3]):
        """
        Args:
            voxceleb2_root: Root directory of VoxCeleb2 dataset
            output_root: Output directory for processed data
            sample_rate: Target audio sample rate
            video_fps: Target video frame rate
            n_workers: Number of parallel workers
            noise_conditions: List of noise conditions to generate
        """
        self.voxceleb2_root = Path(voxceleb2_root)
        self.output_root = Path(output_root)
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.n_workers = n_workers
        self.n_interference = n_interference
        self.noise_conditions = noise_conditions

        # Noise condition definitions
        self.noise_config = {
            1: {
                'n_noise_sources': 1,
                'n_interference_speakers': 1,
                'snr_range': [0, 0],  # 배경 잡음 (음수면 라벨보다 큰 것, 양수면 라벨보다 작은 것)
                'sir_range': [0, 0]  # 타 화자 잡음
            },
            2: {
                'n_noise_sources': 3,
                'n_interference_speakers': 2,
                'snr_range': [-5, 5],
                'sir_range': [-5, 5]
            },
            3: {
                'n_noise_sources': 1,
                'n_interference_speakers': (1, 3),
                'snr_range': [-10, 5],
                'sir_range': [-10, 5]
            }
        }

        # Create output directories
        self._create_output_dirs()

    def _create_output_dirs(self):
        """Create output directory structure"""
        dirs = [
            'train/videos',
            'train/audio/clean',
            'train/audio/noisy_condition_1',
            'train/audio/noisy_condition_2',
            'train/audio/noisy_condition_3',
            'train/interference',
            'test/videos',
            'test/audio/clean',
            'test/audio/noisy_condition_1',
            'test/audio/noisy_condition_2',
            'test/audio/noisy_condition_3',
            'test/interference',
            'metadata'
        ]

        for dir_path in dirs:
            (self.output_root / dir_path).mkdir(parents=True, exist_ok=True)

    def preprocess(self):
        """Main preprocessing pipeline"""
        print("Starting VoxCeleb2 preprocessing with noise generation...")

        # Process train and test splits
        for split in ['dev', 'test']:
            print(f"\nProcessing {split} split...")
            self._process_split(split)

        # Create interference speakers
        print("\nCreating interference speaker sets...")
        self._create_interference_sets()

        # Load noise and interference files
        print("\nLoading noise and interference files...")
        self.noise_files = self._load_noise_files()
        self.interference_files = self._load_interference_files()

        # Generate noisy versions
        print("\nGenerating noisy audio versions...")
        for condition in self.noise_conditions:
            print(f"\nGenerating noise condition {condition}...")
            self._generate_noisy_audios(condition)

        # Create file lists
        print("\nCreating file lists...")
        self._create_file_lists()

        print("\nPreprocessing complete!")

    def _process_split(self, split: str):
        """Process a data split (dev/test)"""
        # Get all video files
        video_dir = self.voxceleb2_root / split / 'mp4'
        video_files = list(video_dir.rglob('*.mp4'))

        print(f"Found {len(video_files)} video files in {split}")

        # Process in parallel
        tasks = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            for video_path in video_files:
                task = executor.submit(self._process_video_audio_pair, video_path, split)
                tasks.append(task)

            # Progress bar
            for future in tqdm(as_completed(tasks), total=len(tasks), desc=f"Processing {split}"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")

    def _process_video_audio_pair(self, video_path: Path, split: str) -> Dict:
        """Process a single video-audio pair (clean only)"""
        # Parse video path structure
        rel_path = video_path.relative_to(self.voxceleb2_root / split / 'mp4')
        path_parts = rel_path.parts

        # Create unique ID
        video_id_parts = []
        for part in path_parts[:-1]:  # All folders
            video_id_parts.append(part)
        video_id_parts.append(rel_path.stem)  # Video name without extension

        unique_video_id = '_'.join(video_id_parts)

        # Extract speaker ID (first folder)
        speaker_id = path_parts[0] if len(path_parts) > 0 else 'unknown'

        # Find corresponding audio
        audio_files = []
        if split == 'dev':
            audio_dir = self.voxceleb2_root / split / 'aac' / speaker_id
            if audio_dir.exists():
                video_name = video_path.stem
                audio_files = list(audio_dir.rglob(f'*{video_name}*.m4a'))
        else:
            audio_files = self._find_test_audio_files(video_path, unique_video_id)

        if not audio_files:
            print(f"Warning: No audio found for {video_path}")
            return None

        audio_path = audio_files[0]

        # Output paths
        output_split = 'train' if split == 'dev' else 'test'
        output_video_path = self.output_root / output_split / 'videos' / f"{unique_video_id}.mp4"
        output_audio_path = self.output_root / output_split / 'audio' / 'clean' / f"{unique_video_id}.wav"

        # Process video
        video_ok = self._process_video(video_path, output_video_path)

        # Process clean audio only
        audio_ok = self._process_audio(audio_path, output_audio_path)

        if not (video_ok and audio_ok):
            print(f"Skipping metadata for {unique_video_id} due to processing failure.")
            return None

        # Save metadata
        metadata = {
            'unique_id': unique_video_id,
            'video_id': video_path.stem,
            'speaker_id': speaker_id,
            'folder_structure': list(path_parts[:-1]),
            'original_video': str(video_path),
            'original_audio': str(audio_path),
            'processed_video': os.path.join(output_split, 'videos', f"{unique_video_id}.mp4"),
            'processed_audio_clean': os.path.join(output_split, 'audio', 'clean', f"{unique_video_id}.wav")
        }

        # Add noisy audio paths to metadata
        for condition in self.noise_conditions:
            metadata[f'processed_audio_noisy_{condition}'] = os.path.join(
                output_split, 'audio', f'noisy_condition_{condition}', f"{unique_video_id}.wav"
            )

        metadata_path = self.output_root / 'metadata' / f"{unique_video_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _generate_noisy_audios(self, noise_condition: int):
        """Generate noisy audio files for all clean audios"""
        # Get all metadata files
        metadata_files = list((self.output_root / 'metadata').glob('*.json'))

        # Process in parallel
        tasks = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            for meta_path in metadata_files:
                task = executor.submit(self._generate_noisy_for_file, meta_path, noise_condition)
                tasks.append(task)

            # Progress bar
            desc = f"Generating noisy audio (condition {noise_condition})"
            for future in tqdm(as_completed(tasks), total=len(tasks), desc=desc):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error generating noisy audio: {e}")

    def _generate_noisy_for_file(self, metadata_path: Path, noise_condition: int):
        """Generate noisy audio for a single file"""
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load clean audio
        clean_path = self.output_root / metadata['processed_audio_clean']
        if not clean_path.exists():
            return

        clean_audio, sr = librosa.load(clean_path, sr=self.sample_rate)

        # Generate noisy mixture
        noisy_audio = self._create_noisy_mixture(clean_audio, noise_condition)

        # Save noisy audio
        noisy_path = self.output_root / metadata[f'processed_audio_noisy_{noise_condition}']
        noisy_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize
        if np.abs(noisy_audio).max() > 0:
            noisy_audio = noisy_audio / np.abs(noisy_audio).max() * 0.95

        sf.write(noisy_path, noisy_audio, self.sample_rate)

    def _create_noisy_mixture(self, clean_audio: np.ndarray, noise_condition: int) -> np.ndarray:
        """Create noisy audio mixture"""
        config = self.noise_config[noise_condition]
        noisy = clean_audio.copy()

        # Add interference speakers
        n_interference = config['n_interference_speakers']
        if n_interference > 0 and len(self.interference_files) > 0:
            for _ in range(n_interference):
                # Select random interference file
                interference_file = random.choice(self.interference_files)

                # Load interference audio
                interference, sr = librosa.load(interference_file, sr=self.sample_rate)

                # Match length
                if len(interference) < len(clean_audio):
                    n_repeat = len(clean_audio) // len(interference) + 1
                    interference = np.tile(interference, n_repeat)
                interference = interference[:len(clean_audio)]

                # Apply SIR
                sir_db = random.uniform(*config['sir_range'])

                # Calculate power and scale
                clean_power = np.mean(clean_audio ** 2)
                interference_power = np.mean(interference ** 2)

                if interference_power > 0:
                    target_interference_power = clean_power / (10 ** (sir_db / 10))
                    scale = np.sqrt(target_interference_power / interference_power)
                    interference = interference * scale
                    noisy += interference

        # Add background noise
        n_noise = config['n_noise_sources']
        if n_noise > 0 and len(self.noise_files) > 0:
            noise_mix = np.zeros_like(clean_audio)

            for _ in range(n_noise):
                noise_file = random.choice(self.noise_files)

                # Load noise
                noise, sr = librosa.load(noise_file, sr=self.sample_rate)

                # Match length
                if len(noise) < len(clean_audio):
                    n_repeat = len(clean_audio) // len(noise) + 1
                    noise = np.tile(noise, n_repeat)
                noise = noise[:len(clean_audio)]

                noise_mix += noise / n_noise

            # Apply SNR
            snr_db = random.uniform(*config['snr_range'])

            # Calculate power and scale
            clean_power = np.mean(clean_audio ** 2)
            noise_power = np.mean(noise_mix ** 2)

            if noise_power > 0:
                target_noise_power = clean_power / (10 ** (snr_db / 10))
                scale = np.sqrt(target_noise_power / noise_power)
                noise_mix = noise_mix * scale
                noisy += noise_mix

        return noisy

    def _load_noise_files(self) -> List[Path]:
        """Load noise file paths"""
        noise_files = []

        # Check for noise file lists
        for split in ['train', 'test']:
            noise_list = self.output_root / f"noise_{split}.txt"
            if noise_list.exists():
                with open(noise_list) as f:
                    for line in f:
                        path = Path(line.strip())
                        if path.exists():
                            noise_files.append(path)

        print(f"Loaded {len(noise_files)} noise files")
        return noise_files

    def _load_interference_files(self) -> List[Path]:
        """Load interference file paths"""
        interference_files = []

        for split in ['train', 'test']:
            interference_dir = self.output_root / split / 'interference'
            if interference_dir.exists():
                interference_files.extend(list(interference_dir.glob('*.wav')))

        print(f"Loaded {len(interference_files)} interference files")
        return interference_files[:self.n_interference]  # Limit to n_interference

    def _create_file_lists(self):
        """Create train/test file lists for each noise condition"""
        for split in ['train', 'test']:
            metadata_dir = self.output_root / 'metadata'
            all_metadata_files = list(metadata_dir.glob('*.json'))

            # Create lists for clean and each noise condition
            for condition_name in ['clean'] + [f'noisy_condition_{i}' for i in self.noise_conditions]:
                valid_pairs = []

                for meta_path in all_metadata_files:
                    with open(meta_path) as f:
                        try:
                            metadata = json.load(f)

                            # Check if this metadata is for current split
                            if split in metadata['processed_video']:
                                # Check if all required files exist
                                video_path = self.output_root / metadata['processed_video']

                                if condition_name == 'clean':
                                    audio_path = self.output_root / metadata['processed_audio_clean']
                                else:
                                    condition_num = condition_name.split('_')[-1]
                                    audio_path = self.output_root / metadata[f'processed_audio_noisy_{condition_num}']

                                if video_path.exists() and audio_path.exists():
                                    valid_pairs.append(metadata)
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Warning: Skipping malformed metadata file {meta_path}: {e}")

                # Write file list
                list_path = self.output_root / f"{condition_name}_{split}.txt"
                with open(list_path, 'w') as f:
                    for pair in sorted(valid_pairs, key=lambda x: x['unique_id']):
                        f.write(f"{pair['processed_video']}\n")

                print(f"Created {split} {condition_name} file list with {len(valid_pairs)} entries")

    # Include all the helper methods from the original code
    def _find_test_audio_files(self, video_path: Path, unique_video_id: str) -> List[Path]:
        """Find audio files for test videos"""
        video_name = video_path.stem
        test_audio_dir = self.voxceleb2_root / 'test' / 'aac'

        rel_path = video_path.relative_to(self.voxceleb2_root / 'test' / 'mp4')
        speaker_id = rel_path.parts[0] if len(rel_path.parts) > 0 else None

        audio_files = []

        if speaker_id:
            speaker_audio_dir = test_audio_dir / speaker_id
            if speaker_audio_dir.exists():
                audio_files = list(speaker_audio_dir.rglob(f'*{video_name}*.m4a'))

        if not audio_files:
            for audio_path in test_audio_dir.rglob('*.m4a'):
                if video_name in audio_path.stem:
                    audio_files.append(audio_path)

        return audio_files

    def _process_video(self, input_path: Path, output_path: Path) -> bool:
        """Process video file. Returns True on success, False on failure."""
        if output_path.exists() and not verify_video_file(str(output_path)):
            output_path.unlink(missing_ok=True)
            print(output_path, 'is not valid video')

        if output_path.exists():
            return True

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-r', str(self.video_fps),
            '-vf', 'crop=96:96:64:64',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-an',
            '-y',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ FFMPEG Error processing video {input_path}.")
            if output_path.exists():
                try:
                    output_path.unlink()
                    print(f"   - Removed incomplete file: {output_path}")
                except OSError as unlink_error:
                    print(f"   - Error removing file {output_path}: {unlink_error}")
            return False

    def _process_audio(self, input_path: Path, output_path: Path) -> bool:
        """Process audio file. Returns True on success, False on failure."""
        if output_path.exists() and not verify_wav_file(str(output_path)):
            output_path.unlink(missing_ok=True)
            print(output_path, 'is not valid wav')

        if output_path.exists():
            return True

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                'ffmpeg', '-i', str(input_path), '-ar', str(self.sample_rate),
                '-ac', '1', '-y', str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            audio, sr = librosa.load(output_path, sr=self.sample_rate)
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.95

            sf.write(output_path, audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error processing audio {input_path}: {e}")
            if output_path.exists():
                try:
                    output_path.unlink()
                except OSError:
                    pass
            return False

    def _create_interference_sets(self):
        """Create interference speaker sets from other speakers"""
        for split in ['train', 'test']:
            audio_dir = self.output_root / split / 'audio' / 'clean'
            interference_dir = self.output_root / split / 'interference'

            # Get all audio files
            audio_files = list(audio_dir.glob('*.wav'))

            # Group by speaker
            speaker_groups = {}
            for audio_path in audio_files:
                unique_id = audio_path.stem
                metadata_path = self.output_root / 'metadata' / f"{unique_id}.json"

                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    speaker_id = metadata.get('speaker_id', 'unknown')

                    if speaker_id not in speaker_groups:
                        speaker_groups[speaker_id] = []
                    speaker_groups[speaker_id].append(audio_path)

            # Create interference files
            interference_files = []
            for speaker_id, speaker_files in tqdm(speaker_groups.items()):
                other_speakers = [s for s in speaker_groups.keys() if s != speaker_id]

                if len(other_speakers) >= 3:
                    for i, audio_path in enumerate(speaker_files[:10]):
                        selected_speakers = random.sample(other_speakers, 3)

                        interference_audios = []
                        for other_speaker in selected_speakers:
                            other_files = speaker_groups[other_speaker]
                            if other_files:
                                interference_audios.append(random.choice(other_files))

                        if len(interference_audios) == 3:
                            output_name = f"{audio_path.stem}_interference.wav"
                            output_path = interference_dir / output_name

                            mixed_audio = self._mix_interference_audios(interference_audios)
                            sf.write(output_path, mixed_audio, self.sample_rate)

                            interference_files.append(str(output_path.relative_to(self.output_root)))

            # Write interference file list
            interference_list_path = self.output_root / f"interference_{split}.txt"
            with open(interference_list_path, 'w') as f:
                for file_path in interference_files:
                    f.write(f"{file_path}\n")

            print(f"Created {len(interference_files)} interference files for {split}")

    def _mix_interference_audios(self, audio_paths: List[Path], target_length: int = None) -> np.ndarray:
        """Mix multiple audio files for interference"""
        audios = []

        for path in audio_paths:
            audio, _ = librosa.load(path, sr=self.sample_rate)
            audios.append(audio)

        if target_length is None:
            target_length = max(len(a) for a in audios)

        processed_audios = []
        for audio in audios:
            if len(audio) < target_length:
                n_repeat = target_length // len(audio) + 1
                audio = np.tile(audio, n_repeat)[:target_length]
            else:
                audio = audio[:target_length]
            processed_audios.append(audio)

        weights = np.random.dirichlet(np.ones(len(processed_audios)))
        mixed = np.zeros(target_length)

        for audio, weight in zip(processed_audios, weights):
            mixed += audio * weight

        if np.abs(mixed).max() > 0:
            mixed = mixed / np.abs(mixed).max() * 0.95

        return mixed


def create_dns_noise_list(dns_root: str, output_path: str):
    """Create noise file list from DNS Challenge dataset"""
    dns_path = Path(dns_root)
    noise_files = []

    noise_dirs = ['noise', 'noise_train', 'datasets/noise']

    for noise_dir in noise_dirs:
        noise_path = dns_path / noise_dir
        if noise_path.exists():
            noise_files.extend(noise_path.rglob('*.wav'))

    with open(output_path, 'w') as f:
        for file_path in noise_files:
            f.write(f"{file_path}\n")

    print(f"Found {len(noise_files)} noise files from DNS dataset")


def main():
    parser = argparse.ArgumentParser(description='Preprocess VoxCeleb2 with noise generation')
    parser.add_argument('--voxceleb2_root', type=str, required=True,
                        help='Root directory of VoxCeleb2 dataset')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--dns_root', type=str, default=None,
                        help='Root directory of DNS Challenge dataset (optional)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Target audio sample rate')
    parser.add_argument('--video_fps', type=int, default=25,
                        help='Target video frame rate')
    parser.add_argument('--n_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--n_interference', type=int, default=10,
                        help='Number of interference files to use')
    parser.add_argument('--noise_conditions', type=int, nargs='+', default=[1, 2, 3],
                        help='Noise conditions to generate (1, 2, 3)')

    args = parser.parse_args()

    # Create DNS noise list if provided
    if args.dns_root:
        noise_list_path = Path(args.output_root) / 'noise_train.txt'
        create_dns_noise_list(args.dns_root, noise_list_path)

        # Copy for test
        shutil.copy2(noise_list_path, Path(args.output_root) / 'noise_test.txt')

    # Preprocess VoxCeleb2 with noise generation
    preprocessor = VoxCeleb2PreprocessorWithNoise(
        voxceleb2_root=args.voxceleb2_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        video_fps=args.video_fps,
        n_workers=args.n_workers,
        n_interference=args.n_interference,
        noise_conditions=args.noise_conditions
    )

    preprocessor.preprocess()


if __name__ == '__main__':
    main()