import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import pickle
from utils.verify_media import verify_video_file, verify_wav_file


class VoxCeleb2AllInOnePreprocessor:
    """
    한 번에 모든 전처리를 수행하는 통합 전처리기
    1. 비디오/오디오 처리
    2. 동적 interference 생성
    3. 노이즈 조건별 오디오 생성
    """

    def __init__(self,
                 voxceleb2_root: str,
                 output_root: str,
                 sample_rate: int = 16000,
                 video_fps: int = 25,
                 n_workers: int = 8,
                 noise_conditions: List[int] = [1, 2, 3],
                 dynamic_interference: bool = True):
        """
        Args:
            voxceleb2_root: Root directory of VoxCeleb2 dataset
            output_root: Output directory for processed data
            sample_rate: Target audio sample rate
            video_fps: Target video frame rate
            n_workers: Number of parallel workers
            noise_conditions: List of noise conditions to generate
            dynamic_interference: If True, generate different interference for each file
        """
        self.voxceleb2_root = Path(voxceleb2_root)
        self.output_root = Path(output_root)
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.n_workers = n_workers
        self.noise_conditions = noise_conditions
        self.dynamic_interference = dynamic_interference

        # Noise condition definitions
        self.noise_config = {
            1: {
                'n_noise_sources': 1,
                'n_interference_speakers': 1,
                'snr_range': [0, 0],
                'sir_range': [0, 0]
            },
            2: {
                'n_noise_sources': (0, 5),
                'n_interference_speakers': (0, 3),
                'snr_range': [-10, 5],
                'sir_range': [-10, 5]
            },
            3: {
                'n_noise_sources': 5,
                'n_interference_speakers': 3,
                'snr_range': [-10, -5],
                'sir_range': [-10, -5]
            }
        }

        # Create output directories
        self._create_output_dirs()

        # Will be populated during processing
        self.speaker_audio_map = {}
        self.noise_files = []

    def _create_output_dirs(self):
        """Create output directory structure"""
        dirs = [
            'train/videos',
            'train/audio/clean',
            'test/videos',
            'test/audio/clean',
            'metadata',
            'noise_params'  # For storing noise generation parameters
        ]

        # Add directories for each noise condition
        for condition in self.noise_conditions:
            dirs.extend([
                f'train/audio/noisy_condition_{condition}',
                f'test/audio/noisy_condition_{condition}'
            ])

        for dir_path in dirs:
            (self.output_root / dir_path).mkdir(parents=True, exist_ok=True)

    def preprocess(self):
        """Main preprocessing pipeline - all in one"""
        print("Starting All-in-One VoxCeleb2 preprocessing...")

        # Step 1: Process videos and clean audio
        print("\n[Step 1/4] Processing videos and clean audio...")
        for split in ['dev', 'test']:
            print(f"\nProcessing {split} split...")
            self._process_split(split)

        # Step 2: Build speaker-audio mapping
        print("\n[Step 2/4] Building speaker-audio mapping...")
        self._build_speaker_audio_map()

        # Step 3: Load noise files if available
        print("\n[Step 3/4] Loading noise files...")
        self._load_noise_files()

        # Step 4: Generate all noisy versions
        print("\n[Step 4/4] Generating noisy audio versions...")
        self._generate_all_noisy_audios()

        # Create file lists
        print("\nCreating file lists...")
        self._create_file_lists()

        print("\nAll-in-One preprocessing complete!")

    def _process_split(self, split: str):
        """Process a data split (dev/test) - videos and clean audio only"""
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
                    result = future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")

    def _process_video_audio_pair(self, video_path: Path, split: str) -> Dict:
        """Process video and clean audio only - noise will be added later"""
        # Parse paths and create unique ID
        rel_path = video_path.relative_to(self.voxceleb2_root / split / 'mp4')
        path_parts = rel_path.parts

        video_id_parts = list(path_parts[:-1]) + [rel_path.stem]
        unique_video_id = '_'.join(video_id_parts)

        speaker_id = path_parts[0] if len(path_parts) > 0 else 'unknown'

        # Find audio
        audio_files = self._find_audio_files(video_path, split, speaker_id)

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

        # Process clean audio
        audio_ok = self._process_audio(audio_path, output_audio_path)

        if not (video_ok and audio_ok):
            print(f"Skipping {unique_video_id} due to processing failure.")
            return None

        # Save metadata (including placeholders for noisy audio paths)
        metadata = {
            'unique_id': unique_video_id,
            'video_id': video_path.stem,
            'speaker_id': speaker_id,
            'folder_structure': list(path_parts[:-1]),
            'original_video': str(video_path),
            'original_audio': str(audio_path),
            'processed_video': os.path.join(output_split, 'videos', f"{unique_video_id}.mp4"),
            'processed_audio_clean': os.path.join(output_split, 'audio', 'clean', f"{unique_video_id}.wav"),
            'split': output_split
        }

        # Add noisy audio paths
        for condition in self.noise_conditions:
            metadata[f'processed_audio_noisy_{condition}'] = os.path.join(
                output_split, 'audio', f'noisy_condition_{condition}', f"{unique_video_id}.wav"
            )

        metadata_path = self.output_root / 'metadata' / f"{unique_video_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _generate_all_noisy_audios(self):
        """Generate noisy audio for all files"""
        metadata_files = list((self.output_root / 'metadata').glob('*.json'))

        # Process each condition separately for better progress tracking
        for condition in self.noise_conditions:
            print(f"\nGenerating noise condition {condition}...")

            tasks = []
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                for meta_path in metadata_files:
                    task = executor.submit(
                        self._generate_noisy_for_file_with_dynamic_interference,
                        meta_path, condition
                    )
                    tasks.append(task)

                # Progress bar
                desc = f"Generating noisy audio (condition {condition})"
                for future in tqdm(as_completed(tasks), total=len(tasks), desc=desc):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error generating noisy audio: {e}")

    def _generate_noisy_for_file_with_dynamic_interference(self, metadata_path: Path, condition: int):
        """Generate noisy audio with dynamic interference for each file"""
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load clean audio
        clean_path = self.output_root / metadata['processed_audio_clean']
        if not clean_path.exists():
            return

        clean_audio, sr = librosa.load(clean_path, sr=self.sample_rate)

        # Generate unique seed for reproducibility
        seed = hash(f"{metadata['unique_id']}_{condition}") % (2 ** 32)
        np.random.seed(seed)
        random.seed(seed)

        # Create noisy mixture
        if self.dynamic_interference:
            noisy_audio = self._create_noisy_with_dynamic_interference(
                clean_audio, metadata, condition
            )
        else:
            # Fallback to simple random mixing
            noisy_audio = self._create_noisy_simple(clean_audio, condition)

        # Save noisy audio
        noisy_path = self.output_root / metadata[f'processed_audio_noisy_{condition}']
        noisy_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize
        if np.abs(noisy_audio).max() > 0:
            noisy_audio = noisy_audio / np.abs(noisy_audio).max() * 0.95

        sf.write(noisy_path, noisy_audio, self.sample_rate)

        # Optionally save noise parameters for reproducibility
        self._save_noise_parameters(metadata['unique_id'], condition, seed)

    def _create_noisy_with_dynamic_interference(self, clean_audio: np.ndarray,
                                                metadata: Dict, condition: int) -> np.ndarray:
        """Create noisy audio with dynamic interference from other speakers"""
        config = self.noise_config[condition]
        noisy = clean_audio.copy()

        # Add interference from other speakers
        n_interference = config['n_interference_speakers']
        if isinstance(n_interference, (list, tuple)):
            # 설정값이 리스트나 튜플이면, 해당 범위 내에서 랜덤 정수 선택
            n_interference = random.randint(n_interference[0], n_interference[1])

        if n_interference > 0:
            # Get other speakers' audio files
            speaker_id = metadata['speaker_id']
            split = metadata['split']

            # Get list of other audio files
            other_files = []
            for other_speaker, files in self.speaker_audio_map.items():
                if other_speaker != speaker_id:
                    # Filter by split to avoid train/test contamination
                    split_files = [f for f in files if split in str(f)]
                    other_files.extend(split_files)

            if len(other_files) >= n_interference:
                # Randomly select interference files
                selected_files = random.sample(other_files, n_interference)

                # Mix interference
                interference_mix = np.zeros_like(clean_audio)
                for file_path in selected_files:
                    try:
                        audio, _ = librosa.load(file_path, sr=self.sample_rate)

                        # Random segment if longer than target
                        if len(audio) > len(clean_audio):
                            start = random.randint(0, len(audio) - len(clean_audio))
                            audio = audio[start:start + len(clean_audio)]
                        elif len(audio) < len(clean_audio):
                            # Repeat if shorter
                            n_repeat = len(clean_audio) // len(audio) + 1
                            audio = np.tile(audio, n_repeat)[:len(clean_audio)]

                        # Random gain for each interference
                        gain = random.uniform(0.5, 1.0)
                        interference_mix += audio * gain
                    except Exception as e:
                        print(f"Error loading interference file {file_path}: {e}")

                # Normalize interference mix
                if n_interference > 0:
                    interference_mix /= n_interference

                # Apply SIR
                sir_db = random.uniform(*config['sir_range'])
                clean_power = np.mean(clean_audio ** 2)
                interference_power = np.mean(interference_mix ** 2)

                if interference_power > 0:
                    target_interference_power = clean_power / (10 ** (sir_db / 10))
                    scale = np.sqrt(target_interference_power / interference_power)
                    noisy += interference_mix * scale

        # Add background noise
        noisy = self._add_background_noise(noisy, clean_audio, config)

        return noisy

    def _create_noisy_simple(self, clean_audio: np.ndarray, condition: int) -> np.ndarray:
        """Simple noise addition without dynamic interference"""
        config = self.noise_config[condition]
        noisy = clean_audio.copy()

        # Add white noise as interference
        n_interference = config['n_interference_speakers']
        if n_interference > 0:
            # Generate synthetic interference
            interference = np.random.randn(len(clean_audio)) * 0.1

            sir_db = random.uniform(*config['sir_range'])
            clean_power = np.mean(clean_audio ** 2)
            interference_power = np.mean(interference ** 2)

            if interference_power > 0:
                target_interference_power = clean_power / (10 ** (sir_db / 10))
                scale = np.sqrt(target_interference_power / interference_power)
                noisy += interference * scale

        # Add background noise
        noisy = self._add_background_noise(noisy, clean_audio, config)

        return noisy

    def _add_background_noise(self, noisy: np.ndarray, clean_audio: np.ndarray,
                              config: Dict) -> np.ndarray:
        """Add background noise from noise files"""
        n_noise = config['n_noise_sources']
        if isinstance(n_noise, (list, tuple)):
            # 설정값이 리스트나 튜플이면, 해당 범위 내에서 랜덤 정수 선택
            n_noise = random.randint(n_noise[0], n_noise[1])

        if n_noise > 0 and len(self.noise_files) > 0:
            noise_mix = np.zeros_like(clean_audio)

            for _ in range(min(n_noise, len(self.noise_files))):
                noise_file = random.choice(self.noise_files)
                try:
                    noise, _ = librosa.load(noise_file, sr=self.sample_rate)

                    # Adjust length
                    if len(noise) > len(clean_audio):
                        start = random.randint(0, len(noise) - len(clean_audio))
                        noise = noise[start:start + len(clean_audio)]
                    elif len(noise) < len(clean_audio):
                        n_repeat = len(clean_audio) // len(noise) + 1
                        noise = np.tile(noise, n_repeat)[:len(clean_audio)]

                    noise_mix += noise / n_noise
                except Exception as e:
                    print(f"Error loading noise file {noise_file}: {e}")

            # Apply SNR
            snr_db = random.uniform(*config['snr_range'])
            clean_power = np.mean(clean_audio ** 2)
            noise_power = np.mean(noise_mix ** 2)

            if noise_power > 0:
                target_noise_power = clean_power / (10 ** (snr_db / 10))
                scale = np.sqrt(target_noise_power / noise_power)
                noisy += noise_mix * scale

        return noisy

    def _save_noise_parameters(self, unique_id: str, condition: int, seed: int):
        """Save noise generation parameters for reproducibility"""
        param_file = self.output_root / 'noise_params' / f"{unique_id}_condition_{condition}.json"
        param_file.parent.mkdir(exist_ok=True)

        params = {
            'unique_id': unique_id,
            'condition': condition,
            'seed': seed,
            'timestamp': str(Path.ctime(param_file.parent))
        }

        with open(param_file, 'w') as f:
            json.dump(params, f)

    def _build_speaker_audio_map(self):
        """Build mapping of speaker to audio files"""
        self.speaker_audio_map = {}

        for split in ['train', 'test']:
            audio_dir = self.output_root / split / 'audio' / 'clean'
            if not audio_dir.exists():
                continue

            for audio_file in tqdm(list(audio_dir.glob('*.wav')),
                                   desc=f"Mapping {split} audio files"):
                unique_id = audio_file.stem
                meta_file = self.output_root / 'metadata' / f'{unique_id}.json'

                if meta_file.exists():
                    with open(meta_file) as f:
                        metadata = json.load(f)
                    speaker_id = metadata.get('speaker_id', 'unknown')

                    if speaker_id not in self.speaker_audio_map:
                        self.speaker_audio_map[speaker_id] = []
                    self.speaker_audio_map[speaker_id].append(audio_file)

        print(f"Mapped {len(self.speaker_audio_map)} speakers")

        # Save mapping for faster future runs
        cache_file = self.output_root / 'speaker_audio_map.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(self.speaker_audio_map, f)

    def _load_noise_files(self):
        """Load paths to noise files"""
        self.noise_files = []

        # Check for noise file lists
        for split in ['train', 'test']:
            noise_list = self.output_root / f"noise_{split}.txt"
            if noise_list.exists():
                with open(noise_list) as f:
                    for line in f:
                        path = Path(line.strip())
                        if path.exists():
                            self.noise_files.append(path)

        print(f"Loaded {len(self.noise_files)} noise files")

    def _find_audio_files(self, video_path: Path, split: str, speaker_id: str) -> List[Path]:
        """Find corresponding audio files"""
        audio_files = []
        video_name = video_path.stem

        if split == 'dev':
            audio_dir = self.voxceleb2_root / split / 'aac' / speaker_id
            if audio_dir.exists():
                audio_files = list(audio_dir.rglob(f'*{video_name}*.m4a'))
        else:
            # For test split
            test_audio_dir = self.voxceleb2_root / 'test' / 'aac'
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
        """Process video file"""
        if output_path.exists() and verify_video_file(str(output_path)):
            return True

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            'ffmpeg', '-i', str(input_path), '-r', str(self.video_fps),
            '-vf', 'crop=96:96:64:64', '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '18', '-an', '-y', str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error processing video {input_path}")
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return False

    def _process_audio(self, input_path: Path, output_path: Path) -> bool:
        """Process audio file"""
        if output_path.exists() and verify_wav_file(str(output_path)):
            return True

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                'ffmpeg', '-i', str(input_path), '-ar', str(self.sample_rate),
                '-ac', '1', '-y', str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Normalize audio
            audio, sr = librosa.load(output_path, sr=self.sample_rate)
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max() * 0.95

            sf.write(output_path, audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error processing audio {input_path}: {e}")
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return False

    def _create_file_lists(self):
        """Create file lists for each condition"""
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

                            if split in metadata['processed_video']:
                                video_path = self.output_root / metadata['processed_video']

                                if condition_name == 'clean':
                                    audio_path = self.output_root / metadata['processed_audio_clean']
                                else:
                                    condition_num = condition_name.split('_')[-1]
                                    audio_path = self.output_root / metadata[f'processed_audio_noisy_{condition_num}']

                                if video_path.exists() and audio_path.exists():
                                    valid_pairs.append(metadata)
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Warning: Skipping malformed metadata file {meta_path}")

                # Write file list
                list_path = self.output_root / f"{condition_name}_{split}.txt"
                with open(list_path, 'w') as f:
                    for pair in sorted(valid_pairs, key=lambda x: x['unique_id']):
                        f.write(f"{pair['processed_video']}\n")

                print(f"Created {split} {condition_name} file list with {len(valid_pairs)} entries")


def create_dns_noise_list(dns_root: str, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    """Create noise file list from DNS Challenge dataset"""
    dns_path = Path(dns_root)
    noise_files = []

    noise_dirs = ['./', 'noise', 'noise_train', 'datasets/noise']

    for noise_dir in noise_dirs:
        noise_path = dns_path / noise_dir
        if noise_path.exists():
            noise_files.extend(noise_path.rglob('*.wav'))

    with open(output_path, 'w') as f:
        for file_path in noise_files:
            f.write(f"{file_path}\n")

    print(f"Found {len(noise_files)} noise files from DNS dataset")


def main():
    parser = argparse.ArgumentParser(description='All-in-One VoxCeleb2 Preprocessing')
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
    parser.add_argument('--noise_conditions', type=int, nargs='+', default=[1, 2, 3],
                        help='Noise conditions to generate (1, 2, 3)')
    parser.add_argument('--dynamic_interference', action='store_true',
                        help='Use dynamic interference generation (recommended)')

    args = parser.parse_args()


    # Run all-in-one preprocessing
    preprocessor = VoxCeleb2AllInOnePreprocessor(
        voxceleb2_root=args.voxceleb2_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        video_fps=args.video_fps,
        n_workers=args.n_workers,
        noise_conditions=args.noise_conditions,
        dynamic_interference=args.dynamic_interference
    )

    # Create DNS noise list if provided
    if args.dns_root:
        noise_list_path = Path(args.output_root) / 'noise_train.txt'
        create_dns_noise_list(args.dns_root, noise_list_path)

        # Copy for test
        shutil.copy2(noise_list_path, Path(args.output_root) / 'noise_test.txt')


    preprocessor.preprocess()


if __name__ == '__main__':
    main()
