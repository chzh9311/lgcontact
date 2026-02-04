import os
import os.path as osp
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from lightning import LightningDataModule

MANO_ROOT = '/home/zxc417/data/mano_v1_2/models'


class HOI4DHandPoseDataset(Dataset):
    """
    PyTorch Dataset for HOI4D hand poses.

    Each sample is a single hand (left or right) with its MANO parameters.
    If a frame has both hands, they are treated as separate samples.

    Args:
        preprocessed_file (str): Path to preprocessed .npz file containing hand poses
    """

    def __init__(self, preprocessed_file):
        if not osp.exists(preprocessed_file):
            raise ValueError(f"Preprocessed file does not exist: {preprocessed_file}")

        data = np.load(preprocessed_file)
        self.theta = data['theta']  # (N, 48) - pose coefficients
        self.beta = data['beta']    # (N, 10) - shape coefficients
        self.trans = data['trans']  # (N, 3) - translation
        self.side = data['side']    # (N,) - 0 for left, 1 for right

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return {
            'theta': torch.from_numpy(self.theta[idx]).float(),
            'beta': torch.from_numpy(self.beta[idx]).float(),
            'trans': torch.from_numpy(self.trans[idx]).float(),
            'side': self.side[idx],  # 0 = left, 1 = right
        }


class HOI4DHandDataModule(LightningDataModule):
    """
    Lightning DataModule for HOI4D hand pose data.

    Handles preprocessing of raw HOI4D data into a format suitable for training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data
        self.data_root = Path(cfg.data.dataset_path)
        self.preprocessed_dir = Path(cfg.data.get('preprocessed_dir', 'data/preprocessed'))
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size

        # Load subdir list from release.txt if provided
        self.release_file = self.data_root / cfg.data.get('release_file', 'release.txt')

    def _load_subdir_list(self):
        """Load list of subdirectories from release.txt"""
        if not self.release_file.exists():
            raise ValueError(f"Release file not found: {self.release_file}")

        with open(self.release_file, 'r') as f:
            subdirs = [line.strip() for line in f if line.strip()]
        return subdirs

    def _split_subdirs(self, subdirs, train_ratio=0.8, val_ratio=0.1):
        """Split subdirectories into train/val/test sets"""
        n = len(subdirs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Shuffle with fixed seed for reproducibility
        rng = np.random.default_rng(42)
        indices = rng.permutation(n)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        return {
            'train': [subdirs[i] for i in train_indices],
            'val': [subdirs[i] for i in val_indices],
            'test': [subdirs[i] for i in test_indices],
        }

    def _get_hand_path(self, subdir, frame_idx, side):
        """Get path to hand pickle file"""
        side_dir = f"handpose_{side}_hand"
        return self.data_root / "Hand_pose" / side_dir / subdir / f"{frame_idx}.pickle"

    def _load_hand_params(self, pkl_path):
        """
        Load raw MANO parameters from pickle file.

        Returns:
            dict with 'theta' (48,), 'beta' (10,), 'trans' (3,) or None if loading fails
        """
        if not pkl_path.exists():
            return None

        try:
            with open(pkl_path, 'rb') as f:
                hand_info = pickle.load(f, encoding='latin1')

            theta = np.array(hand_info['poseCoeff'], dtype=np.float32)  # (48,)
            beta = np.array(hand_info['beta'], dtype=np.float32)        # (10,)
            trans = np.array(hand_info['trans'], dtype=np.float32)      # (3,)

            # Convert trans from mm to meters
            trans = trans / 1000.0

            return {'theta': theta, 'beta': beta, 'trans': trans}
        except Exception as e:
            print(f"Warning: Failed to load hand data from {pkl_path}: {e}")
            return None

    def _scan_available_frames(self, subdir):
        """
        Scan for available frames in a subdirectory.
        Returns list of frame indices.
        """
        # Try both handpose directories
        left_dir = self.data_root / "Hand_pose" / "handpose_left_hand" / subdir
        right_dir = self.data_root / "Hand_pose" / "handpose_right_hand" / subdir

        frames = set()

        if left_dir.exists():
            for pkl_file in left_dir.glob("*.pickle"):
                try:
                    frame_idx = int(pkl_file.stem)
                    frames.add(frame_idx)
                except ValueError:
                    continue

        if right_dir.exists():
            for pkl_file in right_dir.glob("*.pickle"):
                try:
                    frame_idx = int(pkl_file.stem)
                    frames.add(frame_idx)
                except ValueError:
                    continue

        return sorted(frames)

    def prepare_data(self):
        """
        Preprocess HOI4D hand data and save to disk.

        Scans all subdirectories for hand pose files and extracts MANO parameters.
        Each hand (left/right) is saved as a separate sample.
        """
        preprocessed_base = self.preprocessed_dir / 'hoi4d'
        os.makedirs(preprocessed_base, exist_ok=True)

        # Load and split subdirectories
        subdirs = self._load_subdir_list()
        splits = self._split_subdirs(subdirs)

        for split_name, split_subdirs in splits.items():
            output_file = preprocessed_base / f'{split_name}_hand_poses.npz'

            if output_file.exists():
                print(f"Preprocessed file already exists: {output_file}")
                continue

            print(f"Preprocessing {split_name} split with {len(split_subdirs)} subdirectories...")

            all_theta = []
            all_beta = []
            all_trans = []
            all_side = []

            for subdir in tqdm(split_subdirs, desc=f"Processing {split_name}"):
                frames = self._scan_available_frames(subdir)

                for frame_idx in frames:
                    # Try loading left hand
                    left_path = self._get_hand_path(subdir, frame_idx, 'left')
                    left_params = self._load_hand_params(left_path)
                    if left_params is not None:
                        if left_params['theta'].shape != (48,):
                            print(f"Warning: Unexpected theta shape in {left_path}: {left_params['theta'].shape}")
                            continue
                        if left_params['beta'].shape != (10,):
                            print(f"Warning: Unexpected beta shape in {left_path}: {left_params['beta'].shape}")
                            continue
                        if left_params['trans'].shape != (3,):
                            print(f"Warning: Unexpected trans shape in {left_path}: {left_params['trans'].shape}")
                            continue

                        all_theta.append(left_params['theta'])
                        all_beta.append(left_params['beta'])
                        all_trans.append(left_params['trans'])
                        all_side.append(0)  # 0 = left

                    # Try loading right hand
                    right_path = self._get_hand_path(subdir, frame_idx, 'right')
                    right_params = self._load_hand_params(right_path)
                    if right_params is not None:
                        if right_params['theta'].shape != (48,):
                            print(f"Warning: Unexpected theta shape in {right_path}: {right_params['theta'].shape}")
                            continue
                        if right_params['beta'].shape != (10,):
                            print(f"Warning: Unexpected beta shape in {right_path}: {right_params['beta'].shape}")
                            continue
                        if right_params['trans'].shape != (3,):
                            print(f"Warning: Unexpected trans shape in {right_path}: {right_params['trans'].shape}")
                            continue
                        all_theta.append(right_params['theta'])
                        all_beta.append(right_params['beta'])
                        all_trans.append(right_params['trans'])
                        all_side.append(1)  # 1 = right

            if len(all_theta) == 0:
                print(f"Warning: No hand data found for {split_name} split")
                continue

            # Stack and save
            theta = np.stack(all_theta, axis=0)
            beta = np.stack(all_beta, axis=0)
            trans = np.stack(all_trans, axis=0)
            side = np.array(all_side, dtype=np.int32)

            np.savez_compressed(
                output_file,
                theta=theta,
                beta=beta,
                trans=trans,
                side=side
            )
            print(f"Saved {len(theta)} hand samples to {output_file}")

    def setup(self, stage: str):
        preprocessed_base = self.preprocessed_dir / 'hoi4d'

        if stage == 'fit':
            self.train_set = HOI4DHandPoseDataset(preprocessed_base / 'train_hand_poses.npz')
            self.val_set = HOI4DHandPoseDataset(preprocessed_base / 'val_hand_poses.npz')
        elif stage == 'validate':
            self.val_set = HOI4DHandPoseDataset(preprocessed_base / 'val_hand_poses.npz')
        elif stage == 'test':
            self.test_set = HOI4DHandPoseDataset(preprocessed_base / 'test_hand_poses.npz')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.get('num_workers', 4)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.get('num_workers', 4)
        )

