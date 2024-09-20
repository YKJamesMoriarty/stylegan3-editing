from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import data_utils
from utils.common import get_identity_transform


class InferenceDataset(Dataset):
    '''root 参数是由 --data_path 提供的路径
    landmarks_transforms_path：这是可选参数，存储面部特征点变换（landmarks transforms）的路径。如果给定，它会加载和应用这些变换
    transform：这是对图像应用的变换（例如，归一化、裁剪等）
    '''
    def __init__(self, root: Path, landmarks_transforms_path: Path = None, transform=None):
        #读取root路径下所有的图片文件。
        self.paths = sorted(data_utils.make_dataset(root))
        #调用内部函数 _get_landmarks_transforms 读取特征点变换，并将其保存
        self.landmarks_transforms = self._get_landmarks_transforms(landmarks_transforms_path)
        #self.transform：这个变量存储传入的图像变换函数（例如 transforms.Compose）。
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def _get_landmarks_transforms(self, landmarks_transforms_path):
        if landmarks_transforms_path is not None:
            if not landmarks_transforms_path.exists():
                raise ValueError(f"Invalid path for landmarks transforms: {landmarks_transforms_path}")
            landmarks_transforms = np.load(landmarks_transforms_path, allow_pickle=True).item()
            # filter out images not appearing in landmarks transforms
            valid_files = list(landmarks_transforms.keys())
            self.paths = [f for f in self.paths if f.name in valid_files]
        else:
            landmarks_transforms = None
        return landmarks_transforms

    def _get_transform(self, from_path):
        landmarks_transform = self.landmarks_transforms[from_path.name][-1]
        return landmarks_transform

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.landmarks_transforms is not None:
            landmarks_transform = self._get_transform(from_path)
        else:
            landmarks_transform = get_identity_transform()
        if self.transform:
            from_im = self.transform(from_im)
        return from_im, landmarks_transform
