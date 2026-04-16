import os
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')
    
class Dataset_DS(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='pendulum.csv', downsample_factor=None):
        # size [seq_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.downsample_factor = downsample_factor
        self.__read_data__()

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path))
        data = data.astype(np.float32)
        if self.downsample_factor is None:
            # Determine default downsampling factor based on filename
            lower_path = self.data_path.lower()
            if 'pendulum' in lower_path:
                downsample = 5
            elif 'ks' in lower_path or 'kuramoto' in lower_path:
                downsample = 4
            else:
                downsample = 1
        else:
            downsample = self.downsample_factor
        
        if downsample > 1:
            if data.ndim == 2:         
                data = data[:, ::downsample]
            elif data.ndim == 3:         
                data = data[:, ::downsample, :]
            else:
                raise ValueError(f"Unexpected data dimension: {data.ndim}")
            
        # Compute trajectory partition indices
        k = data.shape[0]  # number of trajectories
        l = data.shape[1]  # length of each trajectory

        # Split dataset by trajectory index into train/val/test
        num_train = int(k * 0.7)
        num_test = int(k * 0.2)
        num_val = k - num_train - num_test

        borders = {
            0: [0, num_train],
            1: [num_train, num_train + num_val],
            2: [num_train + num_val, k]
        }
        border1, border2 = borders[self.set_type]
        
        self.data = data[border1:border2]

    def __getitem__(self, index):
        # Return a single trajectory segment depending on mode
        trajectory = self.data[index]
        borders = {
            0: [0, self.seq_len],
            1: [0, self.pred_len],
            2: [0, self.pred_len + 1]
        }
        border1, border2 = borders[self.set_type]
        seq = trajectory[border1:border2]
        return seq
    
    def __len__(self):
        return len(self.data)  # 轨迹数量

class DataModule(LightningDataModule):
    def __init__(self, root_path, data_path, batch_size=32, num_workers=6,
                 seq_len=96, pred_len=48, downsample=1):
        super().__init__()
        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = [seq_len, pred_len]
        self.downsample = downsample
    def setup(self, stage=None):
        self.train_dataset = Dataset_DS(
            root_path=self.root_path,
            data_path=self.data_path,
            flag='train',
            size=self.size,
            downsample_factor=self.downsample,
        )
        self.val_dataset = Dataset_DS(
            root_path=self.root_path,
            data_path=self.data_path,
            flag='val',
            size=self.size,
            downsample_factor=self.downsample,
        )
        self.test_dataset = Dataset_DS(
            root_path=self.root_path,
            data_path=self.data_path,
            flag='test',
            size=self.size,
            downsample_factor=self.downsample,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
    
