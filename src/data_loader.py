
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Scaling function
def scaleclip_fn(x):
    return np.clip(x / np.max(x), a_min=-0.2, a_max=None)

# Scaling and normalizing function Maps to [-1, 1]
def norm1_scaleclip_fn(x):
    x = np.clip(x / np.max(x), a_min=-0.2, a_max=None)
    return -1 + 2 * (x + 0.2) / 1.2

# Scaling and normalizing function Maps to [0, 1]
def norm2_scaleclip_fn(x):
    x = np.clip(x / np.max(x), a_min=-0.2, a_max=None)
    return (x + 0.2) / 1.2
    
# Define normalization function mappings
transforms_mapping = {
    "scaleclip": scaleclip_fn,
    "norm1_scaleclip": norm1_scaleclip_fn,
    "norm2_scaleclip": norm2_scaleclip_fn
}

# Function to create DataLoader
def create_dataloader(oadat_dir, file_name, key, norm, batch_size, shuffle=True, num_workers=0, drop_last=True, prng=None, indices=None):
    fname_h5 = os.path.join(oadat_dir, file_name)
    
    # Select the transform function based on `norm`
    transform_fn = transforms_mapping[norm]
    
    # Initialize Dataset with the chosen transform
    dataset = Dataset(fname_h5=fname_h5, key=key, transforms=transform_fn, inds=indices, shuffle=shuffle, prng=prng)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return dataloader



class Dataset:
    def __init__(self, fname_h5, key, transforms, inds, shuffle=False, **kwargs):
        self.fname_h5 = fname_h5
        self.key = key
        self.inds = inds
        self.transforms = transforms
        self.shuffle = shuffle
        self.prng = kwargs.get('prng', np.random.RandomState(42))
        self.len = None
        self._check_data()
    
    def _check_data(self,):
        len_ = None
        l_keys = self.key if isinstance(self.key, list) else [self.key]
        with h5py.File(self.fname_h5, 'r') as fh:
            for k in l_keys:
                if len_ is None: 
                    len_ = fh[k].shape[0]
                if len_ != fh[k].shape[0]:
                    raise AssertionError('Length of datasets vary across keys. %d vs %d' % (len_, fh[k].shape[0]))
        if self.inds is None:
            self.len = len_
            self.inds = np.arange(len_)
        else:
            self.len = len(self.inds)

    def __len__(self):
        return self.len

    def __getitem__(self, index): 
        with h5py.File(self.fname_h5, 'r') as fh:
            x = fh[self.key][index,...]
            x = x[None,...] ## add a channel dimension [1, H, W]
            if self.transforms is not None:
                x = self.transforms(x)
        return x

    def __iter__(self):
        inds = np.copy(self.inds)
        if self.shuffle:
            self.prng.shuffle(inds)
        for i in inds:
            s = self.__getitem__(index=i)
            yield s

