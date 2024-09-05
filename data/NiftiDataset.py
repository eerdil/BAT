import torch.utils.data as data
import numpy as np
import os
import nibabel as nib
import pandas as pd

class NiftiDataset(data.Dataset):
    """
    Input params:
        path: Path to the folder containing the dataset (one or multiple HDF5 files).
        transform: PyTorch transform to apply to every data instance (default = None).
    """
    def __init__(self, path, transform = None):
        super().__init__()
        self.transform = transform
        self.path = path
        
        # Note that correspongind PET/CT scans should be in the same index after sorting!
        self.ct_names = [s for s in os.listdir(path) if ('_CT_' in s)]
        self.ct_names.sort()
        
        self.pet_names = [s for s in os.listdir(path) if ('_PET_' in s)]
        self.pet_names.sort()
        
        # Store the min/max/99th percentile values of each subjects in a csv file
        path_ct = '%s/ct_min_max_percentile.csv'%(path)
        path_pet = '%s/pet_min_max_percentile.csv'%(path)

        df_ct = pd.read_csv(path_ct)
        df_pet = pd.read_csv(path_pet)

        self.patient_names = df_ct['name']
        self.ct_min, self.ct_max, self.ct_99_percentile = df_ct['min'], df_ct['max'], df_ct['99_percentile']
        self.pet_min, self.pet_max, self.pet_99_percentile = df_pet['min'], df_pet['max'], df_pet['99_percentile']

                    
    def __getitem__(self, index):
        # Read data from Nifti files
        ct = nib.load('%s/%s'%(self.path, self.ct_names[index])).get_fdata()
        pet = nib.load('%s/%s'%(self.path, self.pet_names[index])).get_fdata()
        
        # Find the index of the current patients to be able to get the correct min/max/99th percentile values of the subject/
        try:
            patient_name_current = int(self.ct_names[index][:-14])
        except ValueError:
            patient_name_current = self.ct_names[index][:-14]
        
        index_patient = np.where(self.patient_names == patient_name_current)[0][0]
        ct_min_current, ct_99_percentile_current = self.ct_min[index_patient], self.ct_99_percentile[index_patient]
        pet_min_current, pet_99_percentile_current = np.min(self.pet_min), np.percentile(self.pet_max, 99)

        # Normalize
        ct = self.normalize_min_max(ct, ct_min_current, ct_99_percentile_current)
        pet = self.normalize_min_max(pet, pet_min_current, pet_99_percentile_current)

        ct = ct.astype(np.float32)
        pet = pet.astype(np.float32)

        sample = {'image': ct, 'target': pet}

        # Applt transformation
        if self.transform != None:
            sample = self.transform(sample)

        return sample['image'], sample['target']

    def normalize_min_max(self, img, min_val, max_val):
        img_normalized = (img - min_val) / (max_val - min_val)

        return img_normalized

    def __len__(self):
         return len(self.ct_names)