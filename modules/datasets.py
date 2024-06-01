import numpy as np
import tifffile as tiff
import torch
from torchvision import transforms

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, train_mode=True, transform=None, AFI_MODE=True, REMOVE_8= False):
        self.dataframe = dataframe
        self.train_mode = train_mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.AFI_MODE = AFI_MODE
        self.REMOVE_8 = REMOVE_8

        self.MAX_PIXEL_VALUE = 65535 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        image = tiff.imread(image_path).astype('float32')

        if self.AFI_MODE:
            # AFI 채널 계산
            AFI_channel = image[:, :, 6] / np.maximum(image[:, :, 1], 1)  # 0으로 나누는 것을 방지
            AFI_channel = np.expand_dims(AFI_channel, axis=2)

            # 원하는 채널 선택 및 AFI 채널 추가
            selected_channels = image[:, :, [0,1,2,3,4,5,6,8,9]]  # Blue, SWIR1, SWIR2 선택
            image_with_AFI = np.concatenate((selected_channels, AFI_channel), axis=2)  # AFI 채널 추가

            # 이미지 정규화 및 텐서 변환
            image = torch.from_numpy(image_with_AFI).permute(2, 0, 1) / self.MAX_PIXEL_VALUE

        if self.train_mode or len(self.dataframe.columns) == 2:
            mask_path = self.dataframe.iloc[idx, 1]
            mask = tiff.imread(mask_path).astype('float32')
            mask = torch.from_numpy(mask).unsqueeze(0)

            return image, mask
        else:
            return image

