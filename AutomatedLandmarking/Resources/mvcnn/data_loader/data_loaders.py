import os
from base.base_data_loader import BaseDataLoader
from data_loader.FaceDataset import FaceDataset


class FaceDataLoader(BaseDataLoader):
    """
    Face data loader
    """
    def __init__(self, data_dir, heatmap_size=256, image_size=256, image_channels="RGB", n_views=96, batch_size=8,
                 shuffle=True, validation_split=0.0, num_workers=1, training=True, train_ds='dataset_train.txt',
                 val_ds=None, n_lm=20):
        self.data_dir = data_dir
        self.train_file_name = os.path.join(self.data_dir, train_ds)
        print('Checking train split')
        self.dataset = FaceDataset(csv_file=self.train_file_name,
                                    root_dir=data_dir, n_lm=n_lm,
                                    heatmap_size=heatmap_size, image_size=image_size,
                                    image_channels=image_channels, n_views=n_views)

        self.validation_dataset = None
        if val_ds is not None:
            print('Checking validation split')
            self.val_file_name = os.path.join(self.data_dir, val_ds)
            self.validation_dataset = FaceDataset(csv_file=self.val_file_name,
                                                    root_dir=data_dir, n_lm=n_lm,
                                                    heatmap_size=heatmap_size, image_size=image_size,
                                                    image_channels=image_channels, n_views=n_views)
            validation_split = len(self.validation_dataset) / (len(self.dataset) + len(self.validation_dataset))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, val_ds=self.validation_dataset)
