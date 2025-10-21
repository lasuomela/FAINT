from typing import Optional, Union

import h5py
import numpy as np
from pathlib import Path
import cv2
import time

import albumentations as A
import torchvision.transforms as T

from rclpy.logging import get_logger

class SubgoalDBHandler:

    rgb_suffixes = ['.png', '.jpg', '.jpeg']
    depth_suffixes = ['.tiff']

    def __init__(
        self,
        img_dir_path: Path,
        img_type: str,
        db_path: Optional[Path] = None,
        image_transform: Optional[Union[A.Compose, T.Compose]] = None
        ):

        """
        Load the gallery db

        Expects that the topological map consists of
        a non-cyclical non-branching graph of images,
        where each image's filename is an integer
        that specifies the order in which the images were taken.

        Args:
            img_dir_path: Path to the directory containing the images
            img_type: The type of the images, 'rgb', or 'depth'
            db_path: Path to the gallery db, if None, the db will not be loaded
            image_transform: Transform to apply to the images
        """

        if img_type == 'rgb':
            img_suffixes = self.rgb_suffixes
        elif img_type == 'depth':
            img_suffixes = self.depth_suffixes
        else:
            raise ValueError(f"Invalid image type {img_type}, must be 'rgb' or 'depth'")
            
        # Get the image paths
        img_dir_path = Path(img_dir_path)
        img_paths = [p.resolve() for p in img_dir_path.glob("**/*") if p.suffix in img_suffixes]

        
        # Get the image filenames without the extension and check that they are integers
        try:
            # Assume that anything trailing the integer part is separated by an underscore
            img_idxs = [int(img_path.stem.split('_')[0]) for img_path in img_paths]
        except ValueError:
            raise ValueError("Image filenames must be integers, got {}".format(img_paths))

        # Sort the image paths and descriptors according to the image filename
        self.img_paths = np.array([x for _, x in sorted(zip(img_idxs, img_paths), key=lambda pair: pair[0])])



        # Load the db file containing the extracted global descriptors
        self.descriptors = None
        if db_path is not None:
            descriptors = self.load_descriptors(db_path)

            # Sort the descriptors according to the db ordering
            img_names = self.gallery_db.keys()
            img_idxs = [int(Path(img_name).stem.split('_')[0]) for img_name in img_names]
            self.descriptors = np.array([x for _, x in sorted(zip(img_idxs, descriptors), key=lambda pair: pair[0])])

        # Read the images into memory
        self.images = []
        for img_path in self.img_paths:
            if img_type == 'rgb':
                img = self.read_rgb_image(img_path)
            elif img_type == 'depth':
                img = self.read_depth_image(img_path)
            else:
                raise ValueError(f"Invalid image type {img_type}, must be 'rgb' or 'depth'")

            # Preprocess the image
            if image_transform is not None:
                if isinstance(image_transform, A.Compose):
                    img = image_transform(image=img)['image'].unsqueeze(0)
                elif isinstance(image_transform, T.Compose):
                    img = image_transform(img).unsqueeze(0)
                else:
                    raise ValueError("Invalid image transform, must be either albumentations.Compose or torchvision.transforms.Compose")
            self.images.append(img)            
            
        self.gallery_len = len(self.images)

    def load_descriptors(self, db_path):
        # Check if the db file exists and wait until it does
        first_iter = True
        while not db_path.exists():
            if first_iter:
                first_iter = False
                get_logger('db_handler').info(f"Waiting for {db_path} to be created, sleeping...")
            time.sleep(1)

        # Try to open the db file
        first_iter = True
        while True:
            try:
                self.gallery_db = h5py.File(db_path, 'r')
            except OSError:
                # Another process is probably writing to the file,
                # wait until it is done
                if first_iter:
                    first_iter = False
                    get_logger('db_handler').info(f"The file {db_path} is being written to, sleeping...")
                time.sleep(1)
            else:
                break

        # Load the extracted global descriptors from the gallery db
        img_names = self.gallery_db.keys()
        descriptors = []
        for img_name in img_names:
            img_attrs = self.gallery_db[img_name]
            img_descriptor = img_attrs['global_descriptor']
            descriptors.append(img_descriptor)
        return descriptors

    def read_rgb_image(self, img_path):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def read_depth_image(self, img_path):
        depth_img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        return depth_img

    def get_map_size(self):
        return self.gallery_len

    def get_descriptors(self):
        return self.descriptors

    def get_filepaths(self):
        return self.img_paths
    
    def get_images(self):
        return self.images

    def get_by_filename(self, filename):
        idx = np.argwhere(self.img_paths == filename).flatten()[0]
        return self.descriptors[idx]

    def get_by_idx(self, idx):
        return self.img_paths[idx], self.descriptors[idx]
    
    def get_image_by_idx(self, idx):
        return self.images[idx]