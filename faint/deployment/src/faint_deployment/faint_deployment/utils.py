
from typing import List, Optional

import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ROS
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from faint.common.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)

def read_image(path):
    mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if len(image.shape) == 3:
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_image_transform(
    image_size: List[int],
    normalization_type: Optional[str] = "IMAGENET_DEFAULT",
) -> transforms.Compose:

    if normalization_type == "IMAGENET_DEFAULT":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif normalization_type == "IMAGENET_STANDARD":
        mean = IMAGENET_STANDARD_MEAN
        std = IMAGENET_STANDARD_STD
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    image_size = image_size[::-1] # torchvision's transforms.Resize expects [height, width]
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

def get_depth_image_transform(image_size: List[int]) -> transforms.Compose:
    transform = A.Compose([
        A.Resize(image_size[1], image_size[0]), # Albumentations uses (height, width) format
        A.ToFloat(max_value=2**16-1),
        ToTensorV2()
    ])
    return transform

def create_waypoint_viz_marker(waypoints, timestamp):
    # Create a line strip marker
    marker = Marker()
    marker.header.frame_id = 'base_link'
    marker.header.stamp = timestamp
    marker.ns = "waypoints"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.color.r = 1.0
    marker.color.a = 1.0
    marker.points = []
    for waypoint in waypoints:
        marker.points.append(Point(x=waypoint[0].item()*5, y=waypoint[1].item()*5, z=-0.4))
    return marker