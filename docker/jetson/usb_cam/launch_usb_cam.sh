#!/bin/bash
CAM_TYPE='Arducam'
CAM_ADDRESS=$(v4l2-ctl --list-devices | grep -A 1 ${CAM_TYPE} | tail -n 1 | awk '{$1=$1};1')
ros2 run usb_cam usb_cam_node_exe --remap __ns:=/usb_cam --ros-args --params-file /cam_params.yaml -p video_device:="$CAM_ADDRESS" -p camera_name:=usb_cam