import json
import os

import os
import sys

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(get_app_dir(), "mount_config.json")


MOUNT_CONFIG_CACHE = None

def load_mount_config(filename=CONFIG_PATH):
    global MOUNT_CONFIG_CACHE

    # return cached version if already loaded
    if MOUNT_CONFIG_CACHE is not None:
        return MOUNT_CONFIG_CACHE

    default_data = {
        "idms_tgt_dist_to_consider": 5000,
        "mount_latitude": 30.1645096,
        "mount_longitude": 74.1767266,
        "default_mount_roll": 30,
        "default_mount_yaw": 0,
        "motor_speed_min":350,
        "ESP32_PORT":"/dev/tty.usbserial-0001",
        "Gyro_PORT":"/dev/cu.usbmodem0x80000001"
    }

    # create file if missing
    if not os.path.exists(filename):
        with open(CONFIG_PATH, "w") as f:
            json.dump(default_data, f, indent=4)
        MOUNT_CONFIG_CACHE = default_data
        return MOUNT_CONFIG_CACHE

    # read existing file
    with open(filename, "r") as f:
        mount_config = json.load(f)

    # ensure all keys exist
    for key, value in default_data.items():
        if key not in mount_config:
            mount_config[key] = value

    MOUNT_CONFIG_CACHE = mount_config
    return mount_config