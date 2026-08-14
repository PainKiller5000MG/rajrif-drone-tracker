#BOARD:FURYF4OSD/STM32F405
#GYRO: ICM42688P

# Yaw-X Axis
# Roll-Y Axis

import serial
import struct
import time

from load_mount_data import load_mount_config

# data will be refreshed from the file mount_config.json
mount_config = load_mount_config()

# ---------------- CONFIG ----------------
PORT = "/dev/cu.usbmodem0x80000001"

if mount_config["Gyro_PORT"] is not None:
    PORT = mount_config["Gyro_PORT"]

BAUD = 115200

MSP_ATTITUDE = 108


def build_msp_request(cmd):
    header = b"$M<"
    size = 0
    checksum = size ^ cmd
    return header + bytes([size, cmd, checksum])


def read_packet(ser):

    # wait for start byte
    while ser.read() != b'$':
        pass

    if ser.read() == b'M':
        direction = ser.read()      # should be '>'
        size = ser.read()[0]
        cmd = ser.read()[0]

        data = ser.read(size)
        checksum = ser.read()

        return cmd, data

    return None, None


def parse_attitude(data):

    # unpack 3 int16 values
    roll, pitch, yaw = struct.unpack("<hhh", data)

    # FC sends roll/pitch multiplied by 10
    roll = roll / 10.0
    pitch = pitch / 10.0

    return roll, pitch, yaw


def normalize_yaw(angle):

    # convert to 0-360 first
    angle = angle % 360

    # shift to -180 to +180
    if angle > 180:
        angle -= 360

    return angle


# Open the gyro's serial connection - but never die over it. This used to
# run at import with no guard, so a missing FC (or the Mac-era port name on
# a Windows PC) killed the whole app before its window even opened. The
# callers already handle (None, None), so no gyro simply means no MOUNT
# angle and no IDMS slew, same as a gyro that stops answering.
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Connected to FC")
except Exception as _exc:
    ser = None
    print("Gyro FC not connected (%s) - MOUNT angle and IDMS slew disabled."
          % _exc)


def get_roll_yaw():
    if ser is None:
        return None, None

    # send MSP request
    request = build_msp_request(MSP_ATTITUDE)
    ser.write(request)

    cmd, data = read_packet(ser)

    if cmd == MSP_ATTITUDE and data is not None:

        roll, pitch, yaw = parse_attitude(data)

        # normalize yaw to -180 to +180
        yaw = normalize_yaw(yaw)
        return roll, yaw

    else:
        return None, None

'''
while True:

    roll, yaw = get_roll_yaw()

    if roll is not None:
        print(f"Roll:{roll:.1f}  Yaw:{yaw:.1f}")

    time.sleep(0.02)
'''