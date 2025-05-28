# rs_utils.py
import pyrealsense2 as rs

def get_connected_devices_serial_numbers():
    """Returns a list of serial numbers of connected RealSense devices."""
    ctx = rs.context()
    devices = ctx.query_devices()
    serial_numbers = []
    for dev in devices:
        if dev.supports(rs.camera_info.serial_number):
            serial_numbers.append(dev.get_info(rs.camera_info.serial_number))
    return serial_numbers

if __name__ == "__main__":
    serials = get_connected_devices_serial_numbers()
    if serials:
        print("Connected RealSense devices by serial number:")
        for s in serials:
            print(f"  - {s}")
    else:
        print("No RealSense devices detected.")