import socket
import binascii
import time
import keyboard

CAMERA_IP = 'IP_ADDRESS'
CAMERA_PORT = xxxxx

def send_command_tcp(command): # tcp initialising
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.connect((CAMERA_IP, CAMERA_PORT))
            print(f"Connected to {CAMERA_IP}:{CAMERA_PORT}")

            sock.send(command)
            print(f"Command sent: {command.hex()}")
            time.sleep(1)

            response = sock.recv(4096)
            if not response:
                print("No response received.")
                return None

            print(f"Response received: {response.hex()}")
            return response

    except socket.timeout:
        print("Socket timed out while waiting for a response.")
    except ConnectionRefusedError:
        print("Connection refused by the camera. Check the IP and port.")
    except socket.error as e:
        print(f"Socket error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while sending command: {e}")
    return None

def parse_firmware_version(response):
    try:
        response_hex = binascii.hexlify(response).decode('ascii')
        print(f"Raw response hex: {response_hex}")

        firmware_version_bytes = response[8:12]  #example only, standard
        firmware_version = ''.join(format(b, '02x') for b in firmware_version_bytes)

        try:
            firmware_version_str = bytes.fromhex(firmware_version).decode('ascii')
        except (ValueError, UnicodeDecodeError):
            firmware_version_str = firmware_version

        return firmware_version_str

    except IndexError as e:
        print(f"Index error while parsing response: {e}")
    except ValueError as e:
        print(f"Value error while converting data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while parsing firmware version: {e}")
    return None

def parse_camera_status(response):
    try:
        response_hex = binascii.hexlify(response).decode('ascii')
        print(f"Raw response hex: {response_hex}")

        status_bytes = response[8:12]  #example only, standard
        status = ''.join(format(b, '02x') for b in status_bytes)

        try:
            status_str = bytes.fromhex(status).decode('ascii')
        except (ValueError, UnicodeDecodeError):
            status_str = status

        return status_str

    except IndexError as e:
        print(f"Index error while parsing response: {e}")
    except ValueError as e:
        print(f"Value error while converting data: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while parsing camera status: {e}")
    return None

def request_firmware_version():
    command = bytes.fromhex("55 66 01 00 00 00 00 01 64 c4")
    response = send_command_tcp(command)
    if response:
        firmware_version = parse_firmware_version(response)
        if firmware_version:
            print(f"Firmware Version: {firmware_version}")
        else:
            print("Failed to parse firmware version.")
    else:
        print("Failed to receive a response from the camera.")

def auto_zoom_in():
    command = bytes.fromhex("55 66 01 01 00 00 00 05 01 8d 64")
    response = send_command_tcp(command)
    if response:
        status = parse_camera_status(response)
        if status:
            print(f"Camera is Zooming in: {status}")
        else:
            print("Failed to parse camera status.")
    else:
        print("Failed to receive a response from the camera.")

def auto_zoom_out():
    command = bytes.fromhex("55 66 01 01 00 00 00 05 FF 5c 6a")
    response = send_command_tcp(command)
    if response:
        status = parse_camera_status(response)
        if status:
            print(f"Camera is Zooming out: {status}")
        else:
            print("Failed to parse camera status.")
    else:
        print("Failed to receive a response from the camera.")

def Center():
    command = bytes.fromhex("55 66 01 01 00 00 00 08 01 d1 12")
    response = send_command_tcp(command)
    if response:
        status = parse_camera_status(response)
        if status:
            print(f"Centered: {status}")
        else:
            print("Failed to parse camera status.")
    else:
        print("Failed to receive a response from the camera.")

def Camera_Status(): #this is not for camera status, update needed
    command = bytes.fromhex("55 66 01 02 00 00 00 07 64 64 3d cf") #this command is a test command for yaw and pitch for full length
    response = send_command_tcp(command)
    if response:
        status = parse_camera_status(response)
        if status:
            print(f"Camera status: {status}")
        else:
            print("Failed to parse camera status.")
    else:
        print("Failed to receive a response from the camera.")

def Lock_mode():
    command = bytes.fromhex("55 66 01 01 00 00 00 0c 03 57 fe")
    response = send_command_tcp(command)
    if response:
        status = parse_camera_status(response)
        if status:
            print(f"Lock mode activated: {status}")
        else:
            print("Failed to parse camera status.")
    else:
        print("Failed to receive a response from the camera.")

def Follow_mode():
    command = bytes.fromhex("55 66 01 01 00 00 00 0c 04 b0 8e")
    response = send_command_tcp(command)
    if response:
        status = parse_camera_status(response)
        if status:
            print(f"Follow mode activated: {status}")
        else:
            print("Failed to parse camera status.")
    else:
        print("Failed to receive a response from the camera.")

def Gimbal_Rotation():
    command = bytes.fromhex("55 66 01 04 00 00 00 0e 00 00 ff a6 3b 11")
    response = send_command_tcp(command)
    if response:
        status = parse_camera_status(response)
        if status:
            print(f"Gimbal rotation response: {status}")
        else:
            print("Failed to parse gimbal rotation status.")
    else:
        print("Failed to receive a response from the camera.")

def save_response_to_file(response, filename="camera_response.log"):
    with open(filename, 'a') as file:
        file.write(f"{time.ctime()}: {response.hex()}\n")
    print(f"Response saved to {filename}")

if __name__ == "__main__":
    print("Press 'f' to request firmware version.")
    print("Press 'z' to auto zoom in.")
    print("Press 'x' to auto zoom out.")
    print("Press 'c' to center the gimbal.")
    print("Press 's' to get camera status.")
    print("Press 'l' to activate lock mode.")
    print("Press 'o' to activate follow mode.")
    print("Press 'r' to rotate the gimbal.")

    while True:
        if keyboard.is_pressed('f'):
            request_firmware_version()
            time.sleep(1)
        elif keyboard.is_pressed('z'):
            auto_zoom_in()
        elif keyboard.is_pressed('x'):
            auto_zoom_out()
        elif keyboard.is_pressed('c'):
            Center()
        elif keyboard.is_pressed('s'):
            Camera_Status()
        elif keyboard.is_pressed('l'):
            Lock_mode()
        elif keyboard.is_pressed('o'):
            Follow_mode()
        elif keyboard.is_pressed('r'): 
            Gimbal_Rotation()