#include <iostream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>

#pragma comment(lib, "Ws2_32.lib")

const std::string CAMERA_IP = "IP_ADDRESS";
const int CAMERA_PORT = xxxxx;

void printHex(const char* data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (unsigned int)(unsigned char)data[i];
    }
    std::cout << std::endl;
}

std::string bytesToHex(const char* bytes, size_t length) {
    std::ostringstream oss;
    for (size_t i = 0; i < length; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (unsigned int)(unsigned char)bytes[i];
    }
    return oss.str();
}

std::string sendCommandTCP(const std::string& command) {
    WSADATA wsaData;
    SOCKET ConnectSocket = INVALID_SOCKET;
    struct sockaddr_in clientService;
    std::string response;

    // winsock initialise
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return response;
    }

    // socket connection
    ConnectSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (ConnectSocket == INVALID_SOCKET) {
        std::cerr << "Error at socket(): " << WSAGetLastError() << std::endl;
        WSACleanup();
        return response;
    }

    // TCP connection
    clientService.sin_family = AF_INET;
    clientService.sin_port = htons(CAMERA_PORT);
    if (inet_pton(AF_INET, CAMERA_IP.c_str(), &(clientService.sin_addr)) <= 0) {
        std::cerr << "Invalid IP address" << std::endl;
        closesocket(ConnectSocket);
        WSACleanup();
        return response;
    }

    //  server coonection
    if (connect(ConnectSocket, (SOCKADDR*)&clientService, sizeof(clientService)) == SOCKET_ERROR) {
        std::cerr << "Failed to connect: " << WSAGetLastError() << std::endl;
        closesocket(ConnectSocket);
        WSACleanup();
        return response;
    }

    // transmit data
    int bytesSent = send(ConnectSocket, command.c_str(), command.size(), 0);
    if (bytesSent == SOCKET_ERROR) {
        std::cerr << "Failed to send command: " << WSAGetLastError() << std::endl;
        closesocket(ConnectSocket);
        WSACleanup();
        return response;
    }

    // receive data
    const int MAX_RECVBUF_SIZE = 4096;
    char recvbuf[MAX_RECVBUF_SIZE];
    int recvbuflen = MAX_RECVBUF_SIZE;

    int bytesReceived = recv(ConnectSocket, recvbuf, recvbuflen, 0);
    if (bytesReceived > 0) {
        response.assign(recvbuf, bytesReceived);
        std::cout << "Received response: ";
        printHex(response.c_str(), response.size());
    }
    else if (bytesReceived == 0) {
        std::cerr << "Connection closed" << std::endl;
    }
    else {
        std::cerr << "recv failed: " << WSAGetLastError() << std::endl;
    }

    // socket cleaner
    closesocket(ConnectSocket);
    WSACleanup();

    return response;
}

std::string parseFirmwareVersion(const std::string& response) {
    if (response.size() < 12) return "Invalid response";
    return bytesToHex(response.c_str() + 8, 4);
}

std::string parseCameraStatus(const std::string& response) {
    if (response.size() < 12) return "Invalid response";
    return bytesToHex(response.c_str() + 8, 4);
}

void requestFirmwareVersion() {
    std::string command = "\x55\x66\x01\x00\x00\x00\x00\x01\x64\xc4";
    std::string response = sendCommandTCP(command);
    std::string firmwareVersion = parseFirmwareVersion(response);
    std::cout << "Firmware Version: " << firmwareVersion << std::endl;
}

void autoZoomIn() {
    std::string command = "\x55\x66\x01\x01\x00\x00\x00\x05\x01\x8d\x64";
    std::string response = sendCommandTCP(command);
    std::string status = parseCameraStatus(response);
    std::cout << "Camera is Zooming in: " << status << std::endl;
}

void autoZoomOut() {
    std::string command = "\x55\x66\x01\x01\x00\x00\x00\x05\xFF\x5c\x6a";
    std::string response = sendCommandTCP(command);
    std::string status = parseCameraStatus(response);
    std::cout << "Camera is Zooming out: " << status << std::endl;
}

void center() {
    std::string command = "\x55\x66\x01\x01\x00\x00\x00\x08\x01\xd1\x12";
    std::string response = sendCommandTCP(command);
    std::string status = parseCameraStatus(response);
    std::cout << "Centered: " << status << std::endl;
}

void cameraStatus() {
    std::string command = "\x55\x66\x01\x02\x00\x00\x00\x07\x64\x64\x3d\xcf"; // Note that this is for Yaw and Pitch at full length
    std::string response = sendCommandTCP(command);
    std::string status = parseCameraStatus(response);
    std::cout << "Camera status: " << status << std::endl;
}

void lockMode() {
    std::string command = "\x55\x66\x01\x01\x00\x00\x00\x0c\x03\x57\xfe";
    std::string response = sendCommandTCP(command);
    std::string status = parseCameraStatus(response);
    std::cout << "Lock mode activated: " << status << std::endl;
}

void followMode() {
    std::string command = "\x55\x66\x01\x01\x00\x00\x00\x0c\x04\xb0\x8e";
    std::string response = sendCommandTCP(command);
    std::string status = parseCameraStatus(response);
    std::cout << "Follow mode activated: " << status << std::endl;
}

void gimbalRotation() {
    std::string command = "\x55\x66\x01\x04\x00\x00\x00\x0e\x00\x00\xff\xa6\x3b\x11";
    std::string response = sendCommandTCP(command);
    std::string status = parseCameraStatus(response);
    std::cout << "Gimbal rotation response: " << status << std::endl;
}

int main() {
    std::cout << "Press 'f' to request firmware version." << std::endl;
    std::cout << "Press 'z' to auto zoom in." << std::endl;
    std::cout << "Press 'x' to auto zoom out." << std::endl;
    std::cout << "Press 'c' to center the gimbal." << std::endl;
    std::cout << "Press 's' to get camera status." << std::endl;
    std::cout << "Press 'l' to activate lock mode." << std::endl;
    std::cout << "Press 'o' to activate follow mode." << std::endl;
    std::cout << "Press 'r' to rotate the gimbal." << std::endl;
    std::cout << "Press 'q' to quit the program." << std::endl;

    while (true) {
        if (GetAsyncKeyState('F') & 0x8000) {
            requestFirmwareVersion();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('Z') & 0x8000) {
            autoZoomIn();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('X') & 0x8000) {
            autoZoomOut();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('C') & 0x8000) {
            center();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('S') & 0x8000) {
            cameraStatus();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('L') & 0x8000) {
            lockMode();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('O') & 0x8000) {
            followMode();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('R') & 0x8000) {
            gimbalRotation();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (GetAsyncKeyState('Q') & 0x8000) {
            std::cout << "Quitting program." << std::endl;
            break;
        }
    }

    return 0;
}
