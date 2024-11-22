#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


void detectObjects(cv::Mat& image, const cv::dnn::Net& net) {
    
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    
    for (const auto& output : outputs) {
        for (int i = 0; i < output.rows; ++i) {
            const auto* data = output.ptr<float>(i);
            float confidence = data[4];
            if (confidence > 0.5) { // Confidence threshold
                int centerX = static_cast<int>(data[0] * image.cols);
                int centerY = static_cast<int>(data[1] * image.rows);
                int width = static_cast<int>(data[2] * image.cols);
                int height = static_cast<int>(data[3] * image.rows);

                cv::Rect bbox(centerX - width / 2, centerY - height / 2, width, height);
                cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
}

int main() {
    
    std::string cameraUrl = "http://<IP_ADDRESS>:<xxxxx>/main.264"; 
    std::string modelPath = "path/to/your/yolov8/model";      

    
    cv::dnn::Net net = cv::dnn::readNet(modelPath);

    
    cv::VideoCapture cap(cameraUrl);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video stream: " << cameraUrl << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "End of stream or empty frame" << std::endl;
            break;
        }

        
        detectObjects(frame, net);

        
        cv::imshow("Detection", frame);
        if (cv::waitKey(1) == 27) { // Exit on ESC key
            break;
        }
    }

    return 0;
}
