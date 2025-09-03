#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

static cv::Mat deskew(const cv::Mat& src) {
    cv::Mat gray;
    if (src.channels() > 1)
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else
        gray = src;
    cv::Mat bw;
    cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    std::vector<cv::Point> pts;
    cv::findNonZero(bw, pts);
    cv::RotatedRect box = cv::minAreaRect(pts);
    double angle = box.angle;
    if (angle < -45)
        angle += 90;
    cv::Mat rot = cv::getRotationMatrix2D(box.center, angle, 1);
    cv::Mat dst;
    cv::warpAffine(src, dst, rot, src.size(), cv::INTER_CUBIC);
    return dst;
}

static cv::Rect findPlate(const cv::Mat& src) {
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat filtered;
    cv::bilateralFilter(gray, filtered, 11, 17, 17);
    cv::Mat edged;
    cv::Canny(filtered, edged, 30, 200);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect best;
    double bestArea = 0;
    for (const auto& c : contours) {
        cv::RotatedRect rr = cv::minAreaRect(c);
        cv::Size2f sz = rr.size;
        double area = sz.width * sz.height;
        double ratio = sz.width > sz.height ? sz.width / sz.height : sz.height / sz.width;
        if (ratio > 2 && ratio < 6 && area > bestArea) {
            bestArea = area;
            best = rr.boundingRect() & cv::Rect(0, 0, src.cols, src.rows);
        }
    }
    return best;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <imagem>" << std::endl;
        return 1;
    }
    fs::path inputPath = argv[1];
    cv::Mat image = cv::imread(inputPath.string());
    if (image.empty()) {
        std::cerr << "Erro ao abrir imagem: " << inputPath << std::endl;
        return 1;
    }
    cv::Mat desk = deskew(image);
    cv::Rect plateRect = findPlate(desk);
    if (plateRect.area() == 0) {
        std::cerr << "Placa nao encontrada" << std::endl;
        return 1;
    }
    cv::Mat plate = desk(plateRect);

    fs::path outputPath = inputPath.stem().string() + "_placa" + inputPath.extension().string();
    if (!cv::imwrite(outputPath.string(), plate)) {
        std::cerr << "Erro ao salvar imagem: " << outputPath << std::endl;
        return 1;
    }
    std::cout << outputPath.string() << std::endl;
    return 0;
}
