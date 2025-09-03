#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <caminho_imagem>" << std::endl;
        return 1;
    }
    std::string inputPath = argv[1];
    cv::Mat image = cv::imread(inputPath);
    if (image.empty()) {
        std::cerr << "Erro ao carregar imagem: " << inputPath << std::endl;
        return 1;
    }

    // Pré-processamento para detecção de bordas
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat denoised;
    cv::bilateralFilter(gray, denoised, 11, 17, 17);
    cv::Mat edged;
    cv::Canny(denoised, edged, 30, 200);

    // Encontrar contornos
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> bestContour;
    double maxArea = 0;
    for (auto& contour : contours) {
        double peri = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.018 * peri, true);
        if (approx.size() == 4) {
            double area = cv::contourArea(approx);
            if (area > maxArea) {
                maxArea = area;
                bestContour = approx;
            }
        }
    }
    if (bestContour.empty()) {
        std::cerr << "Placa não encontrada." << std::endl;
        return 1;
    }

    // Extrair a região da placa
    cv::Rect plateRect = cv::boundingRect(bestContour);
    cv::Mat plate = image(plateRect).clone();

    // Melhoria da qualidade para OCR seguindo recomendações do Tesseract
    cv::Mat plateGray;
    cv::cvtColor(plate, plateGray, cv::COLOR_BGR2GRAY);
    // Correção de inclinação (deskew) conforme recomendações de alinhamento
    {
        cv::Moments m = cv::moments(plateGray);
        if (std::abs(m.mu02) > 1e-2) {
            double skew = m.mu11 / m.mu02;
            cv::Mat warpMat = (cv::Mat_<double>(2, 3) << 1, skew, -0.5 * plateGray.rows * skew,
                                0, 1, 0);
            cv::warpAffine(plateGray, plateGray, warpMat, plateGray.size(),
                           cv::WARP_INVERSE_MAP | cv::INTER_LINEAR);
        }
    }
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    cv::Mat equalized;
    clahe->apply(plateGray, equalized);
    cv::Mat thresh;
    cv::threshold(equalized, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat processed;
    // Remoção de ruídos e fechamento de caracteres
    cv::Mat opened, closed;
    cv::morphologyEx(thresh, opened, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(opened, closed, cv::MORPH_CLOSE, kernel);
    // Redimensionar para melhorar legibilidade de caracteres conforme recomendações do Tesseract
    cv::resize(closed, processed, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);

    // Gerar nome de saída: nome base + _placa + extensão
    auto dotPos = inputPath.find_last_of('.');
    std::string base = (dotPos == std::string::npos) ? inputPath : inputPath.substr(0, dotPos);
    std::string ext = (dotPos == std::string::npos) ? "" : inputPath.substr(dotPos);
    std::string outputPath = base + "_placa" + ext;
    if (!cv::imwrite(outputPath, processed)) {
        std::cerr << "Erro ao salvar imagem: " << outputPath << std::endl;
        return 1;
    }
    std::cout << "Imagem da placa salva em: " << outputPath << std::endl;
    return 0;
}
