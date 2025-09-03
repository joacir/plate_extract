#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
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

    // Extrair a região da placa via perspectiva para alinhar corretamente
    std::vector<cv::Point2f> pts;
    for (auto& p : bestContour) pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    std::vector<cv::Point2f> rect(4);
    auto sumCmp = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x + a.y < b.x + b.y; };
    auto diffCmp = [](const cv::Point2f& a, const cv::Point2f& b) { return a.x - a.y < b.x - b.y; };
    rect[0] = *std::min_element(pts.begin(), pts.end(), sumCmp);
    rect[2] = *std::max_element(pts.begin(), pts.end(), sumCmp);
    rect[1] = *std::min_element(pts.begin(), pts.end(), diffCmp);
    rect[3] = *std::max_element(pts.begin(), pts.end(), diffCmp);
    float widthA = std::hypot(rect[2].x - rect[3].x, rect[2].y - rect[3].y);
    float widthB = std::hypot(rect[1].x - rect[0].x, rect[1].y - rect[0].y);
    float maxWidth = std::max(widthA, widthB);
    float heightA = std::hypot(rect[1].x - rect[2].x, rect[1].y - rect[2].y);
    float heightB = std::hypot(rect[0].x - rect[3].x, rect[0].y - rect[3].y);
    float maxHeight = std::max(heightA, heightB);
    std::vector<cv::Point2f> dst = {{0, 0}, {maxWidth - 1, 0}, {maxWidth - 1, maxHeight - 1}, {0, maxHeight - 1}};
    cv::Mat plate;
    if (maxWidth < 1.f || maxHeight < 1.f) {
        // fallback para boundingRect caso perspectiva gere dimensões inválidas
        plate = image(cv::boundingRect(bestContour)).clone();
    } else {
        cv::Mat M = cv::getPerspectiveTransform(rect, dst);
        cv::warpPerspective(image, plate, M, cv::Size((int)maxWidth, (int)maxHeight));
    }

    // Melhoria da qualidade para OCR seguindo recomendações do Tesseract
    cv::Mat plateGray;
    cv::cvtColor(plate, plateGray, cv::COLOR_BGR2GRAY);
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
