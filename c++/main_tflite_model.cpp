//std
#include <memory>
#include <cstdio>
#include <iostream>
#include <tuple>
#include <filesystem>
#include <vector>
#include <chrono>

//tensorflow
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

//json
#include <jsoncpp/json/json.h>
#include <fstream>

//opencv
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "inc/efficientdet_classes.h"

using Efficientdet::ImageFile;
using Efficientdet::Params;

//read model and images path from params.json
Params readParams(const std::string &json){
    std::ifstream paramsJSON(json, std::ifstream::binary);
    Json::Value params;
    paramsJSON >> params;

    std::cout<<params["model_path"] << std::endl; //Prints the value for "model_path"
    std::cout<<params["images_path"] << std::endl; //Prints the value for "images_path"
    std::cout<<params["score_threshold"] << std::endl; //Prints the value for "score_threshold"
    std::cout<<params["image_save_path"] << std::endl; //Prints the value for "score_threshold"

    const std::string modelPath = params["model_path"].asString();
    const std::string imagesPath = params["images_path"].asString();
    const std::string imageSavePath = params["image_save_path"].asString();
    const double scoreThreshold = params["score_threshold"].asDouble();

    std::cout << "Loading model from : " << modelPath << std::endl;
    std::cout << "Loading images from : " << imagesPath << std::endl;
    std::cout << "Score threshold : " << scoreThreshold << std::endl;
    std::cout << "Output images location : " << scoreThreshold << std::endl;

    Params result{
        .modelPath = modelPath,
        .imagesPath = imagesPath,
        .imageSavePath = imageSavePath,
        .scoreThreshold = scoreThreshold
    };

    return result;
}

void displayImage(const cv::Mat &image){
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", image );
    cv::waitKey(0);
    cv::destroyAllWindows();
}

cv::Mat readImage(const std::string& path, bool showImage = false){
    cv::Mat I = cv::imread(path);
    if (I.empty())
    {
        std::cerr << "!!! Failed imread(): image not found" << std::endl;
        // don't let the execution continue, else imshow() will crash.
        return cv::Mat();
    }

    if(showImage){
        displayImage(I);
    }

    return I;
}

std::vector<ImageFile> readAllImagesFromDir(const std::string& imagesPath){
    std::vector<ImageFile> images;
    for (const auto &file : std::filesystem::directory_iterator(imagesPath)){
        if(file.path().extension() == ".jpg" ||
           file.path().extension() == ".png" ||
           file.path().extension() == ".gif" ||
           file.path().extension() == ".jpeg")
        std::cout << "Reading: " << file.path() << std::endl;
        // cv::Mat image = cv::imread(file.path());
        cv::Mat image = readImage(file.path(), false);

        ImageFile img {.name = file.path().stem(), .image = image};
        images.emplace_back(img);
    }
    return images;
}

void deleteAllImagesFromDir(const std::string& directoryName){
    for (const auto &file : std::filesystem::directory_iterator(directoryName)){
        if(file.path().extension() == ".jpg" ||
           file.path().extension() == ".png" ||
           file.path().extension() == ".gif" ||
           file.path().extension() == ".jpeg"){
            std::filesystem::remove(file);
        }
    }
    return;
}

cv::Mat resizeImage(const cv::Mat &img, size_t input_height, size_t input_width){
    int image_width = img.size().width; //
    int image_height = img.size().height; //

    int square_dim = std::min(image_width, image_height);
    int delta_height = (image_height - square_dim) / 2;   // so this is 0
    int delta_width = (image_width - square_dim) / 2;     // and this 140

    cv::Mat resizedImage;

    // crop and resize to 640x640
    cv::resize(img(cv::Rect(delta_width,
                            delta_height,
                            square_dim,
                            square_dim)),
                    resizedImage,
                    cv::Size(input_width, input_height));

    return resizedImage;
}

void printOutputDetails(const TfLiteTensor* boxes, const TfLiteTensor* classes, const TfLiteTensor* scores, const TfLiteTensor* count){

    std::cout << "Boxes type : " << boxes->type << std::endl;
    std::cout << "classes type : " << classes->type << std::endl;
    std::cout << "scores type : " << scores->type << std::endl;
    std::cout << "count type : " << count->type << std::endl;

    std::cout << "Boxes : " << boxes->dims->size << std::endl;
    for(int i = 0; i < boxes->dims->size; ++i){
        std::cout << "boxes->dim["  << i << "] = " << boxes->dims->data[i] << std::endl;
    }
    std::cout << "Boxes bytes: " << boxes->bytes << std::endl;
    auto boxesData = boxes->data.f;
    std::cout << "boxesData[0] : " << boxesData[200] << std::endl;

    std::cout << "classes : " << classes->dims->size << std::endl;
    for(int i = 0; i < classes->dims->size; ++i){
        std::cout << "classes->dim["  << i << "] = " << classes->dims->data[i] << std::endl;
    }
    std::cout << "classes bytes: " << classes->bytes << std::endl;

    std::cout << "scores : " << scores->dims->size << std::endl;
    for(int i = 0; i < scores->dims->size; ++i){
        std::cout << "scores->dim["  << i << "] = " << scores->dims->data[i] << std::endl;
    }
    std::cout << "scores bytes: " << scores->bytes << std::endl;

    std::cout << "count : " << count->dims->size << std::endl;
    for(int i = 0; i < count->dims->size; ++i){
        std::cout << "count->dim["  << i << "] = " << count->dims->data[i] << std::endl;
    }
    std::cout << "count bytes: " << count->bytes << std::endl;
}

void drawRectangleOverObjects(cv::Mat& image, const float* const boxesData, const float* const scoresData, const float* const countData, const float* const classesData, const double scoreThreshold){
    for(size_t i = 0; i < *countData; ++i){
        if(scoresData[i] > scoreThreshold){
            cv::Point p1 {static_cast<int>(boxesData[i * sizeof(float) + 1] * 640), static_cast<int>(boxesData[i * sizeof(float)] * 640)};
            cv::Point p2 {static_cast<int>(boxesData[i * sizeof(float) + 3] * 640), static_cast<int>(boxesData[i * sizeof(float) + 2] * 640)};
            cv::Scalar color{255, 255, 255};
            int thickness = 1;

            cv::rectangle(image, p1, p2, color, thickness);
            cv::putText(image, Efficientdet::classes.at(classesData[i]), p1, cv::FONT_HERSHEY_DUPLEX, 1, color, thickness);
        }
    }
}

void saveImage(const cv::Mat& resizedImage, std::string filename){
    cv::imwrite(filename, resizedImage);
}

int main(int argc, char const *argv[])
{
    /* code */
    // Load the model
    std::string paramsJson;
    if(argc > 0){
        paramsJson = std::string(argv[1]);
        std::filesystem::path filepath = paramsJson;
        if(std::filesystem::exists(filepath))
            std::cout << "Loading params from: " << paramsJson << std::endl;
        else{
            std::cerr << "Given file doesn't exist!" << std::endl;
        }
    }
    else{
        return 0;
    }
    //std::string paramsJson = "../params.json";
    const Params params = readParams(paramsJson);

    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(params.modelPath.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load TFLite model");
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // Prepare GPU delegate.
    auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
    if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk)
        return false;

    // Resize input tensors, if desired.
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }

    // delete images from saved directory
    deleteAllImagesFromDir(params.imageSavePath);

    // tflite::PrintInterpreterState(interpreter.get());
    std::cout << "Loading images..." << std::endl;
    std::vector<ImageFile> images = readAllImagesFromDir(params.imagesPath);

    auto inputData = interpreter->inputs()[0];
    auto input_batch_size = interpreter->tensor(inputData)->dims->data[0];
    auto input_height = interpreter->tensor(inputData)->dims->data[1];
    auto input_width = interpreter->tensor(inputData)->dims->data[2];
    auto input_channels = interpreter->tensor(inputData)->dims->data[3];

    std::cout << "The input tensor has the following dimensions: ["
            << input_batch_size << ","
            << input_height << ","
            << input_width << ","
            << input_channels << "]" << std::endl;

    for(auto image : images){
        // this is the input for the model
        cv::Mat resizedImage = resizeImage(image.image, input_height, input_width);
        // displayImage(resizedImage);
        // Fill `input`.
        memcpy(interpreter->typed_input_tensor<unsigned char>(0),
                resizedImage.data,
                resizedImage.total() * resizedImage.elemSize());


        std::chrono::time_point start = std::chrono::system_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference failed" << std::endl;
            return -1;
        }
        //model inference time
        std::chrono::time_point end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);;
        std::cout << "Inference time[ms] : " << elapsed.count() << std::endl;

        const TfLiteTensor* boxes = interpreter->tensor(interpreter->outputs()[0]);
        const TfLiteTensor* classes = interpreter->tensor(interpreter->outputs()[1]);
        const TfLiteTensor* scores = interpreter->tensor(interpreter->outputs()[2]);
        const TfLiteTensor* count = interpreter->tensor(interpreter->outputs()[3]);

        // printOutputDetails(boxes, classes, scores, count);

        const auto boxesData = boxes->data.f;
        const auto classesData = classes->data.f;
        const auto scoresData = scores->data.f;
        const auto countData = count->data.f;

        drawRectangleOverObjects(resizedImage, boxesData, scoresData, countData, classesData, params.scoreThreshold);
        // displayImage(resizedImage);

        // save resized image
        std::string savedFileName = params.imageSavePath + image.name + "_processed.jpg";
        saveImage(resizedImage, savedFileName);
    }

    delete delegate;

    return 0;
}