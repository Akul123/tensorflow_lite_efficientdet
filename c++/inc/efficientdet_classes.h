#include <array>
#include <iostream>

#define NUMBER_OF_CLASSES 91

namespace Efficientdet{
    struct Params {
        std::string modelPath;
        std::string imagesPath;
        std::string imageSavePath;
        double scoreThreshold;
    };

    struct ImageFile{
        std::string name;
        cv::Mat image;
    };

    std::array<std::string, NUMBER_OF_CLASSES> classes{
                        "person",
                        "bicycle",
                        "car",
                        "motorcycle",
                        "airplane",
                        "bus",
                        "train",
                        "truck",
                        "boat",
                        "traffic light",
                        "fire hydrant",
                        "street sign",
                        "stop sign",
                        "parking meter",
                        "bench",
                        "bird",
                        "cat",
                        "dog",
                        "horse",
                        "sheep",
                        "cow",
                        "elephant",
                        "bear",
                        "zebra",
                        "giraffe",
                        "hat"
                        "backpack",
                        "umbrella",
                        "shoe",
                        "eye glasses",
                        "handbag",
                        "tie",
                        "suitcase",
                        "frisbe",
                        "skis",
                        "snowboard",
                        "sports ball",
                        "kite",
                        "baseball bat",
                        "baseball glove",
                        "skateboard",
                        "surfboard",
                        "tennis racket",
                        "bottle",
                        "plate",
                        "wine glass",
                        "cup",
                        "fork",
                        "knife",
                        "spoon",
                        "bowl",
                        "banana",
                        "apple",
                        "sandwich",
                        "orange",
                        "broccoli",
                        "carrot",
                        "hot dog",
                        "pizza",
                        "donut",
                        "cake",
                        "chair",
                        "couch",
                        "potted plant",
                        "bed",
                        "mirror",
                        "dining table",
                        "window",
                        "desk",
                        "toilet",
                        "door",
                        "tv",
                        "laptop",
                        "mouse",
                        "remote",
                        "keyboard",
                        "cell phone",
                        "microwave",
                        "oven",
                        "toaste",
                        "sink",
                        "refrigerator",
                        "blende",
                        "book",
                        "clock",
                        "vase",
                        "scissors",
                        "teddy bear",
                        "hair drier",
                        "toothbrush",
                        "hair brush"
    };
}
