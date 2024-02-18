# tensorflow lite efficientdet model
Example of tensorflow lite pretrained EfficientDet Object detection model using C++ and Python. 

Download tensorflow lite lite4-detecetion-default efficientdet model from kaggle and save it to the somewhere in the repo.
Change params.json accordingly.

Download tensorflow lite in c++ repo and build it.

Run cpp app with ./main_tflite_model <relative path to params.json>

Results:
![highway1_processed](https://github.com/Akul123/tensorflow_lite_efficientdet/assets/33400468/b15038d4-653e-4b59-92c2-1cdab1c58fe1)
![highway_processed](https://github.com/Akul123/tensorflow_lite_efficientdet/assets/33400468/28eba04d-0cb2-427f-bded-6a8924efe900)
![metro_processed](https://github.com/Akul123/tensorflow_lite_efficientdet/assets/33400468/daf9d1b0-4387-4f3c-92d4-c7c2dbb98194)
![sailing-boat_processed](https://github.com/Akul123/tensorflow_lite_efficientdet/assets/33400468/fa23adb2-dbef-42a3-80fc-60bc1f71ad5b)


Inference times c++:

![image](https://github.com/Akul123/tensorflow_lite_efficientdet/assets/33400468/140d2401-df2b-47e4-a763-306ae0340ea8)

...and python:

![image](https://github.com/Akul123/tensorflow_lite_efficientdet/assets/33400468/8823dcb1-5fcd-4915-b10e-9d5b5cbf862a)

