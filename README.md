# Mask_Detector with MobileNet
 Toy Project with OpenCV
 
## Requirment
 * imutil
 * tensorflow
 * matplotlib
 * numpy
 * scikit-learn
 * python-opencv (cv2)

## Usage
open mask_detector.py in CMD
> python mask_detector.py

## How to learn
Make own dataset(with mask image and without mask image) like below. (Folder name is just example)
```
dataset/
├── withmask/
│   └── with mask images
└── withoutmask/
    └── without mask images
```    
And, define dataset directory path at line 19.

And, run train.py  
You should define model save path at last line.
