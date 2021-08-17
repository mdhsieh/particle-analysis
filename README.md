# particle-analysis

Models which were selected to perform object detection and tracking of indoor disinfectant particles or microscope images.
Each top-level folder has 1 separate model: Mask-RCNN, U-Net, and PointRNN. 

- Mask R-CNN: Bounding box detection and image segmentation of microscope images
- U-Net: Image segmentation of microscope images
- PointRNN: Point cloud object detection and tracking

The `MaskRCNN/r-rcnn` folder performs detection on multiple cell nuclei and tracking on a single cell nucleus.
It was trained on [this dataset](https://www.kaggle.com/c/data-science-bowl-2018/data).

## PointRNN folder is copied from "https://github.com/hehefan/PointRNN" and is NOT an original work of our own. Please see and cite the original repository. 

