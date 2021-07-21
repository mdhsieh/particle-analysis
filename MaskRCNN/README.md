## Development Instructions

Mask R-CNN with LSTM is in `r-rcnn` folder.
This is from the [Recurrent Mask R-CNN repository](https://github.com/cechung/R-RCNN),
which is based off the [Matterport implementation](https://github.com/matterport/Mask_RCNN) of Mask-RCNN.

### Download trained weights
Mask-RCNN uses a [cell nuclei weight](https://drive.google.com/file/d/120B-3C-X2AGAcLDrddvFE39VZj-6-pS5/view?usp=sharing) trained on [this Kaggle dataset](https://www.kaggle.com/c/data-science-bowl-2018) 

LSTM uses [this Object Tracking Benchmark weight](https://drive.google.com/file/d/1g0Yxrs4YeA9ft_1Lul-JRNZvEMcIE781/view)

### Run
Run `lstm_mask_rcnn_train_and_inference.ipynb`, which runs `demo_mrcnn.py` on images in `demo_video_nucleus/frames`.
If using Colab, restart runtime after pip installations to avoid some errors.
