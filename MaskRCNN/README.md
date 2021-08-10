## Development Instructions

Mask R-CNN with LSTM is in `r-rcnn` folder.
This is from the [Recurrent Mask R-CNN repository](https://github.com/cechung/R-RCNN),
which is based off the [Matterport implementation](https://github.com/matterport/Mask_RCNN) of Mask-RCNN.

### Download trained weights
Mask-RCNN uses a [cell nuclei weight](https://drive.google.com/file/d/120B-3C-X2AGAcLDrddvFE39VZj-6-pS5/view?usp=sharing) trained on [this Kaggle dataset](https://www.kaggle.com/c/data-science-bowl-2018) 

LSTM uses [this Object Tracking Benchmark weight](https://drive.google.com/file/d/1g0Yxrs4YeA9ft_1Lul-JRNZvEMcIE781/view)

### Run on HPC
##### To run on SJSU HPC after logging in
```
# create a virtual environment and activate
module load python3/3.7.0 cuda/10.0
virtualenv --system-site-packages -p python3.7 ./rrcnn_env_3.7.0
source ./rrcnn_env_3.7.0/bin/activate
```

```
# install libraries in the environment
pip install -r requirements.txt
pip install tensorflow==1.15.2
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip install keras==2.3.1
pip install h5py==2.10.0
```

Change lines:

`detection_module/detect_model.py`, lines 2054, 2076, 2079 replace the "topology" with "saving".
https://github.com/matterport/Mask_RCNN/issues/694 

`utils/visualize.py`, line 537 replace `font = 	ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)` to `font = ImageFont.load_default()`
https://stackoverflow.com/questions/47694421/pil-issue-oserror-cannot-open-resource 

##### Request GPU node
`demo_rrcnn.py` requires GPU.
```
srun -p gpu --gres=gpu --pty /bin/bash
```

##### Run
`cd LSTM_Mask_RCNN/`

Run detection without tracking on `demo_video_nucleus/frames`:
`python demo_mrcnn.py`

Run detection with tracking on `demo_video_nucleus_blocked/frames`:
`python demo_rrcnn.py`

The results are in `mrcnn_results` and `rrcnn_results` folders, respectively.

### Run using Google Colab
Run `lstm_mask_rcnn_train_and_inference.ipynb`, which runs `demo_mrcnn.py` on images in `demo_video_nucleus/frames`.
When using Colab, restart the runtime after pip installations to avoid some errors.

Change runtime to GPU to run `demo_rrcnn.py`. 
