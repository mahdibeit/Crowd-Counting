# # Crowd Counting with Perceptual Loss Function

## Pipeline
![Pipeline](/image/model.jpg)

## Abstract
Single-image crowd counting is a challenging task, mainly due to the difficulty stems from the huge scale variation of people, severe occlusions among dense crowds, and limited samples in the available dataset. This project aims to develop, analyze, and evaluate methods that can accurately estimate the crowd count from a single image-based and generate the density map of images. To this end, we propose novel ideas in three main parts of preprocessing, model architectures, and loss function of our deep learning pipeline. More specifically, we utilize transfer learning methods by using pre-trained depth and image models to develop depth-guided attention models and VGG-based U-Net architecture to address the limited number of samples in the dataset and increase the accuracy. Further, we systematically analyze the effect of loss functions on the performance of deep learning models and show that the typical loss functions used in research that are based on pixel-wise similarities are defective in high dense crowds. In this regard, we propose a novel perceptual loss function based on a pre-trained autoencoder. Further, for comparison, we successfully implement and systematically analyze the recent works and methods for crowd counting estimation and highlight the gaps and future direction for improvements. Our implementation results on the ShanghaiTech dataset outperform many previous works and show the effectiveness of our novel methods. 

## Outputs
<p float="left">
  <img src="/image/OUT1.png" width="500" />
  <img src="/image/OUT2.png" width="500" /> 
</p>
<p float="left">
  <img src="/image/IMG_45.png" width="500" />
  <img src="/image/IMG_45.jpg" width="500" /> 
</p>

## valuated and Tested Model Architectures
<p align="center">
  <img src="/image/ev.JPG" /> 
</p>

## AE and MSE on the ShanghaiTech A datasets
![Results](/image/res.JPG)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Pytorch. 

```bash
pip install pytorch
```

## Usage
The default dataset for this project is ShanghaiTech Dataset. You can use your own dataset by adjusting the Get_Data module accordingly. Then, adjust the path as follows.
```python
'''Load the path of the data'''
    python preprocess_dataset.py --origin_dir <directory of original data>
    python train.py
    python test.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
The main structrue of the code is inspired from the work of "Bayesian Loss for Crowd Count Estimation with Point Supervision" in https://github.com/ZhihengCV/Bayesian-Crowd-Counting
