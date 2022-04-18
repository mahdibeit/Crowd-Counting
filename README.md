# # Crowd Counting with Transfer Learning-based Models and Perceptual Loss Function

## Pipeline


## Abstract
Single-image crowd counting is a challenging task, mainly due to the difficulty stems from the huge scale variation of people, severe occlusions among dense crowds, and limited samples in the available dataset. This project aims to develop, analyze, and evaluate methods that can accurately estimate the crowd count from a single image-based and generate the density map of images. To this end, we propose novel ideas in three main parts of preprocessing, model architectures, and loss function of our deep learning pipeline. More specifically, we utilize transfer learning methods by using pre-trained depth and image models to develop depth-guided attention models and VGG-based U-Net architecture to address the limited number of samples in the dataset and increase the accuracy. Further, we systematically analyze the effect of loss functions on the performance of deep learning models and show that the typical loss functions used in research that are based on pixel-wise similarities are defective in high dense crowds. In this regard, we propose a novel perceptual loss function based on a pre-trained autoencoder. Further, for comparison, we successfully implement and systematically analyze the recent works and methods for crowd counting estimation and highlight the gaps and future direction for improvements. Our implementation results on the ShanghaiTech dataset outperform many previous works and show the effectiveness of our novel methods. 

![Pipeline](/images/Pipeline.jpg)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Pytorch. 

```bash
pip install pytorch
```

## Usage
The default dataset for this project is ["BCI Competition IV"](http://www.bbci.de/competition/iv/). You can use your own dataset by adjusting the Get_Data module accordingly. Then, adjust the path as follows.
```python
'''Load the training data and test data'''
    Trials, Label= Get_Data(1,True, 'DataSet/')
    Trials_test, Label_test= Get_Data(1,False, 'DataSet/')  
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Please reference the work when using this project.
