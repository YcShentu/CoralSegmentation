# CoralSegmentation
paper: Development of Coral Investigation System Based on Semantic Segmentation of Single-Channel Images  
doi: 10.3390/s21051848  
[Abstract](https://www.mdpi.com/1424-8220/21/5/1848)  
[PDF Version](https://www.mdpi.com/1424-8220/21/5/1848/pdf)  


# Dataset: CoralS
Only RGB images and their masks in CoralS are released.  

|    |  classification   | segmentation  |
|  ----  | ----  |  ----  |
| images  | 3866 | 576 (positive)|
  
notes:  
  * for classification, image name starts with '0' is negative (non-coral),
  otherwise, if it starts with '1' is positive (coral).
  * for segmentation, only positive images are provided with labeled masks. 
  Negative images in classification can also the negative data for segmentation.  
  
  
Links for downloads:  
* [zju pan](**) (links will be released later)
* [baiduyun](https://pan.baidu.com/s/1JDUgInzgikZ3CJgf3OHXTQ) password: 1mna 


# Train
## prepare
1. download dataset
2. download codes & install requirements
```
git clone https://github.com/YcShentu/CoralSegmentation.git
cd CoralSegmentation
pip install -r requirements.txt
```
## pretrain: classification
all the params for training is defined in TrainingHyperparameters [config.py](config.py) 
use the following cmd for help
```
python pretrain.py -h
```

## segmentation
all the params for training are defined in TrainingHyperparameters [config.py](config.py) 
use the following cmd for help
```
python train.py -h
```


# Evaluation
all the params for evaluation are defined in EvalHyperparameters [config.py](config.py)  
use the following cmd for help  
```
python evaluation.py -h 
```

# Inference
all the params for inference are defined in InferenceHyperparameters [config.py](config.py)  
use the following cmd for help
```
python inference.py -h
```

# Notes
* This work is part of my thesis. 
Due to limited time and capacity, this work is still needed to be improved. 
The code has been completed long time ago, it may seem a bit outdated. 
* I am very grateful to Mehdi for reorganizing and publishing my postgraduate work after I graduated. 
In addition, I'd like to thank Chaopeng for his work in spectral data collection, 
although he is not among the authors of the article.

# Citation
Please consider citing the corresponding publication if you use this work in an academic context:  
```
@{sarlin2019coarse,
  title={Development of Coral Investigation System Based on Semantic Segmentation of Single-Channel Images},
  author={Hong Song, Syed Raza Mehdi, Yangfan Zhang, Yichun Shentu, etc},
  booktitle={sensors},
  year={2021}
}
```