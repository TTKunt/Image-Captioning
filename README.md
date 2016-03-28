Generating Natural Descriptions for Images
===================

Image Captioning Project

## Requirements ##
- java 1.8.0
- python 2.7
- other python packages in requirements.txt

## Files ##
./
- main.py (contains all necessary functions, follow instructions on this file for running)
- server.py (for running on your own server for other people can use online)

./dataset
- Download dataset folder from [here](https://drive.google.com/folderview?id=0B34JWl6eUDsUZVJ2UF8xdDVaRDQ&usp=sharing)

./models
- you can run the main.py above to create your own models files
- Or download MSCOCO model 1 and 2 for testing from [here](https://drive.google.com/folderview?id=0B34JWl6eUDsUdC1ocWgxUW16d0E&usp=sharing)

./results: contains the results of running MSCOCO dataset

./eval: The folder where all evaluation codes are stored.

./requirements.txt: All Python packages need for running this project. You can install by running: `pip install -r requirements.txt`

## References ##

- Show and Tell: A Neural Image Caption Generator. Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan; The IEEE [http://arxiv.org/pdf/1411.4555.pdf](http://arxiv.org/pdf/1411.4555.pdf)
- Deep Visual-Semantic Alignments for Generating Image Descriptions. Andrej Karpathy, Li Fei-Fei [http://cs.stanford.edu/people/karpathy/cvpr2015.pdf](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio [http://arxiv.org/pdf/1502.03044v2.pdf](http://arxiv.org/pdf/1502.03044v2.pdf)