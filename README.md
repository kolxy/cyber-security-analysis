# Cyber Security Dataset Analysis Project

This is the codebase for our Computer Science 575 class project at WSU. We are performing attack detection and classification
using a variety of methods such as MLP, logistic regression, random forest, and k-nearest neighbors. We are also toying with
Matrix Profiling to see if it has any use as an attack detector on the UNSW-NB15 Dataset by 
[Nour Moustafa and Jill Slay](https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files). 
In addition, another paper is comparing PCA with autoencoders for dimensionality reduction.

## Associated Papers

1. [Comparing Methods for Network Attack Classification](https://www.overleaf.com/read/txttqnhtdnrp)
2. [Pending](#)

## Setup
Modules and versions
```bash
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Download the data

The data is stored [here](https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys?path=%2FUNSW-NB15%20-%20CSV%20Files). 

## Preprocessing
Produce cleaned dataset

```bash
$ python preprocessing.py
```

## Running Models

```bash
$ python <model_name>.py
```

## Citations

As per the request of the authors, here citations for utilizing the dataset:

1. Moustafa, Nour, and Jill Slay. ["UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)."](https://ieeexplore.ieee.org/abstract/document/7348942) *Military Communications and Information Systems Conference (MilCIS)*, 2015. IEEE, 2015.
2. Moustafa, Nour, and Jill Slay. ["The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset."](https://www.tandfonline.com/doi/abs/10.1080/19393555.2015.1125974) *Information Security Jounal: A Global Perspective* (2016): 1-14.
3. Moustafa, Nour, et al. ["Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks."](https://ieeexplore.ieee.org/abstract/document/7948715) *IEEE Transactions on Big Data (2017).*
4. Moustafa, Nour, et al. ["Big data analytics for intrusion detection system: statistical decision-making using finite dirichlet mixture models."](https://link.springer.com/chapter/10.1007/978-3-319-59439-2_5) *Data Analytics and Decision Support for Cybersecurity. Springer, Cham, 2017.* 127-156.
5. Sarhan, Mohanad, Siamak Layeghy, Nour Moustafa, and Marius Portmann. [NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems.](https://arxiv.org/abs/2011.09144). In [Big Data Technologies and Applications: 10th EAI International Conference, BDTA 2020, and 13th EAI International Conference on Wireless Internet, WiCON 2020, Virtual Event, December 11, 2020, Proceedings (p. 117). Springer Nature.](https://link.springer.com/chapter/10.1007/978-3-030-72802-1_9)
