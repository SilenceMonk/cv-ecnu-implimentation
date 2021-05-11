# cv-ecnu
## include Perceptron & Knn implimentation 
### basic usage
run [train.py](train.py)

### run custom settings
some arguments to use in [train.py](train.py)
```
[--dataset], default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'imagenet'], help='dataset: mnist, cifar10, cifar100, svhn or imagenet'
[--split], default='all', choices=['all', 'train', 'test'], help='choose whether to use train set, test set or both'                   
[--data_dir], default='./data/mnist', help='folder where data is stored'                   
[--out_dir], default='./result/result-default', help='where the output image is stored'                   
[--seed], dest='seed', default=666, type=int, help='define seed for random shuffle of dataset')
[--exp_name], choices=['knn', 'perceptron'], help='select which experiment to run')

# args for training perceptron
[--perceptron_iter], default=1000, help='total perceptron training iterations')
[--lr], default=1e-3, help='learning rate'

# args for knn
[--k], default=1, help='default "k" for knn')
[--k_range], default=3, help='search a range of "k" for knn')
[--test_num], default=100, help='test_num for knn')
```  
### sample script for training perceptron:
```python
"""customized train script for perceptron"""
import os

#os.chdir("D:/perceptron")

os.system("python main.py --dataset mnist "
          "--data_dir ./data/mnist "
          "--seed 821 --split train --out_dir ./result/perceptron-train "
          "exp_name perceptron --perceptron_iter 1500 --lr 1e-5")
```

### sample script for knn:
```python
"""customized train script for knn"""
import os

#os.chdir("D:/desktop_d/cv-ecnu/perceptron")

os.system("python main.py --dataset mnist "
          "--data_dir ./data/mnist "
          "--seed 821 --split all --out_dir ./result/knn-test_num10-k_range3 "
          "--exp_name knn --test_num 50 --k_range 3")
```