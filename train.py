"""train script"""
import os

#os.chdir("D:/desktop_d/cv-ecnu/perceptron")

os.system("python main.py --dataset mnist "
          "--data_dir ./data/mnist "
          "--seed 821 --split train --out_dir ./result/knn-test_num100-k_range20 "
          "--exp_name knn --test_num 100 --k_range 20")
