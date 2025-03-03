import numpy as np
import scipy.io
from math import *
import torch
from torch import from_numpy as torchnp
from principal_RBM_alpha import lire_alpha_digit, RBM
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_data = "data/binaryalphadigs.mat"
caracs = {
    '0':0, '1':1, '2':2, '3':3, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 
    'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18,
    'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27,
    'S':28, 'T':29, 'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35 
}

class DNN:
    def __init__(self):
        pass

    def init_DBN(self, layers):
        self.layers = layers
        self.rbms = []
        for i in range(len(layers)-1):
            rbm = RBM()
            rbm.init_RBM(layers[i], layers[i+1])
            self.rbms.append(rbm)
        return self.rbms
    
    def train_DBN(self, donnees_entree, layers, epochs=1000, lr=0.1, mini_batch_size=10):
        self.rbms = self.init_DBN(layers)
        for i in range(len(self.rbms)):
            if i == 0:
                donnees = donnees_entree
            else:
                donnees = self.rbms[i-1].entree_sortie_RBM(donnees)
            self.rbms[i].train_RBM(donnees, epochs, lr, mini_batch_size)
        return self.rbms
        
    
def generer_image_DBN(DNN, epochs=1000, nb_images=1):
    for i in range(nb_images):
        donnees = torch.randn(1, DNN.rbms[0].p).to(device)
        for i in range(len(DNN.rbms)):
            donnees = DNN.rbms[i].entree_sortie_RBM(donnees)
        donnees = donnees.cpu().detach().numpy()
        plt.imshow(donnees.reshape(39, 39))
        plt.show()
