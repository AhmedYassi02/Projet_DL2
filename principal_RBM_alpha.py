import numpy as np
import scipy.io
from math import *
import torch
from torch import from_numpy as torchnp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_data = "data/binaryalphadigs.mat"
caracs = {
    '0':0, '1':1, '2':2, '3':3, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 
    'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18,
    'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27,
    'S':28, 'T':29, 'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35 
}

def lire_alpha_digit(caractere_list, path_data):
    # caractere_list = [caracs[el] for el in caractere_list]
    caractere_list = map(lambda x: caracs[x], caractere_list)
    data = scipy.io.loadmat(path_data)
    data = list(data['dat'][i] for i in caractere_list)
    final_data = []
    # For each letter
    for i in range(len(caractere_list)):
        data_tmp = [data[i][j].flatten() for j in range(39)] 
        final_data.append(np.vstack(data_tmp))
        if np.vstack(data_tmp).shape[0] != 39:
            raise ValueError
        
    p = final_data[0].shape[1]
    final_data = np.vstack(final_data)
    final_data = np.resize(final_data, (final_data.shape[0], 1, final_data.shape[1]))

    return final_data, p


class RBM:
    def __init__(self):
        pass    

    def init_RBM(self, p, q): 
        self.p = p # Nombre de neurones d'entrée => doit être le même que taille des données, à vérifier lors de l'entraînement
        self.q = q  # hyperparameters 
        self.b = torchnp(np.zeros((1, self.q))).to(device)
        self.a = torchnp(np.zeros((1, self.p))).to(device)
        self.W = torchnp(np.random.normal(0, 0.01, size=(self.p, self.q))).to(device)

    
    def sig(self, x):
        return 1/(1 + torch.exp(-x))
    
    def entree_sortie_RBM(self, donnees_entree, sig=True):
        if donnees_entree.shape[1]!= 1:
            raise ValueError("Erreur : 'donnees_entree' n'est pas de la bonne dimension (n, 1, p).")
        if sig is True:
            res = self.sig(donnees_entree @ self.W + self.b)
        else :
            res = donnees_entree @ self.W + self.b
        return res

    def sortie_entree_RBM(self, donnees_sortie):
        return self.sig(donnees_sortie @ torch.transpose(self.W, 0, 1) + self.a) 