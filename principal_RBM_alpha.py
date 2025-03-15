import numpy as np
import scipy.io
from math import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_data = "data/binaryalphadigs.mat"
caracs = {
    '0':0, '1':1, '2':2, '3':3, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 
    'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18,
    'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27,
    'S':28, 'T':29, 'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35 
}

def lire_alpha_digit(learn_carac, path_data):
    n = len(learn_carac)
    learn_carac = map(lambda x: caracs[x], learn_carac)
    data = scipy.io.loadmat(path_data)
    data = list(data['dat'][i] for i in learn_carac)
    final_data = []
    for i in range(n):
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
        self.p = p
        self.q = q  # ( to fine-tune )
        self.b = torch.zeros((1, self.q), dtype=torch.double, device=device)
        self.a = torch.zeros((1, self.p), dtype=torch.double, device=device)
        self.W = torch.normal(0, 0.01, size=(self.p, self.q), device=device, dtype=torch.double)

    def entree_sortie_RBM(self, X_H, sigmoid=True):
        if sigmoid :
            return torch.sigmoid(X_H @ self.W + self.b)
        else :
            return X_H @ self.W + self.b
        
    def sortie_entree_RBM(self, donnees_sortie):
        return torch.sigmoid(donnees_sortie @ torch.transpose(self.W, 0, 1) + self.a)
    
    def train_RBM(self, epochs, lr, batch_fraction, x, verbose=False) :
        # vérifier type de x et le mette en torch
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).to(device)
            x = x.double()
        x = x.to(device)
        assert 0 < batch_fraction <= 1
        batch_size = max(1, int(np.floor(batch_fraction * x.shape[0])))
        # ////!\\\\ x doit être de la taille (n, 1, p)
        if self.p != x.shape[2]:
            raise ValueError("p doit être égal à la taille d'une donnée d'entrée (nombre de pixels).")
        # à répéter pour chaque epoch
        acc_list = []
        with tqdm(total=epochs, desc="Training RBM", unit="epoch") as pbar:
            for e in range(epochs):
            # for e in tqdm(range(epochs), desc="Training RBM", unit="epoch"):  
                v0 = x # données d'entrees  ////!\\\\ (n, 1, p) n en première dimension !!!
                shuffled_indices = np.arange(v0.shape[0])
                np.random.shuffle(shuffled_indices)
                v_0_shuffled = v0[shuffled_indices]
                num_batches = np.ceil(v0.shape[0] / batch_size).astype(int)
                list_v_0 = [v_0_shuffled[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
                for v0 in list_v_0 :
                    P_h_v0 = self.entree_sortie_RBM(v0)  # (n, 1, q)
                    h0 = torch.bernoulli(P_h_v0)
                    p_v_h0 = self.sortie_entree_RBM(h0)  
                    v1 = torch.bernoulli(p_v_h0)
                    p_h_v1 = self.entree_sortie_RBM(v1)  # (n, 1, q)
                    delta_W = torch.transpose(v0, 1, 2) @ p_h_v1 - torch.transpose(v1, 1, 2) @ p_h_v1  # (p, 1, n) * (1, q, n) = (p, q, n)
                    delta_a = v0 - v1  # (n, 1, p) 
                    delta_b = P_h_v0 - p_h_v1 # (n ,1, q)

                    delta_W = delta_W.mean(axis=0)
                    delta_a = delta_a.mean(axis=0)
                    delta_b = delta_b.mean(axis=0)

                    self.W += lr * delta_W
                    self.a += lr * delta_a
                    self.b += lr * delta_b

                acc = torch.abs((v0 - v1)).mean()
                acc_list.append(acc.cpu())
                pbar.set_postfix(acc=f"{acc:.4f}")
                pbar.update(1) 
        if verbose == 2:
            plt.figure()
            plt.plot(acc_list)
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.grid()
            plt.show()
            
    def generer_image_RBM(self, iterations_gibbs, nb_images, show=False):
        # initialisation donnees v
        v = torch.from_numpy(np.random.random_sample((nb_images, 1, self.p))).to(device)
        v = torch.round(v)
        for i in range(iterations_gibbs):
            tirage_h = self.entree_sortie_RBM(v)  # (n, 1, q)
            h = torch.bernoulli(tirage_h)
            # Tirage de v (taille p x 1) dans loi p(v|h^0)
            tirage_v = self.sortie_entree_RBM(h)  # (n, 1, p)
            v = torch.bernoulli(tirage_v)
        list_img =[]
        for img in range(nb_images):
            X = np.reshape(v[img].cpu().flatten(), (20, 16))
            if show is True:
                plt.figure()
                im = plt.imshow(X, cmap='Greys')
                plt.show()
            list_img.append(X)

        return list_img 