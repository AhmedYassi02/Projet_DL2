import numpy as np
import scipy.io
from math import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RBM:
    def __init__(self, p, q): 
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
    
    def train_RBM(self, x, epochs, lr, batch_size = None, plot=False, show_progress=False) :
        """Boucle d'entraînement pour une RBM.

        Args:
            x (np.array): Données d'entraînement
            epochs (int): Nombre d'itérations pour chaque couche
            lr (float): Taux d'apprentissage
            batch_size (int): Taille du batch
            plot (bool): Si True, affiche la courbe d'erreur de reconstruction
            show_progress (bool): Si True, affiche la barre de progression de l'entraînement
        """

        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x).to(device=device, dtype=torch.double)  
        else:
            x = x.to(device)
            
        error_list = []
        if batch_size is None:
            batch_size = int(x.shape[0]*0.2)
        epoch_iterator = range(epochs)
        if show_progress:
            epoch_iterator = tqdm(epoch_iterator, desc="Training RBM", unit="epoch")
        for epoch in epoch_iterator:
            shuffled_indices = np.arange(x.shape[0])
            np.random.shuffle(shuffled_indices)
            X0_suffled = x[shuffled_indices]
            
            for j in range(0, x.shape[0], batch_size):
                X_batch = X0_suffled[j:min(j + batch_size, x.shape[0])]
                v0 = X_batch # [batch_size, 1, p]
                ## Forward pass
                p_h_v0 = self.entree_sortie_RBM(v0)  # [batch_size, 1, q]
                h0 = torch.bernoulli(p_h_v0) # [batch_size, 1, q]
                p_v_h0 = self.sortie_entree_RBM(h0)  # [batch_size, 1, p]
                v1 = torch.bernoulli(p_v_h0) # [batch_size, 1, p]
                p_h_v1 = self.entree_sortie_RBM(v1)  # [batch_size, 1, q]
                
                ## Updating weights
                grad_W = torch.transpose(v0, 1, 2) @ p_h_v0 - torch.transpose(v1, 1, 2) @ p_h_v1  # (p, 1, n) * (1, q, n) = (p, q, n)               
                grad_a = v0 - v1  # (n, 1, p) 
                grad_b = p_h_v0 - p_h_v1 # (n ,1, q)

                grad_W = grad_W.mean(axis=0) # sum over batch divide by batch_size
                grad_a = grad_a.mean(axis=0)
                grad_b = grad_b.mean(axis=0)

                self.W += lr * grad_W
                self.a += lr * grad_a
                self.b += lr * grad_b
                
            H = self.entree_sortie_RBM(x)
            X_rec = self.sortie_entree_RBM(H)
            erreur = torch.norm(x - X_rec, p='fro')**2 / X_rec.shape[0]
            error_list.append(erreur)
            if show_progress:
                epoch_iterator.set_postfix(reconstruction_error=f"{erreur:.2f}")
            # pbar.update(1)
        if plot :
            plt.figure()
            plt.plot(error_list)
            plt.ylabel('Reconstruction error')
            plt.xlabel('Epoch')
            plt.grid()
            plt.show()
            
    def generer_image_RBM(self, iterations_gibbs, nb_images, show=False):

        v = torch.from_numpy(np.random.random_sample((nb_images, 1, self.p))).to(device)
        v = torch.round(v)
        for _ in range(iterations_gibbs):
            p_h_v = self.entree_sortie_RBM(v)
            h = torch.bernoulli(p_h_v)
            p_v_h = self.sortie_entree_RBM(h) 
            v = torch.bernoulli(p_v_h)
        list_img = []
        for img in range(nb_images):
            X = np.reshape(v[img].cpu().flatten(), (20, 16))
            if show :
                plt.figure()
                im = plt.imshow(X, cmap='Greys')
                plt.show()
            list_img.append(X)

        return list_img 