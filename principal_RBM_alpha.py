import numpy as np
import scipy.io
from math import *
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import lire_alpha_digit, caracs, path_data

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class RBM:
    def __init__(self, p, q): 
        self.p = p
        self.q = q  # ( to fine-tune )
        self.b = torch.zeros(self.q, device=device, dtype=torch.double)
        self.a = torch.zeros(self.p, device=device, dtype=torch.double)
        # self.W = torch.normal(0, 0.01, size=(self.p, self.q), device=device, dtype=torch.double)
        self.W = torch.randn(p, q, device=device, dtype=torch.double) * torch.sqrt(torch.tensor(2.0 / p, device=device, dtype=torch.double))

    def entree_sortie_RBM(self, X_H):
        return torch.sigmoid((X_H @ self.W + self.b))
        
    def sortie_entree_RBM(self, donnees_sortie):
        return torch.sigmoid(donnees_sortie @ self.W.T + self.a)
    
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
        epoch_iterator = tqdm(range(epochs), desc="Training RBM", unit="epoch", disable=not show_progress)

        for epoch in epoch_iterator:
            shuffled_indices = np.arange(x.shape[0])
            np.random.shuffle(shuffled_indices)
            X0_suffled = x[shuffled_indices]
            
            for j in range(0, x.shape[0], batch_size):
                X_batch = X0_suffled[j:min(j + batch_size, x.shape[0])]
                v0 = X_batch # [batch_size,  p]
                ## Forward pass
                p_h_v0 = self.entree_sortie_RBM(v0)  # [batch_size,  q]
                h0 = torch.bernoulli(p_h_v0) # [batch_size, q]
                p_v_h0 = self.sortie_entree_RBM(h0)  # [batch_size, p]
                v1 = torch.bernoulli(p_v_h0) # [batch_size, p]
                p_h_v1 = self.entree_sortie_RBM(v1)  # [batch_size, q]
                
                ## Updating weights
                grad_W = v0.T @ p_h_v0 - v1.T @ p_h_v1  # [p, batch_size] * [batch_size,  q]  = [p, q]     
                grad_a = v0 - v1  # [batch_size,  p]
                grad_b = p_h_v0 - p_h_v1 # [batch_size, q]
                
                grad_a = grad_a.mean(axis=0)# [p] # sum over batch divide by batch_size
                grad_b = grad_b.mean(axis=0) #[q]

                self.W += lr * grad_W/batch_size
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
            
    def generer_image_RBM(self, iterations_gibbs, nb_images):
        v = torch.randint(0, 2, (nb_images, self.p), dtype=torch.double, device=device) # [nb_images, p]
        
        for _ in range(iterations_gibbs):
            p_h_v = self.entree_sortie_RBM(v)
            h = torch.bernoulli(p_h_v)
            p_v_h = self.sortie_entree_RBM(h) 
            v = torch.bernoulli(p_v_h)
        list_img = []
        for img in range(nb_images):
            X = np.reshape(v[img].cpu().flatten(), (20, 16))
            list_img.append(X)

        return list_img 
    

# for debugging     
if __name__ == "__main__":
    list_rbm_caracs = []
    carac = ['C']
    data = lire_alpha_digit([carac], path_data)
    nb_pixels = data.shape[1]
    rbm = RBM(p = nb_pixels, q = 100)
    rbm.train_RBM(x=data, epochs=200, lr=0.1, show_progress=True)
    list_rbm_caracs.append(rbm)