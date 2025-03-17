from principal_DBN_alpha import *
from utils import *
from tqdm import tqdm
import torch


class DNN():
    
    def __init__(self, layers_dbn, nb_classes):
        self.dbn = DBN(layers_dbn)
        # ajouter une couche de classification supplémentaire
        self.dbn.list_RBM.append(RBM(p=layers_dbn[-1], q=nb_classes))
        

    def pretrain_DNN(self, x, epochs, lr, batch_size=None):
        self.dbn.train_DBN(x=x, epochs=epochs, lr=lr, train_layers=self.dbn.nb_couche-1, batch_size=batch_size)
        
        
    def calcul_softmax(self, rbm, data):
        x = data @ rbm.W + rbm.b
        return torch.exp(x) / (torch.exp(x).sum(axis=2, keepdims=True))

    def entree_sortie_reseau(self, data):
        sortie = []
        if type(data) == np.ndarray: 
            data = torch.from_numpy(data).double().to(device)
        for rbm in self.dbn.list_RBM[:-1]:
            data = rbm.entree_sortie_RBM(data)
            sortie.append(data)
            data = torch.bernoulli(data)
        rbm = self.dbn.list_RBM[-1]
        y_hat = self.calcul_softmax(rbm, data)
        sortie.append(y_hat)
        return sortie
    
    def retropropagation(self, X,Y, epochs, lr, batch_size = None, plot=False, show_progress=False):
        if type(X) == np.ndarray: X = torch.from_numpy(X).double().to(device)
        if type(Y) == np.ndarray: Y = torch.from_numpy(Y).double().to(device)
        
        
        if batch_size is None:
            batch_size = int(np.floor(0.2 * X.shape[0]))


        loss_list = []
        epochs_iterator = tqdm(range(epochs), desc="Training DNN", unit="epoch", disable=not show_progress)
        for _ in epochs_iterator:       
                
            shuffled_indices = np.arange(X.shape[0])
            np.random.shuffle(shuffled_indices)
            X_shuffled = X[shuffled_indices]
            Y_shuffled = Y[shuffled_indices]
            
            for j in range(0, X.shape[0], batch_size):
                x = X_shuffled[j:min(j + batch_size, X.shape[0])]
                y = Y_shuffled[j:min(j + batch_size, X.shape[0])]
                

                sortie = self.entree_sortie_reseau(x)
                y_hat = sortie[-1]
                loss = -(y * torch.log(y_hat)).sum() 
                loss_list.append(loss.cpu())
                if show_progress:
                    epochs_iterator.set_postfix({"Loss": loss.item()})
                
                # Mise à jour de la dernière couche 
                last_layer = self.dbn.list_RBM[-1]
                delta_b_last = y_hat - y  
                delta_W_last = sortie[-2].T @ delta_b_last  

                last_layer.W -= lr * delta_W_last.mean(axis=0)
                last_layer.b -= lr * delta_b_last.mean(axis=0)

               
                delta_b = delta_b_last  
                for i in range(self.dbn.nb_couche - 2, -1, -1):  
                    layer = self.dbn.list_RBM[i]
                    
                    if i != 0:
                        # Pour les couches cachées (sauf la première)
                        delta_b = (delta_b @ self.dbn.list_RBM[i + 1].W.T) * (sortie[i] * (1 - sortie[i]))
                        delta_W = sortie[i - 1].T @ delta_b
                    else: 
                        # Pour la première couche
                        delta_b = (delta_b @ self.dbn.list_RBM[i + 1].W.T) * (sortie[i] * (1 - sortie[i]))
                        delta_W = x.T @ delta_b
                    
                    layer.W -= lr * delta_W.mean(axis=0) # (sum over batch divide by batch_size)
                    layer.b -= lr * delta_b.mean(axis=0)
        if plot is True:
            plt.figure()
            plt.plot(loss_list)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.show()
        return 0

    def test_DNN(self, x, y):
        sortie = self.entree_sortie_reseau(x)
        y_hat = sortie[-1].cpu()
        y_pred = torch.argmax(y_hat, axis=2).cpu()
        y_true = torch.argmax(y, axis=2).cpu()
        error = (y_pred != y_true).sum()/len(y_true)
        print(f"Taux d'erreur = {error}")
        return error, y_hat


if __name__ == '__main__':
    # Test DNN sur 2 lettres de Binary Alpha Digits
    path_data = "data/binaryalphadigs.mat"
    data, nb_pixels = lire_alpha_digit(["A", "B"], path_data)
    layers_dbn = [nb_pixels, 200, 200]
    dnn = DNN(
        layers_dbn = layers_dbn,
        nb_classes=2
        )
    dnn.pretrain_DNN(x=data, epochs=[100], lr=0.1)
    # y shape shoulde be (n, 1, 2) where n is the number of images
    # test y shape
    y1 = np.array([[[1, 0]]]*int(data.shape[0]/2))
    y2 = np.array([[[0, 1]]]*int(data.shape[0]/2))
    y = np.concatenate((y1, y2), axis=0)
    y = torch.from_numpy(y).double().to(device)
    if y.shape != (data.shape[0], 1, 2):
        raise ValueError("y shape should be (n, 1, 2) where n is the number of images")
    sortie = dnn.retropropagation(X=data, Y=y, epochs=300, lr=0.1, show_progress=True)
    # Données de test
    tau, y_hat = dnn.test_DNN(x=data, y=y)
    print(y_hat)