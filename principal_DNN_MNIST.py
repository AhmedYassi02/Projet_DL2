from principal_DBN_alpha import *
import torch


class DNN():
    
    def __init__(self, Q, nb_classes):
        self.dbn = DBN(Q)
        # ajouter une couche de classification supplémentaire
        self.dbn.list_RBM.append(RBM())
        self.dbn.list_RBM[-1](p=Q[-1], q=nb_classes) # q est le nombre de classes
        

    def pretrain_DNN(self, x, epochs, lr, batch_size=None):
        self.dbn.train_DBN(x=x, epochs=epochs, lr=lr, train_layers=self.dbn.nb_couche-1, batch_size=batch_size)
        
        
    def calcul_softmax(self, rbm):
        x = np.array(rbm.b) + np.dot(data, rbm.W)
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
        proba = self.calcul_softmax(rbm)
        sortie.append(proba)
        return sortie
    
    def retropropagation(self, X,Y, epochs, lr, batch_size = None, plot=False, show_progress=False):
        if type(X) == np.ndarray: X = torch.from_numpy(X).double().to(device)
        if type(Y) == np.ndarray: Y = torch.from_numpy(Y).double().to(device)
        
        
        if batch_size is None:
            batch_size = int(np.floor(0.2 * X.shape[0]))


        loss_list = []
        epochs_iterator = tqdm(range(epochs), desc="Training DNN", unit="epoch", disable=not show_progress)
        for e in epochs_iterator:       
                
            shuffled_indices = np.arange(X.shape[0])
            np.random.shuffle(shuffled_indices)
            X_shuffled = X[shuffled_indices]
            Y_shuffled = Y[shuffled_indices]
            
            for j in range(0, X.shape[0], batch_size):
                x = X_shuffled[j:min(j + batch_size, X.shape[0])]
                y = Y_shuffled[j:min(j + batch_size, X.shape[0])]
                

                sortie = self.entree_sortie_reseau(x)
                proba = sortie[-1]
                loss = -(y * torch.log(proba)).sum() 
                loss_list.append(loss.cpu())
                if show_progress:
                    epochs_iterator.set_postfix({"Loss": loss.item()})
                
                # backpropagation sur TOUTES les couches du réseau
                for i in reversed(range(self.nb_couche)):
                    couche = self.dbn.list_RBM[i]
                    # calcul du gradient
                    if i == self.dbn.nb_couche - 1:
                        delta_W = torch.transpose(sortie[i-1], 1, 2) @ (proba - y)
                        delta_b = proba - y
                    elif i != 0:
                        delta_b = (delta_b @ self.dbn.list_RBM[i+1].W.T) * (sortie[i] * (1 - sortie[i]))
                        delta_W = torch.transpose(sortie[i-1], 1, 2) @ delta_b
                    else:
                        delta_b = (delta_b @ self.dbn.list_RBM[i+1].W.T) * (sortie[i] * (1 - sortie[i]))
                        delta_W = torch.transpose(x, 1, 2) @ delta_b
                    delta_W_mean = delta_W.mean(axis=0)
                    delta_b_mean = delta_b.mean(axis=0)
                    # mise à jour des poids
                    couche.W -= lr * delta_W_mean
                    couche.b -= lr * delta_b_mean
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
        proba = sortie[-1].cpu()
        y_pred = torch.argmax(proba, axis=2).cpu()
        y_true = torch.argmax(y, axis=2).cpu()
        taux_erreur = (y_pred != y_true).sum()/len(y_true)
        print(f"Taux d'erreur = {taux_erreur}")
        return taux_erreur, proba
