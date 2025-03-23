from principal_DBN_alpha import *
from utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DNN():
    
    def __init__(self, layers_dbn, nb_classes):
        """__init__ Constructeur de la classe DNN
        Args:
            layers_dbn (list): liste de neuronnes de chaque couche du réseau 
            nb_classes (int): Nombre de classes à prédire
        """
        self.dbn = DBN(layers_dbn)
        # ajouter une couche de classification supplémentaire
        self.dbn.list_RBM.append(RBM(p=layers_dbn[-1], q=nb_classes))
        self.pretrained = False
        self.nb_classes = nb_classes
        

    def pretrain_DNN(self, x, epochs, lr, batch_size=None, show_progress=False):
        self.dbn.train_DBN(x=x, epochs=epochs, lr=lr, train_layers=self.dbn.nb_couche, batch_size=batch_size, show_progress=show_progress)
        self.pretrained = True
        
        
    def calcul_softmax(self, rbm, data):
        x = data.float() @ rbm.W + rbm.b
        return torch.exp(x) / (torch.exp(x).sum(axis=1, keepdims=True))

    def entree_sortie_reseau(self, data):
        v = data
        sortie = []
        if type(data) == np.ndarray: 
            data = torch.from_numpy(data).float().to(device)
        for rbm in self.dbn.list_RBM[:-1]:
            p_h = rbm.entree_sortie_RBM(v)
            v = 1*(p_h > torch.rand(p_h.shape, device=device)).to(device)
            v.float()
            sortie.append(p_h)
            
        y_hat = self.calcul_softmax(self.dbn.list_RBM[-1], v)
        sortie.append(y_hat)
        return sortie
    
    
    def retropropagation(self, X,Y, epochs, lr, batch_size = None, retun_loss=False, show_progress=False):
        
        # # initialisation des poids de la dernière couche car sinon ça marche pas
        couche = self.dbn.list_RBM[-1]
        couche.W = torch.randn(int(couche.W.shape[0]), int(couche.W.shape[1])).float().to(device)*0.01
        couche.b = torch.zeros((int(couche.b.shape[0]))).float().to(device)
        

        self.loss_list = []
        epochs_iterator = tqdm(range(epochs), desc="Training DNN", unit="epoch", disable=not show_progress)
        for e in epochs_iterator:       
                
            shuffled_indices = np.arange(X.shape[0])
            np.random.shuffle(shuffled_indices)
            X_shuffled = X[shuffled_indices]
            Y_shuffled = Y[shuffled_indices]
            
            for j in range(0, X.shape[0]+1, batch_size):
                x = X_shuffled[j:min(j + batch_size, X.shape[0])]
                y = Y_shuffled[j:min(j + batch_size, X.shape[0])]
                

                sortie = self.entree_sortie_reseau(x)
                y_hat = sortie[-1]
                
                loss = -torch.sum(y * torch.log(y_hat + 1e-8))/len(y)
                self.loss_list.append(loss.cpu())
                
                if show_progress:
                    epochs_iterator.set_postfix({"Loss": loss.item()})
                
                         
                for i in reversed(range(self.dbn.nb_couche)): 
                    if self.dbn.nb_couche == 1:
                        delta_b = y_hat - y
                        delta_W = x.T @ delta_b  
                    else:
                        if i == self.dbn.nb_couche - 1:
                            # Last layer (output layer)
                            delta_b = y_hat - y
                            delta_W = sortie[i - 1].T @ delta_b
                        elif i != 0:
                            # Hidden layers (except the first)
                            delta_b = (delta_b @ self.dbn.list_RBM[i + 1].W.T) * (sortie[i] * (1 - sortie[i]))  # delta_b @ W_L+1^T * f'(z_L)
                            delta_W = sortie[i - 1].T @ delta_b
                        elif i == 0:
                            # First layer (input layer)
                            delta_b = (delta_b @ self.dbn.list_RBM[i + 1].W.T) * (sortie[i] * (1 - sortie[i]))  # [batch_size, p] @ [p, q] * [batch_size, q] = [batch_size, p]
                            delta_W = x.T @ delta_b  # [p, batch_size] @ [batch_size, q] = [p, q] 
                    
                    # self.dbn.list_RBM[i].W -= lr * delta_W/batch_size 
                    self.dbn.list_RBM[i].W -= lr * delta_W/batch_size
                    self.dbn.list_RBM[i].b -= lr * delta_b.sum(axis=0)/batch_size # (sum over batch divide by batch_size)
            
        if retun_loss:
            return self.loss_list


    def test_DNN(self, x, y):
        sortie = self.entree_sortie_reseau(x)
        y_hat = sortie[-1].cpu()
        y_pred = torch.argmax(y_hat, axis=1).cpu()
        y_true = torch.argmax(y, axis=1).cpu()
        error = (y_pred != y_true).sum()/len(y_true)
        return error, y_hat



# for debug
if __name__ == '__main__':
    import torchvision.datasets
    import torchvision.transforms as transforms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    transform = transforms.ToTensor()
    # if not exist, download mnist dataset
    train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    # take only 300 samples so the model can overfit to see if it works
    # train_set.data = train_set.data[:300]
    # train_set.targets = train_set.targets[:300]
    train_set.data = (train_set.data > 127).float()
    test_set.data = (test_set.data > 127).float()
    # Applatir les images (28*28) en vecteurs (784)
    train_mnist = train_set.data.view(train_set.data.shape[0], -1).float().to(device)
    test_mnist = test_set.data.view(test_set.data.shape[0], -1).float().to(device) 
    labels_train_mnist = torch.nn.functional.one_hot(train_set.targets).float().to(device)
    labels_test_mnist = torch.nn.functional.one_hot(test_set.targets).float().to(device)
    nb_pixels = train_mnist.shape[1]
    neurons = 256
    epochs_pretrain = 100
    epochs_backprop = 100
    learning_rate = 0.1
    nb_layers = 4
    layers_dbn = [nb_pixels] + [neurons]*(nb_layers-1)
    nb_classes = len(train_set.class_to_idx)
    # batch_size = 0.005*len(train_mnist)
    batch_size = 512
    batch_size = int(batch_size)
    dnn_non_pretrain = DNN([784,200], nb_classes=nb_classes)
    dnn_non_pretrain.retropropagation(X=train_mnist, Y=labels_train_mnist, epochs=epochs_backprop, lr=learning_rate, batch_size=batch_size, show_progress=True, plot=True)
    error, y_hat = dnn_non_pretrain.test_DNN(test_mnist, labels_test_mnist)
    print(f"Error rate = {error}")