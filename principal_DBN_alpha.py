from principal_RBM_alpha import *


class DBN():
    def __init__(self, layers):
        """Initialise un Deep Belief Network (DBN).
        Args:
            layers (list): Liste des nombres de neurons dans chaque layer
        """
        self.layers = layers 
        self.nb_couche = len(self.layers)
        self.list_RBM = []
        for i in range(self.nb_couche - 1):
            rbm = RBM( 
                p = layers[i], 
                q = layers[i+1],
            )
            self.list_RBM.append(rbm)
        
    def train_DBN(self, x, epochs, lr, batch_size = None,  plot=False, show_progress=False, layers=None):
        """Entraîne un DBN.

        Args:
            (x, epochs, lr, batch_size, plot, show_progress) : Voir train_RBM
            *epochs (list): Nombre d'itérations pour chaque couche
            layers (int): Nombre de couches à entraîner (si None : toutes les couches)
        """
        if layers is None: 
            layers = self.nb_couche
        
        if batch_size is None :
            batch_size = int(x.shape[0]*0.2)

        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x).to(device=device, dtype=torch.double)  
        else:
            x = x.to(device)
        if len(epochs) != self.nb_couche - 1: 
            epochs = [epochs[0]] * (self.nb_couche - 1)
        if show_progress:
            rbm_iterator = tqdm(self.list_RBM, desc="Training DBN", unit="RBM")
        else:
            rbm_iterator = self.list_RBM
        iter = 0
        for rbm in rbm_iterator:
            iter += 1
            rbm.train_RBM(
                x, epochs=epochs[iter-1], lr=lr, batch_size=batch_size, plot=plot, show_progress=False
            )
            x = rbm.entree_sortie_RBM(x)


    def generer_image_DBN(self, iterations_gibbs, nb_images, show=False):
        """
        Generate images using the Deep Belief Network (DBN) via Gibbs sampling.

        Args:
            iterations_gibbs (int): Number of Gibbs sampling iterations.
            nb_images (int): Number of images to generate.
            show (bool): If True, display the generated images using matplotlib.

        Returns:
            list: A list of generated images, each represented as a 2D NumPy array.
        """
        # Initialize the visible layer
        v = torch.randint(0, 2, (nb_images, 1, self.layers[0]), dtype=torch.double, device=device)

        # Gibbs sampling
        for _ in range(iterations_gibbs):
            for rbm in self.list_RBM:
                p_h_v = rbm.entree_sortie_RBM(v)  
                h = torch.bernoulli(p_h_v) 
                v = h  

            for rbm in self.list_RBM[::-1]:
                p_v_h = rbm.sortie_entree_RBM(v)  
                v = torch.bernoulli(p_v_h) 

        imgs = []
        for img in range(nb_images):
            X = v[img].cpu().numpy().reshape(20, 16)  # Reshape to 20x16 and convert to NumPy
            imgs.append(X)
            if show:
                plt.figure()
                plt.imshow(X, cmap='Greys')
                plt.axis('off')  
                plt.show()

        return imgs

