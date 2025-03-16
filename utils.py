import numpy as np
import scipy.io



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
        
    final_data = np.vstack(final_data)
    final_data = np.resize(final_data, (final_data.shape[0], 1, final_data.shape[1]))

    return final_data

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(z):
    return np.log(z / (1 - z))