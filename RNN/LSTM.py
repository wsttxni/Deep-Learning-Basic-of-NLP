import numpy as np
from rnn_utlis import *

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"] # update gate weight
    bi = parameters["bi"]
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    concat = np.concatenate((a_prev, xt), axis=0)

    # LSTM cell
    ft = sigmoid(np.dot(Wf, concat) + bf)        # whether to forget
    it = sigmoid(np.dot(Wi, concat) + bi)        # whether to update
    cct =  np.tanh(np.dot(Wc, concat) + bc)      # candidate value
    c_next = it*cct + ft*c_prev                  # cell state
    ot = sigmoid(np.dot(Wo, concat) + bo)        # whether to output
    a_next = ot*np.tanh(c_next)                  # hidden state
    
    yt_pred = softmax(np.dot(Wy, a_next) + by)   # activate a to get y

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

def lstm_forward(x, a0, parameters):
    caches = []
    
    Wy = parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters) # forward and update
        a[:,:,t] = a_next
        c[:,:,t] = c_next
        y[:,:,t] = yt
        caches.append(cache) # save
        
    caches = (caches, x)

    return a, y, c, caches