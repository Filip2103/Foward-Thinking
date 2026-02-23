#Ova klasa predstavlja funkciju neuronske mreze koja implementira kompoziciju skrivenih slojeva i finalnog klasifikatora, pri cemu lista hidden_dims odredjuje dubinu mreze i omogucava inkrementalno dodavanje slojeva u Forward Thinking algoritmu.

import torch
import torch.nn as nn

class LayerwiseMLP(nn.Module):

    """
    Konstruktor mreze
    input_dim - dimenezija ulaza MNIST = 784
    hidden_dims - lista skrivenih slojeva

    output_dim - broj klasa = 10

    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        #Kreiranje liste slojeva prev_dim - dimenzija prethodnog sloja
        layers = []
        prev_dim = input_dim
        """
        Petlja za skrivene slojeve
        Svaki Linear + ReLU par je Ci(k) tj funkcija koja transformise dataset
        """
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        """
        Ovo spaja sve delove u 1 f-ju
        Ovo je kompozicija funkcija iz rada
        """
       
        self.hidden_layers = nn.Sequential(*layers)

        # Ovo je output sloj odnosno Cf: Xn -> Y. Prima poslednji hidden output i vraca logits za 10 klasa
        self.output_layer = nn.Linear(prev_dim, output_dim)


    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
