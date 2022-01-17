# HSP
Module Hardware CUDA
# TP Hardware for Signal Processing

# Prise en main de CUDA
# TP1 : Prise en main de Cuda : Multiplication de matrices
on va faire:

Création d'une matrice sur CPU

Affichage d'une matrice sur CPU

Addition de deux matrices sur CPU

Addition de deux matrices sur GPU

Multiplication de deux matrices NxN sur CPU

Multiplication de deux matrices NxN sur GPU

Compléxité et temps de calcul

le resultat quand n et p = 2, quand le numbre de n et p est petit, on trouve que la calcul de viteese dans CPU est plus rapide que dans le GPU:

![image](https://user-images.githubusercontent.com/66156908/149814358-8b88a869-9ef5-4170-8ffa-c470b2c51709.png)

le resultat quand n et p = 500, quand le numbre de n et p est grand, on trouve que la calcul de viteese dans GPU est plus rapide que dans le CPU:

![image](https://user-images.githubusercontent.com/66156908/149815139-7dcb6bc5-04c5-4695-8930-ac93fd81d58a.png)


# TP2 : Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

on va faire:

L'architecture du réseau LeNet-5 est composé de plusieurs couches :

    Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST

    Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.

    Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.
