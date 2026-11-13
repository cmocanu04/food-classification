% LeNet5_Implementation.m
% LeNet-5 adapte a la classification Food-11 (reseau entraine from scratch,
% architecture originale 1998 : tanh + average pooling). Baseline de
% reference pour comparer avec le transfer learning.
%
% ETAPE 1 : lancer FoodClassification_database.m d'abord. Il fournit
%   trainData, valData, labelsTrain, labelsVal, augTrain, augVal.
% ETAPE 2 : lancer ce script (pas de blocs de test : configuration unique).

close all

inputSize = [64 64 3];  % MUST match FoodClassification_database.m

assert(exist("augTrain", "var") == 1, ...
    "Run FoodClassification_database.m first.");

% Echantillon aleatoire du train : sanity check des labels avant d'entrainer.
perm = randperm(numel(labelsTrain), 25);
imgs = cell(1, numel(perm));
for k = 1:numel(perm)
    imgs{k} = imresize(readimage(trainData, perm(k)), inputSize(1:2));
end
figure(Name="Training samples");
montage(imgs);

% LeNet-5 : deux blocs conv + pooling, puis trois couches fully-connected.
% Flux pour une entree 64x64x3 :
%   -> conv1 (6 cartes)  -> tanh -> pool -> conv2 (16 cartes) -> tanh -> pool
%   -> fc 120 -> fc 84 -> fc 11 (un score par classe) -> softmax (probabilites)
layers = [
    imageInputLayer([64 64 3], Name="input")
    % conv1 : 6 filtres 5x5 -> detecte bords et textures simples
    convolution2dLayer(5, 6, Name="conv1")
    tanhLayer(Name="tanh1")
    % average pooling 2x2 : divise les dimensions par 2, reduit le calcul
    averagePooling2dLayer(2, Stride=2, Name="pool1")
    % conv2 : 16 filtres -> motifs plus complexes (combinaisons de conv1)
    convolution2dLayer(5, 16, Name="conv2")
    tanhLayer(Name="tanh2")
    averagePooling2dLayer(2, Stride=2, Name="pool2")
    % fully-connected : combinent tout et produisent les 11 scores de classes
    fullyConnectedLayer(120, Name="fullyconnected1")
    tanhLayer(Name="tanh3")
    fullyConnectedLayer(84, Name="fullyconnected2")
    tanhLayer(Name="tanh4")
    fullyConnectedLayer(11, Name="fullyconnected3")
    % softmax : 11 scores -> 11 probabilites (somme = 1)
    softmaxLayer(Name="softmax")];

% Inspection de l'architecture (verifie les dimensions de chaque couche).
analyzeNetwork(layers)

% Hyperparametres :
%   miniBatchSize = 128 -> 1 iteration = 128 images + 1 maj des poids
%     (avec ~7893 images : 62 iterations par epoque)
%   maxEpochs = 20 -> 62*20 = 1240 iterations au total
%   smoothingWindow = 10 -> lissage des courbes du graphique temps reel
miniBatchSize = 128;
maxEpochs = 20;
smoothingWindow = 10;
progressFcn = makeCourseProgressOutputFcn( ...
    numel(labelsTrain), miniBatchSize, maxEpochs, smoothingWindow);

% Options d'entrainement :
%   adam + LR 1e-3    : optimiseur robuste, LR standard pour Adam
%   Shuffle           : re-melange les images a chaque epoque
%   ValidationFrequency = 62 -> validation exactement 1 fois par epoque,
%     sans maj des poids (mesure de generalisation seulement)
%   Metrics="accuracy" : suivre l'accuracy en plus de la loss
options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData=augVal, ...
    ValidationFrequency=round(numel(labelsTrain)/miniBatchSize), ...
    Verbose=false, ...
    Plots="training-progress", ...
    OutputFcn=progressFcn, ...
    Metrics="accuracy");

% Entrainement : chaque iteration = lecture d'un batch (augmente), forward,
% cross-entropy, backward, maj Adam. La cross-entropie penalise surtout les
% erreurs confiantes -> pousse vers des probabilites calibrees.
net = trainnet(augTrain, layers, "crossentropy", options);

% Sauvegarde + evaluation train vs val (diagnostic sur/sous-apprentissage).
save("LeNet5_Food11.mat", "net", "-v7.3");
evaluateFood11Metrics(net, augTrainEval, labelsTrain, augVal, labelsVal);
