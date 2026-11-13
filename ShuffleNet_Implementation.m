% ShuffleNet_Implementation.m
% ShuffleNet transfer learning for Food-11.
%
% ETAPE 1 : lancer FoodClassification_database.m d'abord. Il fournit
%   trainData, valData, labelsTrain, labelsVal, augTrain, augVal.
% ETAPE 2 : decommenter UN des blocs TEST ci-dessous et lancer ce script.
%
% Test 1 : tete seule (seule la derniere couche apprend)
% Test 2 : fine-tuning complet (toutes les couches)
%
% Apres l'entrainement, evaluateFood11Metrics rapporte accuracy + macro F1.
% Soumission finale : utiliser Predictshufflenet.m sur le meilleur .mat sauvegarde.

close all

inputSize = [64 64 3];  % MUST match FoodClassification_database.m
numClasses = 11;

assert(exist("augTrain", "var") == 1, ...
    "Run FoodClassification_database.m first.");
assert(exist("labelsTrain", "var") == 1, ...
    "Run FoodClassification_database.m first.");

% Display a random sample of training images.
perm = randperm(numel(labelsTrain), 25);
imgs = cell(1, numel(perm));
for k = 1:numel(perm)
    imgs{k} = imresize(readimage(trainData, perm(k)), inputSize(1:2));
end
figure(Name="Training samples");
montage(imgs);

% ---------------------------------------------------------------------
% Construction du reseau : poids ImageNet + tete 11 classes + entree redimensionnee.
% TRANSFER LEARNING : le backbone est deja entraine sur ImageNet -> il sait
% extraire bords, textures, formes. On adapte juste la tete a nos 11 classes
% de nourriture : convergence plus rapide et meilleur score qu'un from
% scratch avec seulement ~7893 images.
% ---------------------------------------------------------------------
pretrainedNet = shufflenet;
lgraph = layerGraph(pretrainedNet);

% Remplace la derniere couche fully-connected (sortie ImageNet a 1000
% classes) par une tete a 11 sorties : les poids ImageNet de cette couche ne
% correspondent pas a nos classes. Nouvelle couche = poids aleatoires.
isFullyConnected = arrayfun(@(layer) ...
    isa(layer, "nnet.cnn.layer.FullyConnectedLayer"), lgraph.Layers);
classifierName = lgraph.Layers(find(isFullyConnected, 1, "last")).Name;
newClassifier = fullyConnectedLayer(numClasses, Name=classifierName);
lgraph = replaceLayer(lgraph, classifierName, newClassifier);

% trainnet attend une couche softmax au lieu de la vieille couche de sortie
% de classification (API legacy trainNetwork).
isClassificationOutput = arrayfun(@(layer) ...
    isa(layer, "nnet.cnn.layer.ClassificationOutputLayer"), lgraph.Layers);
classificationOutputName = lgraph.Layers( ...
    find(isClassificationOutput, 1, "last")).Name;
lgraph = replaceLayer(lgraph, classificationOutputName, ...
    softmaxLayer(Name=classificationOutputName));

% Entree redimensionnee (si 64x64 ici au lieu de 224x224) : possible parce que le reseau finit par
% un global average pooling, insensible a la taille spatiale. On conserve la
% normalisation ImageNet (mean/std) pour rester dans le domaine des poids
% pre-entraines.
inputName = pretrainedNet.InputNames{1};
pretrainedInputLayer = pretrainedNet.Layers( ...
    strcmp({pretrainedNet.Layers.Name}, inputName));
if isempty(pretrainedInputLayer.Mean)
    inputLayer = imageInputLayer(inputSize, ...
        Name=inputName, Normalization=pretrainedInputLayer.Normalization);
else
    inputLayer = imageInputLayer(inputSize, ...
        Name=inputName, ...
        Normalization=pretrainedInputLayer.Normalization, ...
        Mean=pretrainedInputLayer.Mean, ...
        StandardDeviation=pretrainedInputLayer.StandardDeviation);
end
lgraph = replaceLayer(lgraph, inputName, inputLayer);

% Inspection en tant que DLNETWORK (l'objet que trainnet entraine reellement).
netPreview = dlnetwork(lgraph);
% Nombre total de parametres apprenables : c'est la metrique de frugalite
% du projet (penalite au classement si le reseau est trop gros).
numParams = sum(cellfun(@numel, netPreview.Learnables.Value));
fprintf("Total learnable parameters: %d (%.3f M)\n", numParams, numParams/1e6);
analyzeNetwork(netPreview)  % panneau Learnables (en bas a gauche) = params par couche
% netPreview.Learnables      % decommenter pour la table des learnables en texte

%% =======================================================================
%% TEST 1 : on n'entraine que la derniere couche (head only)
% -----------------------------------------------------------------------
% Mecanisme des learn rate factors : LR effectif d'un poids = LR de base x
% facteur de la couche. Facteur 0 partout = backbone gele (aucune maj),
% facteur 10 sur la tete => LR effectif 1e-3 sur la seule couche qui apprend.
% On gele le backbone car ses poids ImageNet sont deja excellents.
%   - adam, MiniBatchSize 32, MaxEpochs 10, patience 4
%   - validation une fois par epoque
%   - ValidationPatience=4 : arret anticipe si la val loss cesse de
%     s'ameliorer pendant 4 validations -> evite l'overfitting inutile
% Resultat sauvegarde dans : ShuffleNet_Food11_test1_head.mat
% >>> Decommenter TOUT le bloc ci-dessous pour lancer ce test. <<<
%% =======================================================================
% netT1 = dlnetwork(lgraph);
% for idx = 1:height(netT1.Learnables)
%     netT1 = setLearnRateFactor(netT1, ...
%         string(netT1.Learnables.Layer(idx)), ...
%         string(netT1.Learnables.Parameter(idx)), 0);
% end
% netT1 = setLearnRateFactor(netT1, classifierName, "Weights", 10);
% netT1 = setLearnRateFactor(netT1, classifierName, "Bias", 10);
%
% miniBatchSize = 32; maxEpochs = 10;
% progressFcn = makeCourseProgressOutputFcn( ...
%     numel(labelsTrain), miniBatchSize, maxEpochs, 10);
% options = trainingOptions("adam", ...
%     InitialLearnRate=1e-4, ...
%     MaxEpochs=maxEpochs, ...
%     MiniBatchSize=miniBatchSize, ...
%     Shuffle="every-epoch", ...
%     ValidationData=augVal, ...
%     ValidationFrequency=round(numel(labelsTrain)/miniBatchSize), ...
%     ValidationPatience=4, ...
%     Verbose=false, ...
%     Plots="training-progress", ...
%     OutputFcn=progressFcn, ...
%     Metrics="accuracy");
% netT1 = trainnet(augTrain, netT1, "crossentropy", options);
% save("ShuffleNet_Food11_test1_head.mat", "netT1", "-v7.3");
% evaluateFood11Metrics(netT1, augTrainEval, labelsTrain, augVal, labelsVal);

%% =======================================================================
%% TEST 2 : fine-tuning complet (toutes les couches)
% -----------------------------------------------------------------------
% Toutes les couches apprennent, avec un LR tres faible pour ne pas
% detruire les poids ImageNet :
%   - LR de base 1e-5, facteur 10 sur la tete => tete a 1e-4
%   - adam, MiniBatchSize 32, MaxEpochs 8, patience 3
% Resultat sauvegarde dans : ShuffleNet_Food11_test2_full.mat
% >>> Decommenter TOUT le bloc ci-dessous pour lancer ce test. <<<
%% =======================================================================
% % Depart des poids ImageNet (defaut) :
% netT2 = dlnetwork(lgraph);
% % OU depart des poids du Test 1 (decommenter les 2 lignes suivantes et
% % commenter la ligne au-dessus) :
% % S = load("ShuffleNet_Food11_test1_head.mat"); netT2 = S.netT1;
%
% for idx = 1:height(netT2.Learnables)
%     netT2 = setLearnRateFactor(netT2, ...
%         string(netT2.Learnables.Layer(idx)), ...
%         string(netT2.Learnables.Parameter(idx)), 1);
% end
% netT2 = setLearnRateFactor(netT2, classifierName, "Weights", 10);
% netT2 = setLearnRateFactor(netT2, classifierName, "Bias", 10);
%
% miniBatchSize = 32; maxEpochs = 8;
% progressFcn = makeCourseProgressOutputFcn( ...
%     numel(labelsTrain), miniBatchSize, maxEpochs, 10);
% options = trainingOptions("adam", ...
%     InitialLearnRate=1e-5, ...
%     MaxEpochs=maxEpochs, ...
%     MiniBatchSize=miniBatchSize, ...
%     Shuffle="every-epoch", ...
%     ValidationData=augVal, ...
%     ValidationFrequency=round(numel(labelsTrain)/miniBatchSize), ...
%     ValidationPatience=3, ...
%     Verbose=false, ...
%     Plots="training-progress", ...
%     OutputFcn=progressFcn, ...
%     Metrics="accuracy");
% netT2 = trainnet(augTrain, netT2, "crossentropy", options);
% save("ShuffleNet_Food11_test2_full.mat", "netT2", "-v7.3");
% evaluateFood11Metrics(netT2, augTrainEval, labelsTrain, augVal, labelsVal);
