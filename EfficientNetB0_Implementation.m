% EfficientNetB0_Implementation.m
% EfficientNet-B0 transfer learning for Food-11.
%
% STEP 1: run FoodClassification_database.m first. It provides
%   trainData, valData, labelsTrain, labelsVal, augTrain, augVal.
% STEP 2: decomment ONE of the TEST blocks below and run this script.
%
% Test 1: head only (seule la derniere couche apprend)
% Test 2: fine-tuning partiel (le dernier bloc MBConv + la tete)
% Test 3: fine-tuning complet (toutes les couches)
%
% After training, evaluateFood11Metrics reports accuracy + macro F1.
% Final submission: use predict_test_set.m on the best saved .mat.

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
% Build the network: ImageNet weights, 11-class head, resized input.
% ---------------------------------------------------------------------
pretrainedNet = efficientnetb0;
lgraph = layerGraph(pretrainedNet);

% Replace the final fully connected layer with an 11-class head.
isFullyConnected = arrayfun(@(layer) ...
    isa(layer, "nnet.cnn.layer.FullyConnectedLayer"), lgraph.Layers);
classifierName = lgraph.Layers(find(isFullyConnected, 1, "last")).Name;
newClassifier = fullyConnectedLayer(numClasses, Name=classifierName);
lgraph = replaceLayer(lgraph, classifierName, newClassifier);

% trainnet expects softmax instead of the legacy classification output.
isClassificationOutput = arrayfun(@(layer) ...
    isa(layer, "nnet.cnn.layer.ClassificationOutputLayer"), lgraph.Layers);
classificationOutputName = lgraph.Layers( ...
    find(isClassificationOutput, 1, "last")).Name;
lgraph = replaceLayer(lgraph, classificationOutputName, ...
    softmaxLayer(Name=classificationOutputName));

% Resized input layer (possible thanks to global average pooling).
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

% analyzeNetwork(lgraph)  % decomment to inspect the architecture

%% =======================================================================
%% TEST 1 : on n'entraine que la derniere couche (head only)
% -----------------------------------------------------------------------
% Toutes les couches pre-entrainees sont gelees (learn rate factor 0).
% Seule la nouvelle couche fully-connected a 11 classes apprend :
%   - LR de base 1e-4, facteur 10 sur la tete => LR effectif 1e-3
%   - adam, MiniBatchSize 32, MaxEpochs 10, patience 4
%   - validation une fois par epoque
% Resultat sauvegarde dans : EfficientNetB0_Food11_test1_head.mat
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
% save("EfficientNetB0_Food11_test1_head.mat", "netT1", "-v7.3");
% evaluateFood11Metrics(netT1, augVal, labelsVal);

%% =======================================================================
%% TEST 2 : fine-tuning partiel (le dernier bloc MBConv + la tete)
% -----------------------------------------------------------------------
% On degele les K dernieres couches du backbone (K = 20, soit environ le
% dernier bloc MBConv + la tete) en plus de la couche de classification :
%   - LR 1e-4 uniforme sur les couches degelees (facteur 1 partout)
%   - adam, MiniBatchSize 32, MaxEpochs 8, patience 3
% Verifier la frontiere des K couches avec analyzeNetwork si besoin.
% Resultat sauvegarde dans : EfficientNetB0_Food11_test2_partial.mat
% >>> Decommenter TOUT le bloc ci-dessous pour lancer ce test. <<<
%% =======================================================================
% % Depart des poids ImageNet (defaut) :
% netT2 = dlnetwork(lgraph);
% % OU depart des poids du Test 1 (decommenter les 2 lignes suivantes et
% % commenter la ligne au-dessus) :
% % S = load("EfficientNetB0_Food11_test1_head.mat"); netT2 = S.netT1;
%
% K = 20;  % nombre de couches degelees a partir de la fin
% layersT2 = netT2.Layers;
% unfreezeFrom = numel(layersT2) - K + 1;
% for idx = 1:height(netT2.Learnables)
%     lyr = netT2.Learnables.Layer(idx);
%     pos = find(strcmp({layersT2.Name}, char(lyr)), 1);
%     factor = double(pos >= unfreezeFrom);
%     netT2 = setLearnRateFactor(netT2, lyr, ...
%         netT2.Learnables.Parameter(idx), factor);
% end
%
% miniBatchSize = 32; maxEpochs = 8;
% progressFcn = makeCourseProgressOutputFcn( ...
%     numel(labelsTrain), miniBatchSize, maxEpochs, 10);
% options = trainingOptions("adam", ...
%     InitialLearnRate=1e-4, ...
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
% save("EfficientNetB0_Food11_test2_partial.mat", "netT2", "-v7.3");
% evaluateFood11Metrics(netT2, augVal, labelsVal);

%% =======================================================================
%% TEST 3 : fine-tuning complet (toutes les couches)
% -----------------------------------------------------------------------
% Toutes les couches apprennent, avec un LR tres faible pour ne pas
% detruire les poids ImageNet :
%   - LR de base 1e-5, facteur 10 sur la tete => tete a 1e-4
%   - adam, MiniBatchSize 32, MaxEpochs 8, patience 3
% Resultat sauvegarde dans : EfficientNetB0_Food11_test3_full.mat
% >>> Decommenter TOUT le bloc ci-dessous pour lancer ce test. <<<
%% =======================================================================
% % Depart des poids ImageNet (defaut) :
% netT3 = dlnetwork(lgraph);
% % OU depart des poids du Test 1 (decommenter les 2 lignes suivantes et
% % commenter la ligne au-dessus) :
% % S = load("EfficientNetB0_Food11_test1_head.mat"); netT3 = S.netT1;
%
% for idx = 1:height(netT3.Learnables)
%     netT3 = setLearnRateFactor(netT3, ...
%         string(netT3.Learnables.Layer(idx)), ...
%         string(netT3.Learnables.Parameter(idx)), 1);
% end
% netT3 = setLearnRateFactor(netT3, classifierName, "Weights", 10);
% netT3 = setLearnRateFactor(netT3, classifierName, "Bias", 10);
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
% netT3 = trainnet(augTrain, netT3, "crossentropy", options);
% save("EfficientNetB0_Food11_test3_full.mat", "netT3", "-v7.3");
% evaluateFood11Metrics(netT3, augVal, labelsVal);
