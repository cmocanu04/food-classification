% LeNet5_Implementation.m
% LeNet-5 adapte a la classification Food-11 (reseau entraine from scratch,
% architecture originale 1998 : tanh + average pooling). Baseline de
% reference pour comparer avec le transfer learning.
%
% STEP 1: run FoodClassification_database.m first. It provides
%   trainData, valData, labelsTrain, labelsVal, augTrain, augVal.
% STEP 2: run this script (no test blocks: single training configuration).

close all

inputSize = [64 64 3];  % MUST match FoodClassification_database.m

assert(exist("augTrain", "var") == 1, ...
    "Run FoodClassification_database.m first.");

% Display a random sample of training images.
perm = randperm(numel(labelsTrain), 25);
imgs = cell(1, numel(perm));
for k = 1:numel(perm)
    imgs{k} = imresize(readimage(trainData, perm(k)), inputSize(1:2));
end
figure(Name="Training samples");
montage(imgs);

% LeNet-5 utilise deux blocs convolution + sous-echantillonnage,
% puis trois couches entierement connectees.
layers = [
    imageInputLayer([64 64 3], Name="input")
    convolution2dLayer(5, 6, Name="conv1")
    tanhLayer(Name="tanh1")
    averagePooling2dLayer(2, Stride=2, Name="pool1")
    convolution2dLayer(5, 16, Name="conv2")
    tanhLayer(Name="tanh2")
    averagePooling2dLayer(2, Stride=2, Name="pool2")
    fullyConnectedLayer(120, Name="fullyconnected1")
    tanhLayer(Name="tanh3")
    fullyConnectedLayer(84, Name="fullyconnected2")
    tanhLayer(Name="tanh4")
    fullyConnectedLayer(11, Name="fullyconnected3")
    softmaxLayer(Name="softmax")];

% Inspect the network architecture.
analyzeNetwork(layers)

miniBatchSize = 128;
maxEpochs = 20;
smoothingWindow = 10;
progressFcn = makeCourseProgressOutputFcn( ...
    numel(labelsTrain), miniBatchSize, maxEpochs, smoothingWindow);

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

net = trainnet(augTrain, layers, "crossentropy", options);

% Save the trained network and evaluate on train and validation sets.
save("LeNet5_Food11.mat", "net", "-v7.3");
evaluateFood11Metrics(net, augVal, labelsVal);
