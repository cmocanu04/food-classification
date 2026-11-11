% MobileNetV2 transfer learning for Food-11.
% The pretrained classifier head is adapted to 11 food classes.

close all

inputSize = [64 64 3];
numClasses = 11;

% Run FoodClassification_database.m first. It provides the 64-by-64 image
% arrays and labels used below.
assert(isequal(size(imgDataTrain, 1:3), inputSize), ...
    "Run FoodClassification_database.m with 64-by-64 preprocessing first.");

% Display a random sample of training images.
perm = randperm(numel(labelsTrain), min(25, numel(labelsTrain)));
subset = imgDataTrain(:, :, :, perm);
figure(Name="Training samples");
montage(subset);

% Load ImageNet weights through the MobileNetV2 support package.
pretrainedNet = mobilenetv2;
lgraph = layerGraph(pretrainedNet);

% MobileNetV2 uses a final fully connected layer for its classifier head.
% Do not replace Conv_1: it must continue to output 1280 feature channels.
isFullyConnected = arrayfun(@(layer) ...
    isa(layer, "nnet.cnn.layer.FullyConnectedLayer"), lgraph.Layers);
classifierName = lgraph.Layers(find(isFullyConnected, 1, "last")).Name;
newClassifier = fullyConnectedLayer(numClasses, ...
    Name=classifierName, ...
    WeightLearnRateFactor=10, ...
    BiasLearnRateFactor=10);
lgraph = replaceLayer(lgraph, classifierName, newClassifier);

% trainnet expects a differentiable network output rather than the legacy
% classification output layer.
isClassificationOutput = arrayfun(@(layer) ...
    isa(layer, "nnet.cnn.layer.ClassificationOutputLayer"), lgraph.Layers);
classificationOutputName = lgraph.Layers( ...
    find(isClassificationOutput, 1, "last")).Name;
lgraph = replaceLayer(lgraph, classificationOutputName, ...
    softmaxLayer(Name=classificationOutputName));

% MobileNetV2 is normally defined for 224-by-224 images. Its global
% pooling head allows the spatial input size to be reduced to 64-by-64.
inputName = pretrainedNet.InputNames{1};
pretrainedInputLayer = pretrainedNet.Layers( ...
    strcmp({pretrainedNet.Layers.Name}, inputName));
inputLayer = imageInputLayer(inputSize, ...
    Name=inputName, ...
    Normalization=pretrainedInputLayer.Normalization, ...
    Mean=pretrainedInputLayer.Mean, ...
    StandardDeviation=pretrainedInputLayer.StandardDeviation);
lgraph = replaceLayer(lgraph, inputName, inputLayer);
net = dlnetwork(lgraph);

% Freeze every pretrained parameter except the new Food-11 classifier.
for idx = 1:height(net.Learnables)
    net = setLearnRateFactor(net, ...
        string(net.Learnables.Layer(idx)), ...
        string(net.Learnables.Parameter(idx)), 0);
end
net = setLearnRateFactor(net, classifierName, "Weights", 10);
net = setLearnRateFactor(net, classifierName, "Bias", 10);

% Inspect the adapted network before training.
analyzeNetwork(net)

miniBatchSize = 32;
maxEpochs = 12;
validationFrequency = 50;
smoothingWindow = 10;
progressFcn = makeCourseProgressOutputFcn( ...
    numel(labelsTrain), miniBatchSize, maxEpochs, smoothingWindow);

options = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData={imgDataTest, labelsTest}, ...
    ValidationFrequency=validationFrequency, ...
    Verbose=false, ...
    Plots="training-progress", ...
    OutputFcn=progressFcn, ...
    Metrics="accuracy");

net = trainnet(imgDataTrain, labelsTrain, net, "crossentropy", options);

% Save the trained network and its learned weights.
modelFile = fullfile(fileparts(mfilename("fullpath")), ...
    "MobileNetV2_Food11.mat");
save(modelFile, "net", "-v7.3");

% Evaluate the final network on the held-out split.
trainScores = minibatchpredict(net, imgDataTrain);
trainLabelsPred = scores2label(trainScores, categories(labelsTrain));
trainAccuracy = mean(trainLabelsPred == labelsTrain);
fprintf("Train-set accuracy: %.2f%%\n", 100 * trainAccuracy);

testScores = minibatchpredict(net, imgDataTest);
testLabelsPred = scores2label(testScores, categories(labelsTest));
testAccuracy = mean(testLabelsPred == labelsTest);
fprintf("Test-set accuracy: %.2f%%\n", 100 * testAccuracy);
