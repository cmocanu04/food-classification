% LeNet-5 adapte a la classification Food-11.
% Charge les images et leurs labels.

close all

% Display a random sample of training images.
perm = randperm(numel(labelsTrain), 25);
subset = imgDataTrain(:, :, :, perm);
montage(subset)

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

% The official test set is supplied through ValidationData so its
% held-out accuracy and loss can be plotted during training. The custom plot
% labels these curves "Test set".
% Train the network using the R2024a dlnetwork workflow.
miniBatchSize = 128;
% Keep MATLAB's progress window for device/status information, and
% add a companion plot with smoothed curves and epoch bands.
maxEpochs = 20;
smoothingWindow = 10;
progressFcn = makeCourseProgressOutputFcn( ...
    numel(labelsTrain), miniBatchSize, maxEpochs, smoothingWindow);

options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData={imgDataTest, labelsTest}, ...
    ValidationFrequency=50, ...
    Verbose=false, ...
    Plots="training-progress", ...
    OutputFcn=progressFcn, ...
    Metrics="accuracy");

net = trainnet(imgDataTrain, labelsTrain, layers, "crossentropy", options);

% Evaluate the final network on the complete training and test sets.
trainScores = minibatchpredict(net, imgDataTrain);
trainLabelsPred = scores2label(trainScores, categories(labelsTrain));
trainAccuracy = mean(trainLabelsPred == labelsTrain);
fprintf("Train-set accuracy: %.2f%%\n", 100 * trainAccuracy);

testScores = minibatchpredict(net, imgDataTest);
testLabelsPred = scores2label(testScores, categories(labelsTrain));
testAccuracy = mean(testLabelsPred == labelsTest);
fprintf("Test-set accuracy: %.2f%%\n", 100 * testAccuracy);
