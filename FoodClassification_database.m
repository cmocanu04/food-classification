clearvars
close all

% Indique le dossier contenant les images Food-11.
root = fullfile(fileparts(mfilename("fullpath")), "food-11", "food-11");

% Charge les images d'entraînement et utilise les noms des dossiers
% comme labels des 11 classes.
data = imageDatastore(fullfile(root, "train"), ...
    IncludeSubfolders=true, LabelSource="foldernames");

% Divise chaque classe en 80 % pour l'entraînement et 20 % pour le test.
rng default
[trainData, testData] = splitEachLabel(data, 0.8, "randomized");

% Convertit les images et récupère leurs labels.
labelsTrain = trainData.Labels;
imgDataTrain = zeros(32, 32, 3, numel(trainData.Files), "single");
for i = 1:numel(trainData.Files)
    image = readimage(trainData, i);
    imgDataTrain(:, :, :, i) = im2single(imresize(image(:, :, 1:3), [32 32]));
end

labelsTest = testData.Labels;
imgDataTest = zeros(32, 32, 3, numel(testData.Files), "single");
for i = 1:numel(testData.Files)
    image = readimage(testData, i);
    imgDataTest(:, :, :, i) = im2single(imresize(image(:, :, 1:3), [32 32]));
end
