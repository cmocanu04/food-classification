% FoodClassification_database.m
% Data loading for Food-11 via datastores (works at any resolution, with
% on-the-fly data augmentation for the train split).
%
% Produces in the workspace:
%   trainData / valData : underlying imageDatastores (80/20 split, FIXED seed)
%   labelsTrain / labelsVal
%   augTrain : augmentedImageDatastore (train, resize + random augmentation)
%   augVal   : augmentedImageDatastore (val, resize ONLY, never augmented)
%
% IMPORTANT: inputSize must match the inputSize in the *_Implementation.m
% script you run afterwards.
%   - [64 64 3]  : fast experiments (current setting)
%   - [224 224 3]: final training (native ImageNet resolution)

clearvars
close all

rng(42)  % FIXED seed: identical train/val split for every network (fair comparison)

inputSize = [64 64 3];  % <<< set [224 224 3] for the final training

root = fullfile(fileparts(mfilename("fullpath")), "food-11", "food-11");

data = imageDatastore(fullfile(root, "train"), ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

% 80 % train / 20 % validation, stratified per class.
[trainData, valData] = splitEachLabel(data, 0.8, "randomized");
labelsTrain = trainData.Labels;
labelsVal = valData.Labels;

% ---- Data augmentation: TRAIN split only ----
% - horizontal flip : a mirrored plate is still the same dish
% - small rotation / translation / scale : camera angle and framing
% - NO vertical flip (upside-down plates are unnatural)
% - NO color jitter (imageDataAugmenter has none; color is informative)
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-15 15], ...
    RandXTranslation=[-10 10], ...
    RandYTranslation=[-10 10], ...
    RandXScale=[0.9 1.1], ...
    RandYScale=[0.9 1.1]);

augTrain = augmentedImageDatastore(inputSize(1:2), trainData, ...
    DataAugmentation=augmenter);

% Validation: resize only. Never augment validation data.
augVal = augmentedImageDatastore(inputSize(1:2), valData);

fprintf("Train: %d images | Val: %d images | input %dx%d\n", ...
    numel(labelsTrain), numel(labelsVal), inputSize(1), inputSize(2));
