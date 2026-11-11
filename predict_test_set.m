% PREDICT_TEST_SET  Generate the JSON submission file for the official
% Food-11 test set (food-11/food-11/test, 3097 unlabeled images).
%
% Model-agnostic: works with any network trained with trainnet and saved
% as a dlnetwork named "net" inside a .mat file (MobileNetV2, EfficientNetB0,
% ResNet, LeNet, ...). The input size is auto-detected from the network.
%
% Preprocessing (imresize + im2single) replicates FoodClassification_database.m
% exactly, so the network sees images in the same format as during training.
%
% The class-name order MUST match the order of categories(labelsTrain) used
% at training time. categories() returns labels sorted alphabetically, and
% the train directory names are listed alphabetically below, so both match.
% WARNING: if you ever train with a non-alphabetical class order, update
% classNames manually.
%
% Usage: set modelFile and outputJson below, then run this script.

close all

% ----------------------------- User settings ------------------------------
modelFile  = "EfficientNetB0_Food11.mat";  % .mat file containing the net
outputJson = "predictions.json";           % submission file to upload
% ---------------------------------------------------------------------------

thisDir = fileparts(mfilename("fullpath"));
root = fullfile(thisDir, "food-11", "food-11");

% Class names = train folder names, lowercase, alphabetical. This ordering
% matches categories(labelsTrain) from the training scripts.
d = dir(fullfile(root, "train"));
d = d([d.isdir] & ~ismember({d.name}, {".", ".."}));
classNames = lower(sort({d.name}));
numClasses = numel(classNames);

% Load the trained network.
S = load(modelFile);
if isfield(S, "net")
    net = S.net;
else
    fn = fieldnames(S); % fall back: first variable in the file
    net = S.(fn{1});
end

% Input size: auto-detect from the input layer, or set manually.
try
    inputSize = net.Layers(1).InputSize;
catch
    warning("Could not auto-detect input size. Assuming [224 224 3]. " + ...
        "VERIFY this matches the net or predictions will be wrong!");
    inputSize = [224 224 3]; % <-- set manually if auto-detection fails
end
fprintf("Model: %s | input %dx%d | %d classes\n", ...
    modelFile, inputSize(1), inputSize(2), numClasses);

% Read the official test images, preprocess exactly like training.
testDir = fullfile(root, "test");
testDS = imageDatastore(testDir, IncludeSubfolders=false);
files = testDS.Files;
numTest = numel(files);
[~, names, ~] = cellfun(@fileparts, files, UniformOutput=false);

imgTest = zeros(inputSize(1), inputSize(2), 3, numTest, "single");
for i = 1:numTest
    img = readimage(testDS, i);
    if size(img, 3) == 1 % guard against grayscale images
        img = repmat(img, 1, 1, 3);
    end
    imgTest(:, :, :, i) = im2single(imresize(img(:, :, 1:3), inputSize(1:2)));
end

% Predict in chunks to keep memory usage low.
chunkSize = 250;
predIdx = zeros(numTest, 1);
for s = 1:chunkSize:numTest
    e = min(s + chunkSize - 1, numTest);
    scores = minibatchpredict(net, imgTest(:, :, :, s:e));
    [~, idx] = max(scores, [], 2);
    predIdx(s:e) = idx;
end

predClasses = classNames(predIdx);

% Sanity check: predicted class distribution (should roughly mirror the
% train-set proportions: dessert/soup largest, rice smallest).
fprintf("\nPredicted class distribution on the test set:\n");
for c = 1:numClasses
    fprintf("  %-16s %4d\n", classNames{c}, sum(predIdx == c));
end

% Write the JSON file in the exact format required by the evaluation server:
% { "imageName": "classname", ... } (image name without extension).
fid = fopen(outputJson, "w");
fprintf(fid, "{\n");
for i = 1:numTest
    if i == numTest, comma = ""; else, comma = ","; end
    fprintf(fid, '    "%s": "%s"%s\n', names{i}, predClasses{i}, comma);
end
fprintf(fid, "}\n");
fclose(fid);

fprintf("\nWrote %d predictions to %s\n", numTest, outputJson);
