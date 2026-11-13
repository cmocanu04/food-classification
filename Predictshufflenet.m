
% PREDICT_TEST_SET.m
% Final ShuffleNet Test 3 predictions on official Food-11 test set.

close all;
clc;

%% 1. PARAMETERS

modelFile = "ShuffleNet_Food11_test3_full.mat";

testDir = fullfile("food-11", "food-11", "test");

if isfolder(fullfile("food-11", "train"))
    trainDir = fullfile("food-11", "train");
elseif isfolder(fullfile("food-11", "food-11", "train"))
    trainDir = fullfile("food-11", "food-11", "train");
else
    error("Training directory not found.");
end

outputJson = "predictions_shufflenet_final.json";

inputSize = [224 224 3];
miniBatchSize = 250;

%% 2. CHECK FILES AND DIRECTORIES

assert(isfile(modelFile), ...
    "Model file not found: %s", modelFile);

assert(isfolder(testDir), ...
    "Test directory not found: %s", testDir);

assert(isfolder(trainDir), ...
    "Training directory not found: %s", trainDir);

fprintf("Training directory: %s\n", trainDir);
fprintf("Test directory: %s\n", testDir);

%% 3. LOAD TRAINED MODEL

fprintf("\nLoading ShuffleNet Test 3 model...\n");

S = load(modelFile);

assert(isfield(S, "netT3"), ...
    "Variable netT3 not found in model file.");

net = S.netT3;

%% 4. GET CLASS NAMES IN TRAINING ORDER

fprintf("\nRetrieving class order...\n");

% Reconstruct the class order from the training dataset.
trainingDS = imageDatastore( ...
    trainDir, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

classNames = categories(trainingDS.Labels);

fprintf("\nClasses in training order:\n");
disp(classNames);

assert(numel(classNames) == 11, ...
    "Expected 11 classes, found %d.", numel(classNames));

%% 5. LOAD TEST DATASET

fprintf("\nLoading test dataset...\n");

testDS = imageDatastore( ...
    testDir, ...
    IncludeSubfolders=false);

files = testDS.Files;
numTest = numel(files);

fprintf("Number of test images: %d\n", numTest);

assert(numTest == 3097, ...
    "Expected 3097 test images, found %d.", numTest);

%% 6. EXTRACT IMAGE IDENTIFIERS

names = cell(numTest, 1);

for i = 1:numTest
    [~, name, ~] = fileparts(files{i});
    names{i} = name;
end

assert(numel(unique(names)) == numTest, ...
    "Duplicate image identifiers detected.");

%% 7. PREPARE TEST IMAGES

fprintf("\nPreparing test images...\n");

imgTest = zeros( ...
    inputSize(1), ...
    inputSize(2), ...
    inputSize(3), ...
    numTest, ...
    "uint8");

for i = 1:numTest

    img = readimage(testDS, i);

    % Convert grayscale images to RGB.
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);

    % Keep only the first three channels if necessary.
    elseif size(img, 3) > 3
        img = img(:, :, 1:3);
    end

    % Resize while preserving pixel type.
    img = imresize(img, inputSize(1:2));

    imgTest(:, :, :, i) = img;

    if mod(i, 500) == 0
        fprintf("Prepared %d / %d images\n", i, numTest);
    end

end

fprintf("Image preparation completed.\n");

%% 8. RUN PREDICTIONS

fprintf("\nStarting predictions...\n");

predIdx = zeros(numTest, 1);

for startIdx = 1:miniBatchSize:numTest

    endIdx = min(startIdx + miniBatchSize - 1, numTest);

    batch = imgTest(:, :, :, startIdx:endIdx);

    scores = minibatchpredict(net, batch);

    [~, batchPredIdx] = max(scores, [], 2);

    predIdx(startIdx:endIdx) = batchPredIdx;

    fprintf("Predicted %d / %d images\n", endIdx, numTest);

end

%% 9. CONVERT INDICES TO CLASS NAMES

assert(all(predIdx >= 1 & predIdx <= numel(classNames)), ...
    "Invalid predicted class index.");

predClasses = classNames(predIdx);

%% 10. CHECK PREDICTION DISTRIBUTION

fprintf("\n====================================\n");
fprintf("PREDICTION DISTRIBUTION\n");
fprintf("====================================\n");

[classList, ~, classIdx] = unique(predClasses);
counts = accumarray(classIdx, 1);

for k = 1:numel(classList)

    fprintf("%-20s : %4d images (%.2f%%)\n", ...
        classList{k}, ...
        counts(k), ...
        100 * counts(k) / numTest);

end

fprintf("------------------------------------\n");
fprintf("Total predictions: %d\n", numel(predClasses));
fprintf("Number of predicted classes: %d\n", ...
    numel(unique(predClasses)));

%% 11. CREATE JSON

fprintf("\nCreating JSON file...\n");

jsonEntries = strings(numTest, 1);

for i = 1:numTest

    jsonEntries(i) = ...
        string(jsonencode(names{i})) + ":" + ...
        string(jsonencode(predClasses{i}));

end

jsonText = "{" + strjoin(jsonEntries, ",") + "}";

fid = fopen(outputJson, "w");

assert(fid ~= -1, ...
    "Unable to create JSON file.");

fprintf(fid, "%s", jsonText);
fclose(fid);

%% 12. VERIFY JSON

fprintf("\nVerifying JSON file...\n");

fid = fopen(outputJson, "r");

assert(fid ~= -1, ...
    "Unable to reopen JSON file.");

jsonContent = fread(fid, "*char")';
fclose(fid);

checkStruct = jsondecode(jsonContent);
jsonKeys = fieldnames(checkStruct);

assert(numel(jsonKeys) == numTest, ...
    "JSON does not contain the expected number of entries.");

fprintf("Number of JSON entries: %d\n", numel(jsonKeys));

%% 13. FINAL SUMMARY

fprintf("\n====================================\n");
fprintf("PREDICTION COMPLETED\n");
fprintf("====================================\n");

fprintf("Model: %s\n", modelFile);
fprintf("Test images: %d\n", numTest);
fprintf("Predictions: %d\n", numel(predClasses));
fprintf("JSON file: %s\n", outputJson);

fprintf("\nReady for submission.\n");
