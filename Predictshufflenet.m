
% Predictshufflenet.m
% Predictions finales du ShuffleNet (Test 2) sur le jeu de test officiel
% Food-11 : charge le reseau entraine, predit chaque image, ecrit le JSON
% de soumission {"nom_image": "classe", ...}.

close all;
clc;

%% 1. PARAMETRES

% Reseau sauvegarde par ShuffleNet_Implementation.m (Test 2 : fine-tuning complet).
modelFile = "ShuffleNet_Food11_test2_full.mat";

testDir = fullfile("food-11", "food-11", "test");

% Le dossier train sert seulement a retrouver l'ordre des classes (ci-dessous).
if isfolder(fullfile("food-11", "train"))
    trainDir = fullfile("food-11", "train");
elseif isfolder(fullfile("food-11", "food-11", "train"))
    trainDir = fullfile("food-11", "food-11", "train");
else
    error("Training directory not found.");
end

outputJson = "predictions_shufflenet_final.json";

% IMPORTANT : doit correspondre au inputSize utilise PENDANT l'entrainement
% du modele charge ([224 224 3] pour le modele final).
inputSize = [224 224 3];
% Batch de prediction plus grand qu'a l'entrainement : pas de gradients ->
% beaucoup moins de memoire par image, on peut aller plus vite.
miniBatchSize = 250;

%% 2. VERIFICATION DES FICHIERS ET DOSSIERS

assert(isfile(modelFile), ...
    "Model file not found: %s", modelFile);

assert(isfolder(testDir), ...
    "Test directory not found: %s", testDir);

assert(isfolder(trainDir), ...
    "Training directory not found: %s", trainDir);

fprintf("Training directory: %s\n", trainDir);
fprintf("Test directory: %s\n", testDir);

%% 3. CHARGEMENT DU MODELE ENTRAINE

fprintf("\nLoading ShuffleNet Test 2 model...\n");

% Le .mat contient la variable netT2 sauvegardee apres l'entrainement.
S = load(modelFile);

assert(isfield(S, "netT2"), ...
    "Variable netT2 not found in model file.");

net = S.netT2;

%% 4. ORDRE DES CLASSES (celui de l entrainement)

fprintf("\nRetrieving class order...\n");

% On reconstruit l'ordre des classes depuis le dataset d'entrainement :
% la sortie du reseau est un vecteur de 11 scores, ou l'indice k correspond
% a la k-ieme classe de categories(labelsTrain) (tri alphabetique). Si l'ordre
% recree ici differait de celui de l'entrainement, les indices seraient
% silencieusement permutes -> predictions toutes fausses sans erreur visible.
trainingDS = imageDatastore( ...
    trainDir, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

classNames = categories(trainingDS.Labels);

fprintf("\nClasses in training order:\n");
disp(classNames);

assert(numel(classNames) == 11, ...
    "Expected 11 classes, found %d.", numel(classNames));

%% 5. CHARGEMENT DU JEU DE TEST

fprintf("\nLoading test dataset...\n");

% Le dossier test est plat (pas de sous-dossiers de classes) : pas de labels,
% c'est justement ce qu'on doit predire.
testDS = imageDatastore( ...
    testDir, ...
    IncludeSubfolders=false);

files = testDS.Files;
numTest = numel(files);

fprintf("Number of test images: %d\n", numTest);

assert(numTest == 3097, ...
    "Expected 3097 test images, found %d.", numTest);

%% 6. EXTRACTION DES IDENTIFIANTS D IMAGES

% Le JSON de soumission associe le NOM de fichier (sans extension) a la
% classe predite : on extrait donc ces identifiants. Verif unicite pour
% garantir une entree JSON par image.
names = cell(numTest, 1);

for i = 1:numTest
    [~, name, ~] = fileparts(files{i});
    names{i} = name;
end

assert(numel(unique(names)) == numTest, ...
    "Duplicate image identifiers detected.");

%% 7. PREPARATION DES IMAGES DE TEST

fprintf("\nPreparing test images...\n");

% Tout le test set est charge en RAM dans un seul tenseur 224x224x3xN :
% preparation en une passe, puis predictions par lots rapides.
imgTest = zeros( ...
    inputSize(1), ...
    inputSize(2), ...
    inputSize(3), ...
    numTest, ...
    "uint8");

for i = 1:numTest

    img = readimage(testDS, i);

    % Normalisation du nombre de canaux vers 3 (RGB), ce qu'attend la
    % couche d'entree : niveaux de gris dupliques, RGBA ramene a RGB.
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);

    % Ne garder que les 3 premiers canaux si besoin (ex : RGBA -> RGB).
    elseif size(img, 3) > 3
        img = img(:, :, 1:3);
    end

    % Resize en preservant le type des pixels (uint8).
    img = imresize(img, inputSize(1:2));

    imgTest(:, :, :, i) = img;

    if mod(i, 500) == 0
        fprintf("Prepared %d / %d images\n", i, numTest);
    end

end

fprintf("Image preparation completed.\n");

%% 8. EXECUTION DES PREDICTIONS

fprintf("\nStarting predictions...\n");

% Predictions par lots de miniBatchSize : inference pure (pas de gradients,
% pas de maj des poids). max(scores, [], 2) = indice du score maximal de
% chaque image = classe predite.
predIdx = zeros(numTest, 1);

for startIdx = 1:miniBatchSize:numTest

    endIdx = min(startIdx + miniBatchSize - 1, numTest);

    batch = imgTest(:, :, :, startIdx:endIdx);

    scores = minibatchpredict(net, batch);

    [~, batchPredIdx] = max(scores, [], 2);

    predIdx(startIdx:endIdx) = batchPredIdx;

    fprintf("Predicted %d / %d images\n", endIdx, numTest);

end

%% 9. CONVERSION DES INDICES EN NOMS DE CLASSES

assert(all(predIdx >= 1 & predIdx <= numel(classNames)), ...
    "Invalid predicted class index.");

predClasses = classNames(predIdx);

%% 10. VERIFICATION DE LA DISTRIBUTION DES PREDICTIONS

% Sanity check : la distribution des predictions doit ressembler a celle du
% train (classes desequilibrees mais toutes presentes). Une classe jamais
% predite ou une classe a 80 % serait suspecte (bug d'ordre des classes,
% mauvais modele charge...).
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

%% 11. CREATION DU JSON

% Format de soumission : {"nom_image": "classe_predite", ...}.
% jsonencode sur chaque cle/valeur pour echapper correctement les caracteres,
% puis re-decodage en fin de script pour verifier l'integralite du fichier.
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

%% 12. VERIFICATION DU JSON

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

%% 13. BILAN FINAL

fprintf("\n====================================\n");
fprintf("PREDICTION COMPLETED\n");
fprintf("====================================\n");

fprintf("Model: %s\n", modelFile);
fprintf("Test images: %d\n", numTest);
fprintf("Predictions: %d\n", numel(predClasses));
fprintf("JSON file: %s\n", outputJson);

fprintf("\nReady for submission.\n");
