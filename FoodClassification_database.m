% FoodClassification_database.m
% Chargement des donnees Food-11 via datastores (resolution quelconque,
% augmentation a la volee pour le split train).
%
% Variables produites : trainData, valData, labelsTrain, labelsVal,
% augTrain (resize + augmentation), augVal (resize seul),
% augTrainEval (resize seul, pour mesurer l'accuracy train apres coup).
%
% IMPORTANT : inputSize doit correspondre au inputSize du script
% *_Implementation.m lance ensuite.
%   - [64 64 3]  : experiments rapides (reglage actuel)
%   - [224 224 3]: entrainement final (resolution native ImageNet)

clearvars
close all

% Germe fixe : le split 80/20 randomise est identique a chaque execution ->
% tous les reseaux sont entraines/valides sur les memes images (comparaison equitable).
rng(42)

inputSize = [64 64 3];  % <<< mettre [224 224 3] pour l'entrainement final

root = fullfile(fileparts(mfilename("fullpath")), "food-11", "food-11");

% imageDatastore : ne charge rien en RAM, liste juste les chemins. Les images
% sont lues du disque a la demande, batch par batch. Le nom de chaque
% sous-dossier devient le label de l'image (pas d'annotation manuelle).
data = imageDatastore(fullfile(root, "train"), ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

% 80 % train / 20 % validation, stratifie par classe : chaque classe garde
% la meme proportion dans les deux jeux (les classes sont desequilibrees,
% soup: 2000 vs rice: 376). Le reseau apprend sur le train ; la val sert
% seulement a mesurer la generalisation sur des images jamais vues.
[trainData, valData] = splitEachLabel(data, 0.8, "randomized");
labelsTrain = trainData.Labels;
labelsVal = valData.Labels;

% ---- Augmentation de donnees : TRAIN uniquement ----
% A chaque lecture, une transformation aleatoire est appliquee : le reseau ne
% voit jamais deux fois la meme image -> il apprend le concept, pas les
% pixels (anti-overfitting). Chaque transformation modelise une variation
% reelle plausible :
% - horizontal flip : une assiette en miroir reste le meme plat
% - rotation / translation / scale : angle de camera, cadrage, distance
% - PAS de flip vertical (assiette a l'envers = contre nature)
% - PAS de jitter de couleur (la couleur est informative en food)
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-15 15], ...
    RandXTranslation=[-10 10], ...
    RandYTranslation=[-10 10], ...
    RandXScale=[0.9 1.1], ...
    RandYScale=[0.9 1.1]);

% Train : resize + augmentation aleatoire a chaque lecture (c'est ce datastore
% qui nourrit trainnet).
augTrain = augmentedImageDatastore(inputSize(1:2), trainData, ...
    DataAugmentation=augmenter);

% Validation : resize seul. Jamais d'augmentation : il faut une mesure stable
% et reproductible, comparable d'une epoque a l'autre.
augVal = augmentedImageDatastore(inputSize(1:2), valData);

% Copie du train SANS augmentation : sert a mesurer l'accuracy train APRES
% l'entrainement. augTrain ne peut pas servir (il re-augmente a chaque
% lecture -> score sous-estime et non reproductible).
augTrainEval = augmentedImageDatastore(inputSize(1:2), trainData);

fprintf("Train: %d images | Val: %d images | input %dx%d\n", ...
    numel(labelsTrain), numel(labelsVal), inputSize(1), inputSize(2));
