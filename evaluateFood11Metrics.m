function [accTrain, f1macroTrain, accVal, f1macroVal] = evaluateFood11Metrics( ...
    net, augTrainEval, labelsTrain, augVal, labelsVal)
% EVALUATEFOOD11METRICS Evalue un reseau entraine sur le train ET la val.
%   evaluateFood11Metrics(net, augTrainEval, labelsTrain, augVal, labelsVal)
%
% Rapporte accuracy + MACRO F1 (metrique du leaderboard) sur :
%   - le TRAIN (resize seul, sans augmentation) : qualite du fit
%   - la VAL                                    : generalisation
% plus l'ecart train-val (indicateur de surapprentissage), le F1 par classe
% et la matrice de confusion de la validation.

% augTrainEval doit etre le datastore train resize-seul cree par
% FoodClassification_database.m (PAS augTrain, qui augmente).

% ---- Train (qualite du fit) ----
% minibatchpredict : inference par lots, sans gradients ni maj des poids.
% scores2label : retient la classe au score softmax maximal de chaque image.
% accuracy = proportion d'images bien classees.
trainScores = minibatchpredict(net, augTrainEval);
trainPred = scores2label(trainScores, categories(labelsTrain));
accTrain = mean(trainPred == labelsTrain);
f1macroTrain = mean(perClassF1(confusionmat(labelsTrain, trainPred)));

% ---- Validation (generalisation) ----
valScores = minibatchpredict(net, augVal);
classNames = categories(labelsVal);
valPred = scores2label(valScores, classNames);
accVal = mean(valPred == labelsVal);
f1perclass = perClassF1(confusionmat(labelsVal, valPred));
f1macroVal = mean(f1perclass);

% ---- Rapport ----
% Diagnostic par l'ecart train-val :
%   train >> val        -> surapprentissage (memorisation, pas de generalisation)
%   train faible aussi  -> sous-apprentissage (capacite/entrainement insuffisant)
%   train ~ val eleves  -> bon compromis
fprintf("Train-set accuracy : %.2f %%  (macro F1 %.4f)\n", 100*accTrain, f1macroTrain);
fprintf("Val-set   accuracy : %.2f %%  (macro F1 %.4f)\n", 100*accVal, f1macroVal);
fprintf("Ecart train-val    : %.2f points ", 100*(accTrain - accVal));
if accTrain - accVal > 10
    fprintf("(surapprentissage probable)\n");
elseif accTrain < 0.75
    fprintf("(sous-apprentissage probable)\n");
else
    fprintf("(bon compromis)\n");
end

% F1 par classe : identifie les classes faibles que la moyenne masque.
fprintf("Per-class F1 (val):\n");
for c = 1:numel(classNames)
    fprintf("  %-16s %.4f\n", classNames{c}, f1perclass(c));
end

% Matrice de confusion normalisee par ligne : ligne = vraie classe,
% colonne = classe predite. Les cellules hors diagonale montrent OU le
% reseau se trompe (ex : rice confondu avec noodles).
figure(Name="Validation confusion matrix");
confusionchart(labelsVal, valPred, RowSummary="row-normalized");
end

function f1 = perClassF1(cm)
% F1 par classe depuis la matrice de confusion (lignes = vrai, colonnes = predit).
% Pour la classe c :
%   tp = bien classees c ; fp = autres classes predites c (fausses alertes) ;
%   fn = c predites autre chose (oublis)
%   precision = tp/(tp+fp) : quand je dis "c", ai-je raison ?
%   recall    = tp/(tp+fn) : trouve-je tous les "c" ?
%   F1 = moyenne harmonique = 2*tp / (2*tp + fp + fn)
% F1 = 0 si la classe n'est jamais trouvee (tp = 0) : c'est ce qui rend le
% macro F1 plus strict que l'accuracy sur les petites classes.
tp = diag(cm);
fp = sum(cm, 1)' - tp;
fn = sum(cm, 2) - tp;
den = 2*tp + fp + fn;
f1 = 2*tp ./ den;
f1(den == 0) = 0;
end
