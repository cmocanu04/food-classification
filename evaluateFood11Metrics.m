function [accTrain, f1macroTrain, accVal, f1macroVal] = evaluateFood11Metrics( ...
    net, augTrainEval, labelsTrain, augVal, labelsVal)
% EVALUATEFOOD11METRICS Evaluate a trained network on train AND val sets.
%   evaluateFood11Metrics(net, augTrainEval, labelsTrain, augVal, labelsVal)
%
% Reports accuracy + MACRO F1 (the leaderboard metric) on:
%   - the TRAIN set (resize only, no augmentation) : fit quality
%   - the VAL set                                  : generalization
% plus the train-val gap (surapprentissage indicator), per-class val F1
% and the validation confusion matrix.
%
% augTrainEval must be the resize-only train datastore created by
% FoodClassification_database.m (NOT augTrain, which augments).

% ---- Train set (fit quality) ----
trainScores = minibatchpredict(net, augTrainEval);
trainPred = scores2label(trainScores, categories(labelsTrain));
accTrain = mean(trainPred == labelsTrain);
f1macroTrain = mean(perClassF1(confusionmat(labelsTrain, trainPred)));

% ---- Validation set (generalization) ----
valScores = minibatchpredict(net, augVal);
classNames = categories(labelsVal);
valPred = scores2label(valScores, classNames);
accVal = mean(valPred == labelsVal);
f1perclass = perClassF1(confusionmat(labelsVal, valPred));
f1macroVal = mean(f1perclass);

% ---- Report ----
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

fprintf("Per-class F1 (val):\n");
for c = 1:numel(classNames)
    fprintf("  %-16s %.4f\n", classNames{c}, f1perclass(c));
end

figure(Name="Validation confusion matrix");
confusionchart(labelsVal, valPred, RowSummary="row-normalized");
end

function f1 = perClassF1(cm)
% Per-class F1 from a confusion matrix (rows = true, cols = predicted).
tp = diag(cm);
fp = sum(cm, 1)' - tp;
fn = sum(cm, 2) - tp;
den = 2*tp + fp + fn;
f1 = 2*tp ./ den;
f1(den == 0) = 0;
end
