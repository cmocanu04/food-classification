function [acc, f1macro, f1perclass] = evaluateFood11Metrics(net, augVal, labelsVal)
% EVALUATEFOOD11METRICS Evaluate a trained network on the validation set.
%   [acc, f1macro, f1perclass] = evaluateFood11Metrics(net, augVal, labelsVal)
%
% Reports accuracy, MACRO F1 (the leaderboard metric) and per-class F1,
% and displays the confusion matrix. Works with any dlnetwork trained by
% trainnet, on an augmentedImageDatastore used for validation (resize only).

scores = minibatchpredict(net, augVal);
classNames = categories(labelsVal);
pred = scores2label(scores, classNames);

acc = mean(pred == labelsVal);

cm = confusionmat(labelsVal, pred);
tp = diag(cm);
fp = sum(cm, 1)' - tp;
fn = sum(cm, 2) - tp;
den = 2*tp + fp + fn;
f1perclass = 2*tp ./ den;
f1perclass(den == 0) = 0;
f1macro = mean(f1perclass);

fprintf("Validation accuracy : %.2f %%\n", 100*acc);
fprintf("Validation macro F1 : %.4f\n", f1macro);
disp(table(string(classNames)', f1perclass, ...
    VariableNames={"class", "F1"}));

confusionchart(labelsVal, pred, RowSummary="row-normalized");
end
