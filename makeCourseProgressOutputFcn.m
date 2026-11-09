function outputFcn = makeCourseProgressOutputFcn(numTrainingObservations, miniBatchSize, maxEpochs, smoothingWindow)
% Create a live training plot with smoothed training curves and epoch bands.
% The test set is passed to trainnet as ValidationData, but is labeled as
% the test set here to match the terminology used in these exercises.

iterationsPerEpoch = ceil(numTrainingObservations / miniBatchSize);
totalIterations = iterationsPerEpoch * maxEpochs;

fig = figure(Name="Training Progress", NumberTitle="off");
layout = tiledlayout(fig, 2, 1, TileSpacing="compact");

axAccuracy = nexttile(layout);
hold(axAccuracy, "on");
ylim(axAccuracy, [0 100]);
xlim(axAccuracy, [0 totalIterations]);
ylabel(axAccuracy, "Accuracy (%)");
title(axAccuracy, "Accuracy");
addEpochBands(axAccuracy, iterationsPerEpoch, maxEpochs, [0 100]);
accuracyTrainLine = animatedline(axAccuracy, ...
    Color=[0 0.4470 0.7410], LineWidth=1.8);
accuracyTestLine = animatedline(axAccuracy, ...
    Color=[0.8500 0.3250 0.0980], LineWidth=1.4, ...
    Marker="o", MarkerSize=4);
legend(axAccuracy, [accuracyTrainLine accuracyTestLine], ...
    "Training (smoothed)", "Test set", Location="southeast");

axLoss = nexttile(layout);
hold(axLoss, "on");
lossTop = 3;
ylim(axLoss, [0 lossTop]);
xlim(axLoss, [0 totalIterations]);
xlabel(axLoss, sprintf("Iteration (each epoch has %d iterations; alternating bands mark epochs)", iterationsPerEpoch));
ylabel(axLoss, "Cross-entropy loss");
title(axLoss, "Loss");
lossBands = addEpochBands(axLoss, iterationsPerEpoch, maxEpochs, [0 lossTop]);
lossTrainLine = animatedline(axLoss, ...
    Color=[0 0.4470 0.7410], LineWidth=1.8);
lossTestLine = animatedline(axLoss, ...
    Color=[0.8500 0.3250 0.0980], LineWidth=1.4, ...
    Marker="o", MarkerSize=4);
legend(axLoss, [lossTrainLine lossTestLine], ...
    "Training (smoothed)", "Test set", Location="northeast");

title(layout, sprintf("Training progress (%d-iteration moving average)", smoothingWindow));

trainingLossHistory = [];
trainingAccuracyHistory = [];
outputFcn = @updatePlot;

    function stop = updatePlot(info)
        stop = false;
        if ~isgraphics(fig)
            return
        end
        if string(info.State) ~= "iteration"
            return
        end

        iteration = info.Iteration;
        if ~isempty(info.TrainingLoss)
            trainingLossHistory(end + 1) = info.TrainingLoss;
            smoothedLoss = movmean(trainingLossHistory, [smoothingWindow - 1 0]);
            addpoints(lossTrainLine, iteration, smoothedLoss(end));

            requiredLossTop = max(3, 1.05 * max(smoothedLoss));
            if requiredLossTop > lossTop
                lossTop = requiredLossTop;
                ylim(axLoss, [0 lossTop]);
                updateEpochBands(lossBands, [0 lossTop]);
            end
        end

        if isfield(info, 'TrainingAccuracy') && ~isempty(info.TrainingAccuracy)
            trainingAccuracyHistory(end + 1) = info.TrainingAccuracy;
            smoothedAccuracy = movmean(trainingAccuracyHistory, [smoothingWindow - 1 0]);
            addpoints(accuracyTrainLine, iteration, smoothedAccuracy(end));
        end

        if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss)
            addpoints(lossTestLine, iteration, info.ValidationLoss);
        end
        if isfield(info, 'ValidationAccuracy') && ~isempty(info.ValidationAccuracy)
            addpoints(accuracyTestLine, iteration, info.ValidationAccuracy);
        end

        drawnow limitrate
    end
end

function bands = addEpochBands(ax, iterationsPerEpoch, maxEpochs, yLimits)
% Shade alternate epochs and draw a light boundary at every epoch change.
bands = gobjects(0);
for epoch = 1:maxEpochs
    firstIteration = (epoch - 1) * iterationsPerEpoch + 0.5;
    lastIteration = epoch * iterationsPerEpoch + 0.5;
    if mod(epoch, 2) == 0
        bands(end + 1) = patch(ax, ...
            [firstIteration lastIteration lastIteration firstIteration], ...
            [yLimits(1) yLimits(1) yLimits(2) yLimits(2)], ...
            [0.94 0.94 0.94], EdgeColor="none", ...
            HandleVisibility="off", PickableParts="none"); %#ok<AGROW>
    end
    if epoch > 1
        xline(ax, firstIteration, Color=[0.78 0.78 0.78], ...
            HandleVisibility="off", PickableParts="none");
    end
end
end

function updateEpochBands(bands, yLimits)
for idx = 1:numel(bands)
    x = bands(idx).XData;
    bands(idx).YData = [yLimits(1) yLimits(1) yLimits(2) yLimits(2)];
    bands(idx).XData = x;
end
end
