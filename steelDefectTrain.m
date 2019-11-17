clear; close all; clc

%% Process data
imgDir = 'C:\LearningData\SteelDefect\train_images';
labelDir = 'C:\LearningData\SteelDefect\train_labels';
imds = imageDatastore(imgDir);

classes = [
    "Defect1"
    "Defect2"
    "Defect3"
    "Defect4"
    "Background"
    ];
labelIDs = (1:5)';
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

%% Test with 1 image
% imgNum = 24;
% I = readimage(imds,imgNum);
% cmap = defectColorMap();
% subplot(2,1,1)
% imshow(I)
% C = readimage(pxds,imgNum);
% B = labeloverlay(I,C,'ColorMap',cmap);
% subplot(2,1,2)
% imshow(B)
% pixelLabelColorbar(cmap,classes);

%% Analyze Dataset Statistics
% tbl = countEachLabel(pxds)
load('labelTable.mat','tbl')
frequency = tbl.PixelCount/sum(tbl.PixelCount);
% bar(1:numel(classes),frequency)
% xticks(1:numel(classes)) 
% xticklabels(tbl.Name)
% xtickangle(45)
% ylabel('Frequency')

% a = tbl;
% a(5,:) = [];
% figure;frequency = a.PixelCount/sum(a.PixelCount);
% bar(1:numel(classes)-1,frequency)
% xticks(1:numel(classes))
% xticklabels(a.Name)
% xtickangle(45)
% ylabel('Frequency')

%% Partition
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds);
numTrainingImages = numel(imdsTrain.Files)
numValImages = numel(imdsVal.Files)
numTestingImages = numel(imdsTest.Files)

%% Create U-net
% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [256 1600 3];

% Specify the number of classes.
numClasses = numel(classes);

% Create default u-net
lgraph = unetLayers(imageSize,numClasses);

%% Balance Classes Using Class Weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,lgraph.Layers(end).Name,pxLayer);

%%
% Define validation data.
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

% Define training options. 
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.1,...
    'InitialLearnRate',2e-3, ...
    'L2Regularization',1e-4, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',50, ...  
    'MiniBatchSize',6, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress');

%% Data Augmentation
% TODO: enable augmentation
% augmenter = imageDataAugmenter('RandXReflection',true,...
%     'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
%     'RandRotation',[-10 10]);

%% Train
% pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
%     'DataAugmentation',augmenter);
pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);
[net, info] = trainNetwork(pximds,lgraph,options);

%%Test Network on One Image
I = readimage(imdsTest,35);
C = semanticseg(I, net);
expectedResult = readimage(pxdsTest,35);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

%% Local functions

% Data processing
function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionData(imds,pxds)
% Partition data by randomly selecting 80% of the data for training. The
% rest is used for testing.

% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 80% of the images for training.
numTrain = round(0.80 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 10% of the images for validation
numVal = round(0.10 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = (1:5)';

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

% Visualization
function cmap = defectColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    128 0 0       % Defect 1
    128 64 128    % Defect 2
    60 40 222     % Defect 3
    192 128 128   % Defect 4
    128 128 128   % Background
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end

function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end
