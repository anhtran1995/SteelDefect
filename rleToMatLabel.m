clear; close all; clc

%% Convert RLE (running length encoding) to jpg file
originalTrainLabel = readmatrix('C:\LearningData\SteelDefect\train.csv','OutputType','string');
originalTrainLabel(1,:) = [];
numImg = size(originalTrainLabel,1)/4;

image1 = imread('C:\LearningData\SteelDefect\train_images\0002cc93b.jpg');
[imgHeight,imgWidth,~] = size(image1);
baseNum = '00000';
for ctImg = 1:numImg
    imgIdx = 4*(ctImg-1)+1:4*(ctImg);
    imgMask = ones(imgHeight,imgWidth,'uint8')*5; % background is class 5
    for classNum = 1:4
        imgMask = registerClass(originalTrainLabel,imgMask,imgIdx,classNum);
    end
    % NOTE: left pad file number with 0, else file will come in wrong order
    % e.g. 1 10 100 1000 instead of 1 2 3 4
    fileNum = [baseNum num2str(ctImg)];
    fileNum = fileNum(end-4:end);
    % NOTE: if write to jpg file will change the mask value
    fileName = ['label_' fileNum '.png'];
    disp(fileName);
    imwrite(imgMask,['C:\LearningData\SteelDefect\train_labels\' fileName]);
end

%% Local function
function imgMask = registerClass(originalTrainLabel,imgMask,imgIdx,classNum)
label = originalTrainLabel(imgIdx(classNum),2);
if ~ismissing(label)
    label = str2num(label);
    startPixel = label(1:2:end);
    numPixelIncrement = label(2:2:end);
    endPixel = startPixel+numPixelIncrement-1;
    
    for ct = 1:numel(startPixel)
        imgMask(startPixel(ct):endPixel(ct)) = classNum;
    end
end
end