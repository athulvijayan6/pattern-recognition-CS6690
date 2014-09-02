% @Author: athul
% @Date:   2014-08-28 13:50:48
% @Last Modified by:   athul
% @Last Modified time: 2014-09-01 17:01:17


clear all
inputImage = '29.jpg';
oriImg = imread(inputImage);

imageInfo = imfinfo(inputImage);
imgRChannel = im2double(oriImg(:,:,1));                % extract R channel
imgGChannel = im2double(oriImg(:,:,2));                % extract G channel
imgBChannel = im2double(oriImg(:,:,3));                % extract B channel
% singleMat is the matrix where image is stored in 2D, map in the color map. and it can map into maximum of 65536 colors
[singleMat, map] = rgb2ind(oriImg, 65536);
singleMat = double(singleMat);

clear all
inputImage = '29.jpg';
oriImg = imread(inputImage);

imageInfo = imfinfo(inputImage);
imWidth = imageInfo.Width;
imHeight = imageInfo.Height;

% singleMat is the matrix where image is stored in 2D, map in the color map. and it can map into maximum of 65536 colors
[singleMat, map] = rgb2ind(oriImg, 65536);
singleMat = double(singleMat);

% Matlab scales the values from 0-1
imgRChannel = double(oriImg(:,:,1));                % extract R channel
imgGChannel = double(oriImg(:,:,2));                % extract G channel
imgBChannel = double(oriImg(:,:,3));                % extract B channel

% Now we have all the matrices for applying svd. we choose a particular set of SVs and apply svdPlot function

% case 1  ----> Top SVs, all SVs
choosenSVs = [1:imHeight];
caption = 'All SVs selcted';
eigPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);

% case 2  ----> Top Half SVs selcted
choosenSVs = [1:imHeight/2];
caption = 'Top Half SVs selcted';
eigPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);
