% @Author: athul
% @Date:   2014-08-28 13:50:19
% @Last Modified by:   athul
% @Last Modified time: 2014-09-01 14:26:03

% This is a part of Assignment 1 of CS6690.

% In this part we do the svd analysis of the input image and plot the results.
% input argument is the path to the image file. make sure the image file is inside the search path of the code.
% Executing the code

%% svdAnanlysis: function to get image name as argument
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
svdPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);

% case 2  ----> Top Half SVs selcted
choosenSVs = [1:imHeight/2];
caption = 'Top Half SVs selcted';
svdPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);

% case 3  ----> selected top N so that 90% of 'power' is captured
i = 1;
[U_R, S_R, V_R] = svd(imgRChannel);
[U_G, S_G, V_G] = svd(imgGChannel);
[U_B, S_B, V_B] = svd(imgBChannel);
Rpower = sum(diag(S_R(1,1)))/sum(diag(S_R));
Gpower = sum(diag(S_G(1,1)))/sum(diag(S_G));
Bpower = sum(diag(S_B(1,1)))/sum(diag(S_B));
while ((Rpower+Gpower+Bpower)/3 < 0.9)
    i = i+1;
    Rpower = sum(diag(S_R(1:i, 1:i)))/sum(diag(S_R));
    Gpower = sum(diag(S_G(1:i, 1:i)))/sum(diag(S_G));
    Bpower = sum(diag(S_B(1:i, 1:i)))/sum(diag(S_B));
end
choosenSVs = [1: i];
clear('i', 'Rpower', 'Gpower', 'Bpower', 'U_B', 'U_G', 'U_R', 'V_B', 'V_G', 'V_R', 'S_G', 'S_R', 'S_B');

caption = 'top N so that 90 of power is captured';
svdPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);

% case 4  ----> Bottom Half SVs selcted
choosenSVs = imHeight/2: imHeight;
caption = 'Top Half SVs selcted';
svdPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);

% case 5  ----> selected bottom N so that 90% of 'power' is captured
[U_R, S_R, V_R] = svd(imgRChannel);
[U_G, S_G, V_G] = svd(imgGChannel);
[U_B, S_B, V_B] = svd(imgBChannel);
i = size(S_R,1);
Rpower = sum(diag(S_R(i,i)))/sum(diag(S_R));
Gpower = sum(diag(S_G(i,i)))/sum(diag(S_G));
Bpower = sum(diag(S_B(i,i)))/sum(diag(S_B));
while ((Rpower+Gpower+Bpower)/3 < 0.9)
    i = i-1;
    Rpower = sum(diag(S_R(i:end, i:end)))/sum(diag(S_R));
    Gpower = sum(diag(S_G(i:end, i:end)))/sum(diag(S_G));
    Bpower = sum(diag(S_B(i:end, i:end)))/sum(diag(S_B));
end
choosenSVs = 1:size(S_R,1);
clear('i', 'Rpower', 'Gpower', 'Bpower', 'U_B', 'U_G', 'U_R', 'V_B', 'V_G', 'V_R', 'S_G', 'S_R', 'S_B');
caption = 'bottom N so that 90 of power is captured';
svdPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);

% case 6  ----> Random half SVs. 
choosenSVs = sort(randi(imHeight, 1, round(imHeight/2)));
caption = 'Random half SVs';
svdPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption);


% Now we create Plots of RMSE Vs change in number of SVs
if (imHeight<280 && imHeight<280)
    for i=1:imHeight
        choosenSVs = [1: i];

        [U_R, S_R, V_R] = svd(imgRChannel);
        [U_G, S_G, V_G] = svd(imgGChannel);
        [U_B, S_B, V_B] = svd(imgBChannel);

        S_Rhat = S_R(choosenSVs, choosenSVs);   %selected singular vectors
        imgRhat = 255*U_R(:, choosenSVs)*S_Rhat*transpose(V_R(:, choosenSVs)); %recreate R channel

        S_Ghat = S_G(choosenSVs, choosenSVs);   %selected singular vectors
        imgGhat = 255*U_G(:, choosenSVs)*S_Ghat*transpose(V_G(:, choosenSVs)); %recreate G channel

        S_Bhat = S_B(choosenSVs, choosenSVs);   %selected singular vectors
        imgBhat = 255*U_B(:, choosenSVs)*S_Bhat*transpose(V_B(:, choosenSVs)); %recreate B channel

        imgHat = uint8(imgRhat);
        imgHat(:,:,2) = uint8(imgGhat); imgHat(:,:,3) = uint8(imgBhat);
        rmse(i) = rms(rms(rms(oriImg - imgHat)));
    end

    % plot RMSE Vs number of SVs
    figure;
    plot(choosenSVs, rmse);
    title('RMSE error Vs number of SVs')
    set(get(gca,'XLabel'),'String','number of SVs');
    set(get(gca,'YLabel'),'String','RMSE');
end