% @Author: athul
% @Date:   2014-09-01 11:59:54
% @Last Modified by:   athul
% @Last Modified time: 2014-09-01 13:40:50

%% svdPlot: function description
function [] = svdPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption)
    Rhat = svdfn(imgRChannel, choosenSVs); Ghat = svdfn(imgGChannel, choosenSVs);
    Bhat = svdfn(imgBChannel, choosenSVs);
    singleMatHat = round(svdfn(singleMat, choosenSVs));
    singleMatHat = uint16(singleMatHat);
    imgHat = uint8(Rhat);
    imgHat(:,:,2) = uint8(Ghat); imgHat(:,:,3) = uint8(Bhat);
    singleMatHat = ind2rgb(singleMatHat, map);


    RMSE = rms(rms(rms(oriImg - imgHat)));
    figure;
    subplot(2,2,1);
    image(oriImg);
    title('original image');
    subplot(2, 2,2);
    image(imgHat);
    title('image recreated using seperate channels')
    subplot(2, 2, 3);
    image(oriImg-imgHat);
    title('Error image. $$A - \hat{A}$$', 'Interpreter','Latex')
    tsp = subplot(2,2,4);
    text(0.2,0.5,['The RMSE error is ', num2str(RMSE)]);
    set ( tsp, 'visible', 'off')
    ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(0.5, 1,caption,'HorizontalAlignment','center','VerticalAlignment', 'top')

    figure;
    subplot(2, 2,1);
    image(oriImg);
    title('original image');
    RMSE = rms(rms(rms(oriImg - im2uint8(singleMatHat))));
    subplot(2, 2,2);
    image(singleMatHat);
    title('image recreated using indexed matrix')
    subplot(2, 2, 3);
    image(oriImg- im2uint8(singleMatHat));
    title('Error image. $$A - \hat{A}$$', 'Interpreter','Latex')
    tsp = subplot(2,2,4);
    text(0.2,0.5,['The RMSE error is ', num2str(RMSE)]);
    set ( tsp, 'visible', 'off')
    ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(0.5, 1,caption,'HorizontalAlignment','center','VerticalAlignment', 'top')
    return
end

function [imgHat] = svdfn(img, choosenSVs)
    [U, S, V] = svd(img);                           %do svd on input
    Shat = S(choosenSVs, choosenSVs);               %  selected singular vectors
    imgHat = U(:, choosenSVs)*Shat*transpose(V(:, choosenSVs)); %recreate image
end

