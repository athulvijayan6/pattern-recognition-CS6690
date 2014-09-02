% @Author: athul
% @Date:   2014-09-01 15:28:20
% @Last Modified by:   athul
% @Last Modified time: 2014-09-01 16:59:20
function [] = eigPlot(imgRChannel, imgGChannel, imgBChannel, singleMat, choosenSVs, map, oriImg, caption)
    Rhat = eigfn(imgRChannel, choosenSVs); Ghat = eigfn(imgGChannel, choosenSVs);
    Bhat = eigfn(imgBChannel, choosenSVs);
    singleMatHat = round(eigfn(singleMat, choosenSVs));
    singleMatHat = uint16(real(singleMatHat));
    imgHat = uint8(Rhat);
    imgHat(:,:,2) = uint8(Ghat); imgHat(:,:,3) = uint8(Bhat);
    singleMatHat = ind2rgb(singleMatHat, map);

    % RMSE = rms(rms(rms(oriImg - imgHat)));
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
    % text(0.2,0.5,['The RMSE error is ', num2str(RMSE)]);
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
    % text(0.2,0.5,['The RMSE error is ', num2str(RMSE)]);
    set ( tsp, 'visible', 'off')
    ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(0.5, 1,caption,'HorizontalAlignment','center','VerticalAlignment', 'top')
    return
end

function [imgHat] = eigfn(img, choosenSVs)
    if (size(img, 1) == size(img, 2))
        [V, D] = eig(img);                                  % do svd on input
        Dhat = D(choosenSVs, choosenSVs);                   %  selected singular vectors
        Vinv = inv(V);
        imgHat = V(:, choosenSVs)*Dhat*Vinv(choosenSVs, :); %recreate image
    else                                                    % do svd using eig
        [V, D1] = eig(img'*img);
        [U, D2] = eig(img*img');
        S = sqrt(D1(1:size(U,2),1:size(V, 1)));
        Shat = S(choosenSVs, choosenSVs);               %  selected singular vectors
        imgHat = U(:, choosenSVs)*Shat*transpose(V(:, choosenSVs)); %recreate image
    end
end