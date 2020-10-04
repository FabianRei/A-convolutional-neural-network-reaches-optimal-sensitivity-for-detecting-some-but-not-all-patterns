% Script to calculate the optimal observer accuracy in Matlab
% lowest val at 60, highest at 181

pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files/10000_samplesPerClass_freq_1_contrast_0_001.h5";
noNoiseImg = hdf5read(pathMat, 'noNoiseImg');
imgNoise = hdf5read(pathMat, 'imgNoise');
imgNoiseFreqs = hdf5read(pathMat, 'imgNoiseFreqs');
sizeNoNoiseImg = size(noNoiseImg);

allAccuracy = [];
allHits = [];
allFalseAlarms = [];
for i = 1:length(imgNoise)
    noiseImg = imgNoise(:,:,i);
    noiseLabel = imgNoiseFreqs(i);
    llVals = [];
    for j = 1:sizeNoNoiseImg(3)
        param.mean = noNoiseImg(:,:,j);
        param.mean = param.mean(:);
        sample = noiseImg(:);
        ll = ieLogLikelihood(sample, param, 'distribution', 'poisson');
        llVals = [llVals ll];
    end
    prediction = llVals(1)<llVals(2);
    allAccuracy = [allAccuracy prediction == noiseLabel];
    if noiseLabel == 1
        allHits = [allHits prediction == noiseLabel];
    else
        allFalseAlarms = [allFalseAlarms prediction ~= noiseLabel];
    end
    if mod(i, 100) == 0
        fprintf("%d out of %d scenes are done.\n", i, length(imgNoise))
        disp("Currently, optimal observer accuracy is: " + num2str(mean(allAccuracy)));

    end
end
    
disp("Here in Matlab, optimal observer accuracy is: " + num2str(mean(allAccuracy)));
disp("Here in Matlab, optimal observer hit rate is: " + num2str(mean(allHits)));
disp("Here in Matlab, optimal observer false alarms is: " + num2str(mean(allFalseAlarms)));

d = norminv(mean(allHits)) - norminv(mean(allFalseAlarms));
dTheo = sum((beta-alpha) .* log(beta./alpha), 'all') / sqrt(0.5*sum(((alpha+beta) .* log(beta./alpha).^2), 'all'));
disp("Here in Matlab, optimal observer d value is: " + num2str(d));
disp("Here in Matlab, theoretical optimal observer d value is: " + num2str(dTheo));


