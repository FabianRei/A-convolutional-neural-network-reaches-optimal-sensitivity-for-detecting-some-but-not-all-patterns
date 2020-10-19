% use simple example to check the magnitude of the rounding error of ll
% calculation

pathMat = "/black/localhome/reith/Desktop/projects/WLDiscriminationNetwork/deepLearning/data/mat_files/10000_samplesPerClass_freq_1_contrast_0_001.h5";
noNoiseImg = hdf5read(pathMat, 'noNoiseImg');
alpha = noNoiseImg(:,:,1);
beta = noNoiseImg(:,:,2);

sample = ones(50000,1);
noNoise.mean = ones(50000,1);

expectedValue = -1000;
test = ieLogLikelihood(sample, noNoise);
disp(test)


t = 1.817120592832139;

sample = sample*3;
noNoise.mean = noNoise.mean*t;
expectedValue2 = -2000;
test2 = ieLogLikelihood(sample, noNoise);
disp(test2)