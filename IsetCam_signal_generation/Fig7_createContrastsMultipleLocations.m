% t_CreateManyConeAbsorptionSignalNoiseDatasets
% Create multiple datasets consisting of the cone absorption of signals
% with and without added noise
%
% Description:
%    This tutorial creates a dataset for each of the contrast values in
%    "contrastValues". The resulting dataset consists of a signal, whose strength is 
%    defined by "contrast". The signal consists of stripes that can be seen
%    with a frequency of "frequency" on the generated image. In addition to
%    the signal, there is noise added as well. 
%    This image is then processed by a cone absorption function that 
%    simulates the stimulus generated, would this signal/noise image be 
%    absorbed by eye cones. The resulting image is then center cropped from
%    a size of 249x249 to a size of 238x238. This does not affect the 
%    created signal pattern itself, as it is pretty much only visible
%    within the cropped 238x238 center. 
%
%    "numSamples" images with noise are generated for each frequency, as
%    well as "numSamples" of images with noise only (-> no signal). For
%    each of these "numSamples" images, two addtitional images are created,
%    each consisting of a mean image (only signal, no noise).  {Why two
%    when one is enough? - Just for the user to verify that different runs
%    create the same no noise image. Also, to preserve the dimensional
%    structure.}
%
% See Also:
%    CreateConeAbsorptionSignalNoiseDataset_function

% Values to set
superOutputFolder = fullfile(onRootPath, 'local', 'multiple_locations');

mkdir(superOutputFolder);
numSamples = 1;
frequencies = 1;
% We increase contrast 40x, as garbor decreases max contast. The area of
% harmonic is decreased significantly as well. 
contrastValues = logspace(-5, -1.7, 12)*40;
contrastFreqPairs = [];
frequencyValues = 1;

% This encodes the locations of the signal (see addSignalToLoc)
gridlocs = {{1, {1}}, {3, {4,6}}, {3, {1,2,3,4,5,6,7,8,9}},{2, {1,2,3,4}}, {4, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}}};


% This creates the resulting datasets
gridlocs = gridlocs(2);
for f = 1:length(frequencyValues)
    for gl = 1:length(gridlocs)
        signalGridSize = gridlocs{gl}{1};
        signalLocation = gridlocs{gl}{2};
        frequencies = frequencyValues(f);
        outputFolder = [superOutputFolder sprintf('_harmonic_frequency_of_%s_loc_%d_signalGridSize_%d', string(frequencies),signalLocation{1}(1), signalGridSize)];
        disp(outputFolder)
        mkdir(outputFolder);
        for i = 1:length(contrastValues)
            fprintf('starting at %s\n', datetime('now'))
            contrast = contrastValues(i);
            fileName = sprintf('%d_samplesPerClass_freq_%s_contrast_%s_loc_%d_signalGrid_%d',numSamples, join(string(frequencies),'-'), strrep(sprintf("%.12f", contrast), '.', '_'), signalLocation{1}(1), signalGridSize);
            disp(fileName);
            CreateMultipleLocationsGabor_function(frequencies, contrast, signalGridSize, signalLocation, numSamples, fileName, outputFolder)
            fprintf('ending at %s\n', datetime('now'))
        end
    end
end