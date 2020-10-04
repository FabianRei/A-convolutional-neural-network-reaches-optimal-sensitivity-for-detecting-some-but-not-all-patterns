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
%    a size of 249x249 to a size of 227x227. This does not affect the 
%    created signal pattern itself, as it is pretty much only visible
%    within the cropped 227x227 center. 
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
superOutputFolder = '/share/wandell/data/reith/redo_experiments/sensor_harmonic_phase_shift/';
superOutputFolder = 'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\viz_templates\';
mkdir(superOutputFolder);
numSamples = 1;
% frequencies = 1;
% frequencyValues = round(logspace(log10(1), log10(200), 12));
frequencyValues = round(logspace(log10(1), log10(100), 8));
% contrastValues = [0.0003, 0.0002, 0.0004];
contrastValues = 0.1;
contrastFreqPairs = [];
shiftValues = logspace(-4, -1, 12);
% lower values for lines, as above shiftvalues are equivalent to 100%
% shiftValues = logspace(-9, -4, 12);

% for i = 1:length(contrastValues)
%     for j = 1:length(frequencies)       
%         contrast = contrastValues(i);
%         freq = frequencies(j);
%         contrastFreqPairs = cat(1, contrastFreqPairs, [contrast, freq]);
%     end
% end

% This creates the resulting datasets
for f = 1:length(frequencyValues)
    frequencies = frequencyValues(f);
    outputFolder = [superOutputFolder sprintf('harmonic_frequency_of_%s', string(frequencies))];
    mkdir(outputFolder);
    if frequencies == 200
        shiftVals = logspace(-4, -1, 12);
        shiftGrowth = shiftVals(12)/shiftVals(11);
        shiftValues = [shiftVals shiftVals(12)*shiftGrowth shiftVals(12)*shiftGrowth*shiftGrowth];
    end
    for i = 1:length(shiftValues)
        if i > 12
            disp('working overtime here!')
        end
        fprintf('starting at %s\n', datetime('now'))
        contrast = contrastValues;
        shiftValue = shiftValues(i);
        fileName = sprintf('%d_samplesPerClass_freq_%s_contrast_%s_shift_%s_pi',numSamples, join(string(frequencies),'-'), strrep(sprintf("%.2f", contrast), '.', '_'), strrep(sprintf("%.9f", shiftValue), '.', '_'));
        disp(fileName);
        CreateSensorAbsorptionSignalShiftDataset_function(frequencies, contrast, shiftValue, numSamples, fileName, outputFolder)
        fprintf('ending at %s\n', datetime('now'))
    end
end