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
%    Used to create harmonic signals, used in figure 2 and 6 of our paper.
%
% See also:
%    CreateSensorAbsorptionsSignalNoiseDataset_function
% 
% Values to set:
%   numSamples: Number of samples created with Poisson noise. Generally not
%   needed as Poisson noise samples are being generated during the training
%   process
%   frequencyValues: Various frequencies for which the harmonic signal will
%   be created for
%   superOutputFolder: Folder in whicch the generated harmonic frequencies,
%   as well as their various contrasts, will be saved to
%   
%

%%
ieInit

% Values to set
superOutputFolder = fullfile(onRootPath, 'local', 'harmonic_contrasts');
if ~exist(superOutputFolder,'dir'), mkdir(superOutputFolder); end

%%
numSamples = 1;
contrastValues = logspace(-5, -1.7, 12);
contrastFreqPairs = [];
frequencyValues = round(logspace(log10(1), log10(100), 8));

% This creates the resulting datasets
for f = 1:length(frequencyValues)
    frequencies = frequencyValues(f);
    outputFolder = [superOutputFolder sprintf('_harmonic_frequency_of_%s', string(frequencies))];
    mkdir(outputFolder);
    for i = 1:length(contrastValues)
        fprintf('starting at %s\n', datetime('now'))
        contrast = contrastValues(i);
        fileName = sprintf('%d_samplesPerClass_freq_%s_contrast_%s',numSamples, join(string(frequencies),'-'), strrep(sprintf("%.12f", contrast), '.', '_'));
        disp(fileName);
        CreateSensorAbsorptionSignalNoiseDataset_function(frequencies, contrast, numSamples, fileName, outputFolder)
        fprintf('ending at %s\n', datetime('now'))
    end
end

%% END