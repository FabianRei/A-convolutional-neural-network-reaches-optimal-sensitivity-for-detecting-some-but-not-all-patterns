
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
%    Used to create the "Faces" signals
%
% See Also:
%    CreateConeAbsorptionSignalNoiseDataset_function

% Values to set

% imageDir = fullfile(onRootPath,'data','faces','single_faces');
% imageDir = fullfile(onRootPath,'data','automata');
%
% If you need a local directory to write out a temporary file or sub
% directory, use this
%
%    localDir = fullfile(onRootPath,'local');
%
% You can put stuff in there and git will not pay any attention to it.
%

imageDir = 'C:\Users\Fabian\Documents\data\faces\single_faces\';
imageDir = 'C:\Users\Fabian\Documents\data\windows2rsync\windows_data\redo_automaton\matlab_templates\';
imageNames = dir([imageDir '*.png']);
imageNames = dir([imageDir '*.h5']);
imagePaths = fullfile(imageDir, {imageNames.name});
numSamples = 2;
frequencies = 1;
% contrastValues = [0.0003, 0.0002, 0.0004];
contrastValues = logspace(-5, -1.7, 12);
% contrastFreqPairs = [];
shiftValues = 0;

% for i = 1:length(contrastValues)
%     for j = 1:length(frequencies)       
%         contrast = contrastValues(i);
%         freq = frequencies(j);
%         contrastFreqPairs = cat(1, contrastFreqPairs, [contrast, freq]);
%     end
% end

for i = 1:length(imagePaths)
    imagePath = imagePaths(i);
    imagePath = imagePath{1};
    [~,fname,~] = fileparts(imagePath);
    % outputFolder = ['/share/wandell/data/reith/circle_fun/h5_data/' fname '/'];
    % outputFolder = fullfile(onRootPath,'local','circle_fun','hg_data',fname);
    outputFolder = ['C:\Users\Fabian\Documents\data\windows2rsync\windows_data\redo_automaton\matlab_contrastsiii\' fname '\'];
    status = mkdir(outputFolder);
    for j = 1:length(contrastValues)
        % This creates the resulting datasets
        fprintf('starting at %s\n', datetime('now'))
        contrast = contrastValues(j);
        shiftValue = shiftValues;
        fileName = sprintf('%d_samplesPerClass_freq_%s_contrast_%s_image_%s',numSamples, join(string(frequencies),'-'), strrep(sprintf("%.12f", contrast), '.', '_'), fname);
        disp(fileName);
        CreateContrastDatasetFromImage_function(frequencies, contrast, shiftValue, numSamples, fileName, outputFolder, imagePath)
        fprintf('ending at %s\n', datetime('now'))
    end
end