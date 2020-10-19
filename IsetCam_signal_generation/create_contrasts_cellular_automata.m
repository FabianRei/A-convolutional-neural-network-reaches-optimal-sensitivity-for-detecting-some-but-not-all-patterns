
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
%    Used to create the "Automata" signals
%
% See Also:
%    CreateConeAbsorptionSignalNoiseDataset_function

% Values to set

% imageDir = fullfile(onRootPath,'data','faces');
% imageDir = fullfile(onRootPath,'data','automata');
%
% If you need a local directory to write out a temporary file or sub
% directory, use this
%
%    localDir = fullfile(onRootPath,'local');
%
% You can put stuff in there and git will not pay any attention to it.
%
ieInit

imageDirAutomata = fullfile(onRootPath, 'data', 'automata');

% Choose whether automaton or face signals shall be generated:
imageDir = imageDirFaces;

imageNames = dir(fullfile(imageDir, '*.h5'));
imagePaths = fullfile(imageDir, {imageNames.name});
numSamples = 2;
frequencies = 1;
contrastValues = logspace(-5, -1.7, 12);
shiftValues = 0;

for i = 1:length(imagePaths)
    imagePath = imagePaths(i);
    imagePath = imagePath{1};
    [~,fname,~] = fileparts(imagePath);
    outputFolder = fullfile(onRootPath, 'local', 'automata_contrasts', fname);
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