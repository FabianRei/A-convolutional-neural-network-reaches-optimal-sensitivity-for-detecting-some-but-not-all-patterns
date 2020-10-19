function outFile = CreateSensorAbsorptionSignalNoiseDataset_function(scanFreq, scanContrast, numSamples, name, outputFolder)
%CREATEDATAFUNCTION Summary of this function goes here
%   Detailed explanation goes here
%% Important prameters to set

% Examples:
%{
scanFreq = 1;
scanContrast = 1;
numSamples = 1;
name = 'foo';
outputFolder = tempdir;
CreateSensorAbsorptionSignalNoiseDataset_function(scanFreq, scanContrast, numSamples, name, outputFolder)
%}

%%
saveName = fullfile(outputFolder, name);
saveFlag = true;

%% Set up the camera sensor

p = harmonicP;
eTime = 1e-3;
fov = 1;

sensorSize = [256 256];
sensor = sensorCreate('monochrome');
sensor = sensorSet(sensor,'size',sensorSize);
sensor = sensorSet(sensor,'exp time',eTime);
sensor = sensorSet(sensor,'noise flag',1);

%% Create data variables

nImages = (length(scanFreq)+1)*length(scanContrast)*numSamples;

% With noise (for each frequency + no signal)
imgNoise = zeros(256,256, nImages);
imgNoiseContrasts = zeros(nImages,1);
imgNoiseFreqs = zeros(nImages,1);

% Without noise (two for each frequency + no signal)
noNoiseImg = zeros(256,256,length(scanFreq)+1);
noNoiseImgFreq = zeros(length(scanFreq)+1, 1);
noNoiseImgContrast = zeros(length(scanFreq)+1, 1);

%% Run a loop over all frequencies (1), all contrast strengths (1) and over the number of samples
k = 1;
% p.row & p.row are not necessarily the resulting image size, as # scene pixesl > # pixels of
% sensor, which captures its pixels..
p.row = 512;
p.col = 512;
for cc = 1:length(scanContrast)
    p.contrast = scanContrast(cc);
    for ff = 0 : length(scanFreq)
        if ff == 0
            p.freq = 0;
        else
            p.freq = scanFreq(ff);
        end
        scene = sceneCreate('harmonic',p);  % sceneWindow(scene);
        % scene = sceneSet(scene,'fov',fov);  
        oi = oiCreate;
        oi = oiCompute(oi,scene);           % oiWindow(oi);
        sensor = sensorSet(sensor,'noise flag',1);
        for nn = 1:numSamples
            fprintf('Generating image: %i \n',k)
            sensor = sensorCompute(sensor,oi);
            imgNoise(:,:,k) = sensorGet(sensor, 'electrons');
            imgNoiseFreqs(k) = p.freq;
            imgNoiseContrasts(k) = p.contrast;
            
            % Calculate without noise
            if nn == 1
                sensor = sensorSet(sensor,'noise flag',-1);  % The proper noise flag is in question.  Maybe -1 is better.
                sensor = sensorCompute(sensor,oi);
                pixel = sensorGet(sensor,'pixel');
                meanVal = sensorGet(sensor,'volts')/pixelGet(pixel,'conversionGain');
                % noNoiseImg(:,:,ff+1) = sensorGet(sensor, 'electrons');
                noNoiseImg(:,:,ff+1) = meanVal;
                noNoiseImgFreq(ff+1) = p.freq;
                noNoiseImgContrast(ff+1) = p.contrast;
                sensor = sensorSet(sensor,'noise flag',1);
            end
            
            k = k+1;

        end
    end
end


%% Crop
imgNoise = imgNoise(11:248, 11:248,:);
noNoiseImg = noNoiseImg(11:248, 11:248,:);
%% Save everything

if(saveFlag)
    outFile = sprintf('%s.h5',saveName);
    % currDate = datestr(now,'mm-dd-yy_HH_MM');
    hdf5write(outFile, ...
        'imgNoise', imgNoise, ...
        'imgNoiseFreqs', imgNoiseFreqs, ...
        'imgNoiseContrasts', imgNoiseContrasts, ...
        'noNoiseImg', noNoiseImg, ...
        'noNoiseImgFreq', noNoiseImgFreq, ...
        'noNoiseImgContrast', noNoiseImgContrast);
        
end
end

