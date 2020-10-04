function CreateContrastDatasetFromImage_function(scanFreq, scanContrast, shiftValues, numSamples, name, outputFolder, imagePath)
%CREATEDATAFUNCTION Summary of this function goes here
%   Detailed explanation goes here
%% Important prameters to set

saveName = fullfile(outputFolder, name);
saveFlag = true;

%% Set up the camera sensor

resolution = [256 256];
p = harmonicP;
p.col = 256;
p.row = 256;
eTime = 1e-3;
% fov = 5;
sensor = sensorCreate('monochrome');
sensor = sensorSet(sensor,'size',resolution);
sensor = sensorSet(sensor,'exp time',eTime);
sensor = sensorSet(sensor,'noise flag',1);

%% Create data variables

nImages = (length(scanFreq)+1)*length(scanContrast)*length(shiftValues)*numSamples;

% With noise (for each frequency + no signal)
imgNoise = zeros(256,256, nImages);
imgNoiseContrasts = zeros(nImages,1);
imgNoiseFreqs = zeros(nImages,1);
imgNoisePhases = zeros(nImages,1);


% Without noise (two for each frequency + no signal)
noNoiseImg = zeros(256,256,length(scanFreq)*length(shiftValues)+1);
noNoiseImgFreq = zeros(length(scanFreq)*length(shiftValues)+1, 1);
noNoiseImgContrast = zeros(length(scanFreq)*length(shiftValues)+1, 1);
noNoiseImgPhase = zeros(length(scanFreq)*length(shiftValues)+1, 1);

%% Run a loop over all frequencies (1), all contrast strengths (1) and over the number of samples
k = 1;
originalPhase = p.ph;
for cc = 0:length(scanContrast)
    if cc == 0
        p.contrast = 0;
    else
        p.contrast = scanContrast(cc);
    end
    for ff = 1 : length(scanFreq)
        if ff == 0
            p.freq = 0;
        else
            p.freq = scanFreq(ff);
        end
        for sh = 1:length(shiftValues)
            if sh >= 1
                p.ph = originalPhase + shiftValues(sh)*(pi/3000);
                % disp(originalPhase);
                % disp(p.ph);
            else
                p.ph = originalPhase;
            end
            scene = createSceneFromImage(p, imagePath);
            % adjust scene fov to sensor fov
            scene = sceneSet(scene, 'fov', sensorGet(sensor, 'fov'));
            oi = oiCreate;
            oi = oiCompute(oi,scene);
            sensor = sensorSet(sensor,'noise flag',1);
            for nn = 1:numSamples
                fprintf('Generating image: %i \n',k)
                sensor = sensorCompute(sensor,oi);
                imgNoise(:,:,k) = sensorGet(sensor, 'electrons');
                imgNoiseFreqs(k) = p.freq;
                imgNoiseContrasts(k) = p.contrast;
                imgNoisePhases(k) = p.ph;

                % Calculate without noise
                if nn == 1
                    sensor = sensorSet(sensor,'noise flag',0);
                    sensor = sensorCompute(sensor,oi);
                    % electrons are integers. We need mean electron
                    % values..
                    % noNoiseImg(:,:,cc+1) = sensorGet(sensor, 'electrons');
                    pixel = sensorGet(sensor,'pixel');
                    meanVal = sensorGet(sensor,'volts')/pixelGet(pixel,'conversionGain');
                    noNoiseImg(:,:,cc+1) = meanVal;
                    noNoiseImgFreq(1+cc) = p.freq;
                    noNoiseImgContrast(1+cc) = p.contrast;
                    noNoiseImgPhase(1+cc) = p.ph-originalPhase;
                    sensor = sensorSet(sensor,'noise flag',1);
                end

                k = k+1;
            end
        end
    end
end


%% Crop
% imgNoise = imgNoise(11:248, 11:248,:);
% noNoiseImg = noNoiseImg(11:248, 11:248,:);
%% Save everything

if(saveFlag)
    % currDate = datestr(now,'mm-dd-yy_HH_MM');
    hdf5write(sprintf('%s.h5',saveName), ...
        'imgNoise', imgNoise, ...
        'imgNoiseFreqs', imgNoiseFreqs, ...
        'imgNoiseContrasts', imgNoiseContrasts, ...
        'imgNoisePhases', imgNoisePhases, ...
        'noNoiseImg', noNoiseImg, ...
        'noNoiseImgFreq', noNoiseImgFreq, ...
        'noNoiseImgContrast', noNoiseImgContrast, ...
        'noNoiseImgPhase', sprintfc('%.15f', noNoiseImgPhase));
        
%     save(sprintf('%s.mat',saveName),...
%         'imgNoise',...
%         'imgNoiseLabels',...
%         'meanImg',...
%         'meanImgLabels',...
%         'imgContrasts',...
%         'imgFreqs');
end
end

