function CreateConeAbsorptionSignalNoiseDataset_function(scanFreq, scanContrast, numSamples, name, outputFolder)
%CREATEDATAFUNCTION Summary of this function goes here
%   Detailed explanation goes here
%% Important prameters to set

saveName = fullfile(outputFolder, name);

%% Parameter initialization
sFreq         = 4;
nPCs          = 2;
fov           = 1;
sContrast     = 1;

% Save the generated data?
saveFlag = true;
% accuracy = zeros(numel(scanFreq), numel(scanContrast));

%% Set up the stimulus parameters
clear hparams
hparams(2)           = harmonicP;
hparams(2).freq      = sFreq;  % Set the Frequency
hparams(2).contrast  = sContrast;
hparams(1)           = hparams(2);
hparams(1).contrast  = 0;

sparams.fov = 1.5;

nTimeSteps = 20;
stimWeights = ones(1, nTimeSteps);

%% Set up cone mosaic parameters

integrationTime = 0.005;
sampleTimes = ((1:nTimeSteps) - 1) * integrationTime;   % Five ms integration time
nTrials    = 100;

%% Randomly generate parameters:
% scanFreq
% scanContrast
% aberrations
rng(1); % Set the random seed


nImages = (length(scanFreq)+1)*length(scanContrast)*numSamples;

% Aberrations
measPupilMM = 4.5; % 4.5 mm pupil size (Thibos data)
calcPupilMM = 3; % 3 mm pupil size (what we want to calculate with)
zCoeffs = VirtualEyes(nImages,measPupilMM);

%% Generate images

% With noise (for each frequency + no signal (=0))
imgNoise = zeros(249,249, nImages);
imgNoiseLabels = zeros(nImages,1);

% Without noise (two for each frequency + no signal)
meanImg = zeros(249,249,length(scanFreq)+1);
meanImgLabels = zeros(length(scanFreq)+1, 1);

% Contrast
imgContrasts = zeros(nImages,1);

% Frequencies
imgFreqs = zeros(nImages,1);

k = 1;
%% One cone mosaic for all data
cm = coneMosaic;

%% Run a loop over all frequencies (1), all contrast strengths (1) and over the number of samples
for cc = 1:length(scanContrast)
    for ff = 0 : length(scanFreq)
        if ff == 0
            % 0 means we create a non signal class
            hparams(2).freq      = scanFreq(1);  % Frequency is irrelevant here
            hparams(2).contrast  = 0.0;
        else
            hparams(2).freq      = scanFreq(ff);  % Set the Frequency
            hparams(2).contrast  = scanContrast(cc);
        end
        for nn = 1:numSamples
            fprintf('Generating image: %i \n',k)

       
            %% Create the oi with aberrations
            
            % Default
            oi = oiCreate('wvf human');
            test = oiCreate('human', 'harmonic', 'blend', stimWeights, ...
                'testParameters', hparams, 'sceneParameters', sparams);
            %% Create the OIS
            
            ois = oisCreate('harmonic', 'blend', stimWeights, ...
                'testParameters', hparams, 'sceneParameters', sparams,...
                'oi',oi);
            % ois.visualize('movie illuminance');
            
            %% Set the coneMosaic parameters according to the OI
            
            cm.integrationTime = ois.timeStep;
            
            % Make the cm smaller than the oi size, but never smaller than 0.2 deg
            fovDegs = max(oiGet(ois.oiFixed,'fov') - 0.2, 0.2);  % Degrees
            cm.setSizeToFOV(fovDegs);
            
            %% Calculate the absorption template for the high contrast example of the stimulus
            
            cm.noiseFlag = 'random';
            imgNoise(:,:,k) = mean(squeeze(cm.compute(ois)), 3);
            imgNoiseLabels(k) = ff;
            imgContrasts(k) = hparams(2).contrast;
            if ff == 0
                imgFreqs(k) = 0;
            else
                imgFreqs(k) = hparams(2).freq;
            end
            
            % Calculate without noise
            if nn == 1
                cm.noiseFlag = 'none';
                meanImg(:,:,ff+1) = mean(squeeze(cm.compute(ois)), 3);
                meanImgLabels(ff+1) = ff;
            end
            
            k = k+1;
            

        end
    end
end


%% Exam the mean cone absorption
% meanImgNoiseStimulus = mean(imgNoiseStimulus, 3);
% meanImgNoNoiseStimulus = mean(imgNoNoiseStimulus, 3);
% meanImgNoiseNoStimulus = mean(imgNoiseNoStimulus, 3);
% meanImgNoNoiseNoStimulus = mean(imgNoNoiseNoStimulus, 3);
% 
% figure(1); imagesc(meanImgNoiseStimulus, [6 8]); colorbar; title('meanImgNoiseStimulus');
% figure(2); imagesc(meanImgNoNoiseStimulus, [6 8]); colorbar; title('meanImgNoNoiseStimulus');
% 
% figure(3); imagesc(meanImgNoiseNoStimulus, [6 8]); colorbar; title('meanImgNoiseNoStimulus');
% figure(4); imagesc(meanImgNoNoiseNoStimulus, [6 8]); colorbar; title('meanImgNoNoiseNoStimulus');
% 
% figure(5); imagesc(meanImgNoiseStimulus - meanImgNoiseNoStimulus, [-5 5]); colorbar; title('mean Noise sub');
% figure(6); imagesc(meanImgNoNoiseStimulus - meanImgNoNoiseNoStimulus, [-0.02 0.02]);colorbar; title('mean No Noise sub');
%% Crop
imgNoise = imgNoise(14:237,14:237,:);
meanImg = meanImg(14:237,14:237,:);
%% Save everything

if(saveFlag)
    % currDate = datestr(now,'mm-dd-yy_HH_MM');
    hdf5write(sprintf('%s.h5',saveName), ...
        'imgNoise', imgNoise, ...
        'imgNoiseLabels', imgNoiseLabels, ...
        'meanImg', meanImg, ...
        'meanImgLabels', meanImgLabels, ...
        'imgContrasts', imgContrasts, ...
        'imgFreqs', imgFreqs);
        
%     save(sprintf('%s.mat',saveName),...
%         'imgNoise',...
%         'imgNoiseLabels',...
%         'meanImg',...
%         'meanImgLabels',...
%         'imgContrasts',...
%         'imgFreqs');
end
end

