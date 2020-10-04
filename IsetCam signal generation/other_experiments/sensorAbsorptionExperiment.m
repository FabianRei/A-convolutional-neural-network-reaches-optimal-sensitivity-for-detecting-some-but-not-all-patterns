

%% Calculate the ideal
fov = 5;
p = harmonicP;
p.freq = 0;
resolution = [256 256];
scene = sceneCreate('harmonic',p);
% scene = sceneSet(scene, 'size', resolution);
% scene = sceneSet(scene,'fov',fov);

oi = oiCreate;
oi = oiCompute(oi,scene);

% This is the part where we calculate the noisy samples.
sensor = sensorCreate('monochrome');
sensor = sensorSet(sensor,'size',resolution);

% sensor = sensorSetSizeToFOV(sensor,fov);
eTime = 1e-3;
sensor = sensorSet(sensor,'exp time',eTime);

% Create sample without noise
sensor = sensorSet(sensor,'noise flag',0);
sensor = sensorCompute(sensor,oi);
noNoiseSample = sensorGet(sensor, 'electrons');
disp(size(noNoiseSample))
imagesc(noNoiseSample)

sensor = sensorSet(sensor,'noise flag',1);
sensor = sensorCompute(sensor,oi);
noiseSample = sensorGet(sensor, 'electrons');
figure;
imagesc(noiseSample)




