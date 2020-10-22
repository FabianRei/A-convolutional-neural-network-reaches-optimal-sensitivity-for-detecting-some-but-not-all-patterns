function [outputArg1,outputArg2] = sceneCreateMulti(sceneName,varArgIn)
%SCENECREATEMULTI Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = sceneName;
outputArg2 = varArgIn;


%% Initial definition
if ieNotDefined('sceneName'), sceneName = 'default'; end
parms = [];  % Returned in some cases, not many.

% Identify the object type
scene.type = 'scene';

sceneName = ieParamFormat(sceneName);
scene.metadata = [];   % Metadata for machine learning apps


%% Handle the Macbeth parameter cases here
if strncmp(sceneName,'macbeth',5) || ...
        strcmp(sceneName,'default') || ...
    strcmp(sceneName,'empty')
    patchSize = 16; wave = 400:10:700; surfaceFile = 'macbethChart.mat';
    if ~isempty(varargin), patchSize = varargin{1}; end  % pixels per patch
    if length(varargin) > 1, wave = varargin{2}; end     % wave
    if length(varargin) > 2, surfaceFile = varargin{3}; end % Reflectances
end

switch sceneName
    case {'harmonic','sinusoid'}
        if isempty(varargin)
            [scene,parms] = sceneHarmonic(scene);
        elseif length(varargin) == 1
            parms = varargin{1};
            [scene,parms] = sceneHarmonic(scene,parms);
        elseif length(varargin) == 2
            parms = varargin{1};
            wave = varargin{2};
            [scene,parms] = sceneHarmonic(scene,parms, wave);
        else
            error('Wrong number of parameters! Input params structure and optional wavelengths.')
        end
end

function [scene,p] = sceneHarmonic(scene,parms, wave)
%% Create a scene of a (windowed) harmonic function.
%
% Harmonic parameters are: parms.freq, parms.row, parms.col, parms.ang
% parms.ph, parms.contrast
%
% Missing default parameters are supplied by imageHarmonic.
%
% The frequency is with respect to the image (cyces/image).  To determine
% cycles/deg, use cpd: freq/sceneGet(scene,'fov');
%

scene = sceneSet(scene,'name','harmonic');

if ieNotDefined('wave')
    scene = initDefaultSpectrum(scene,'hyperspectral');
else
    scene = initDefaultSpectrum(scene, 'custom',wave);
end

nWave = sceneGet(scene,'nwave');

% TODO: Adjust pass the parameters back from the imgHarmonic window. In
% other cases, they are simply attached to the global parameters in
% vcSESSION.  We can get them by a getappdata call in here, but not if we
% close the window as part of imageSetHarmonic
if ieNotDefined('parms')
    global parms; %#ok<REDEF>
    h   = imageSetHarmonic; waitfor(h);
    img = imageHarmonic(parms);
    p   = parms;
    clear parms;
else
    [img,p] = imageHarmonic(parms);
end

% To reduce rounding error problems for large dynamic range, we set the
% lowest value to something slightly more than zero.  This is due to the
% ieCompressData scheme.
img(img==0) = 1e-4;
img   = img/(2*max(img(:)));    % Forces mean reflectance to 25% gray

% Mean illuminant at 100 cd
wave = sceneGet(scene,'wave');
il = illuminantCreate('equal photons',wave,100);
scene = sceneSet(scene,'illuminant',il);

img = repmat(img,[1,1,nWave]);
[img,r,c] = RGB2XWFormat(img);
illP = illuminantGet(il,'photons');
img = img*diag(illP);
img = XW2RGBFormat(img,r,c);
scene = sceneSet(scene,'photons',img);

return;

end

