function [scene] = sceneCreateMulti(sceneName,varargin)
% SCENECREATEMULTI This function integrates parts of the IsetCam library,
% but then calls imageHarmonicMulti.m instead of imageHarmonic.m.
% imageHarmonicMulti.m can create signals at specified locations.
%
%
% See also:
%   imageHarmonicMulti.m


%% Initial definition
if ieNotDefined('sceneName'), sceneName = 'default'; end
parms = [];  % Returned in some cases, not many.

% Identify the object type
scene.type = 'scene';

sceneName = ieParamFormat(sceneName);
scene.metadata = [];   % Metadata for machine learning apps


switch sceneName
    case {'harmonic','sinusoid'}
        if isempty(varargin)
            [scene,parms] = sceneHarmonic(scene);
        elseif length(varargin) == 1
            parms = varargin{1};
        elseif length(varargin) == 2
            parms = varargin{1};
            wave = varargin{2};
        else
            error('Wrong number of parameters! Input params structure and optional wavelengths.')
        end
end

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize scene geometry, spatial sampling
scene = sceneInitGeometry(scene);
scene = sceneInitSpatial(scene);

% Scenes are initialized to a mean luminance of 100 cd/m2.  The illuminant
% is adjusted so that dividing the radiance (in photons) by the illuminant
% (in photons) produces a peak reflectance of 0.9.
%
% Also, a best guess is made about one known reflectance.
if checkfields(scene,'data','photons') && ~isempty(scene.data.photons)
    
    if isempty(sceneGet(scene,'known reflectance')) && checkfields(scene,'data','photons')
        % Since there is no known reflectance, we set things up here.  If
        % there is one, then stuff must have been set up elsewhere.
        
        % If there is no illuminant yet, create one with the same
        % wavelength samples as the scene and a 100 cd/m2 mean luminance
        if isempty(sceneGet(scene,'illuminant'))
            il = illuminantCreate('equal photons',sceneGet(scene,'wave'),100);
            scene = sceneSet(scene,'illuminant',il);
        end
        
        % There is no known scene reflectance, so we set the peak radiance
        % point as if it has a reflectance of 0.9.
        v = sceneGet(scene,'peak radiance and wave');
        wave = sceneGet(scene,'wave');
        idxWave = find(wave == v(2));
        p = sceneGet(scene,'photons',v(2));
        [tmp,ij] = max2(p); %#ok<ASGLU>
        v = [0.9 ij(1) ij(2) idxWave];
        scene = sceneSet(scene,'known reflectance',v);
    end
    
    % Calculate and store the scene luminance
    luminance = sceneCalculateLuminance(scene);
    scene = sceneSet(scene,'luminance',luminance);
    
    % Adjust the mean illumination level to 100 cd/m2.
    scene = sceneAdjustLuminance(scene,100);
end

end

