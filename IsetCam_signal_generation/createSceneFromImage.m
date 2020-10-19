function scene = createSceneFromImage(params, path)
% This creates a scene, similar to harmonic, but with an input image as
% pattern, instead of a wave. Instead of returning a wave pattern image as
% template for the scene (one that has a mean of one and the biggest wave
% deviation being 1*params.contrast), this function tries to make the same
% out of an input image. If the input image is of color, the color values
% are being averaged. 


parms = params;
scene.type = 'scene';

scene = sceneSet(scene,'name','fromImage');

if ieNotDefined('wave')
    scene = initDefaultSpectrum(scene,'hyperspectral');
else
    scene = initDefaultSpectrum(scene, 'custom',wave);
end
if checkfields(parms,'ang'), ang = parms.ang; else ang = 0; parms.ang = ang; end
if checkfields(parms,'contrast'), contrast = parms.contrast; else contrast = 1; parms.contrast = contrast; end
if checkfields(parms,'freq'), freq = parms.freq; else freq = 1; parms.freq = freq; end
if checkfields(parms,'ph'), ph = parms.ph; else ph = pi/2; parms.ph = ph; end
if checkfields(parms,'row'), row = parms.row; else row = 64; parms.row = row; end
if checkfields(parms,'col'), col = parms.col; else col = 64; parms.col = col; end
nWave = sceneGet(scene,'nwave');

% other cases, they are simply attached to the global parameters in
% vcSESSION.  We can get them by a getappdata call in here, but not if we
% close the window as part of imageSetHarmonic
if endsWith(path, ".h5")
    % matlab messes the axes up here, because fortran bin encoding of arr
    % instead of C endoding. The updated version h5read() doesn't seem to
    % improve this compatability issue..
    % will transpose here, as python code transposes this back
    % automatically already.
    img = h5read(path, '/face_mat');
    img = transpose(img);
    background = 1;
    img = img*params.contrast+background;
else    
    img = imread(path);
    if length(size(img)) == 3
        img = mean(img,3);
    end
    img = imresize(img, [parms.row, parms.col]);

    % normalize to same range as sine pattern
    img = double(img);
    % don't normalize the mean anymore!
    % img = img - mean(img(:));
    % s = abs(min(img(:)));
    background = 1;
    % simply add contrast to 1.0 background
    img = (img/255)*parms.contrast+background;
end
disp(parms.contrast)


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
return;

