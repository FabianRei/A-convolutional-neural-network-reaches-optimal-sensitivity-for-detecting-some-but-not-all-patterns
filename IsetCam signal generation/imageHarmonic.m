function [img,parms] = imageHarmonic(parms)
%Create a windowed spatial harmonic image 
%
%   [img,parms]  = imgHarmonic(parms)
%
% Creates a windowed, oriented, spatial harmonic
%
%      contrast*window.*cos(2*pi*f([cos(ang)*X + sin(ang)*Y] + ph) + 1
%
% The parameters are  in the structure parms. See the fields in the
% example, below. 
%
% When some of the fields are vectors (freq, contrast, ang, phase) then the
% return produces the sum of these harmonics.  The sum always has a mean of
% 1.  
%
% The Gabor Flag is used to set the window values (a Gaussian).
% When the flag is non-zero, the value specifies the standard deviation
% of the Gaussian as a fraction of the image size.  For example, if the
% image size is 128 and GaborFlag = 0.25, the standard deviation is 32.
%
% Default parameters are applied if parms is not sent in.  You can see the
% defaults by requesting them on return as below.
%
% Example:
%   [img,p] = imageHarmonic;
%   figure; imagesc(img), colormap(gray); axis image
%
%   parms.row = 32; parms.col = 32; parms.contrast = 1; 
%   parms.ph = pi/2; parms.freq = 2; parms.ang = pi/6;
%   parms.GaborFlag = 0.2;
%   [img,p] = imageHarmonic(parms);
%   vcNewGraphWin; imagesc(img), colormap(gray); axis image
%
% Now, for a sum of two harmonics
%   parms.freq(2) = 3; parms.ang(2) = parms.ang(1);
%   parms.contrast(2) = 1; parms.ph(2) = pi/2;
%   [img,p] = imageHarmonic(parms);
%   vcNewGraphWin; imagesc(img), colormap(gray); axis image
%   plot(img(16,:))
% 
% Copyright ImagEval Consultants, LLC, 2003.


if ~exist('parms','var'), parms = []; end

if checkfields(parms,'ang'), ang = parms.ang; else ang = 0; parms.ang = ang; end
if checkfields(parms,'contrast'), contrast = parms.contrast; else contrast = 1; parms.contrast = contrast; end
if checkfields(parms,'freq'), freq = parms.freq; else freq = 1; parms.freq = freq; end
if checkfields(parms,'ph'), ph = parms.ph; else ph = pi/2; parms.ph = ph; end
if checkfields(parms,'row'), row = parms.row; else row = 64; parms.row = row; end
if checkfields(parms,'col'), col = parms.col; else col = 64; parms.col = col; end
if isfield(parms, 'signalGridSize'), dparms.signalGridSize = parms.signalGridSize; else dparms.signalGridSize=1; end
if isfield(parms, 'signalLocation'), dparms.signalLocation = parms.signalLocation; else dparms.signalLocation=1; end

% The Gabor Flag is a non-zero value that specifies the standard deviation
% of the Gaussian as a fraction of the image size.  For example, if the
% image size is 128 and GaborFlag = 0.5, the standard deviation is 64.
if checkfields(parms,'GaborFlag')
    GaborFlag = parms.GaborFlag; 
else
    GaborFlag = 0; 
    parms.GaborFlag = GaborFlag; 
end

% Calculate the harmonic
[X,Y] = meshgrid((0:(col-1))/col,(0:(row-1))/row);

% Calculate the gabor window
if GaborFlag
    hsize = size(X);
    sigma = GaborFlag*min(row,col);
    g = fspecial('gauss',hsize,sigma);
    g = g/max(g(:));
else 
    g = ones(size(X));
end

% harmonics are (1 + sum(cos(2*pi*f*x + ph))
% with the additional issue that X and Y can be at some angle.
img = zeros(size(X));
% the "normal" harmonic has two pixel columns, which are exactly the mean
% value, 1.0. We add a very small shift value two all lines, to avoid this
% problem. if 1.0 values exist, the split over the mean value creates an
% extremely easy problem..
lines = false;
angle = false;
if lines
    ph = ph + 1e-10;
end
if angle
    ang = ang + 1e-10;
end
for ii=1:length(freq)
    img = img + ...
        contrast(ii)*g.*cos( 2*pi*freq(ii)*(cos(ang(ii))*X + sin(ang(ii))*Y) ...
        + ph(ii)) + 1;
end
img = img / length(freq);

gs = parms.signalGridSize;
if gs > 1
    % move the middle part to the specified location. Uncovered parts (by
    % this movement) are set to the default value of 1.
    newImg = ones(size(X));
    signalPart = img-1;
    for jj = 1:length(parms.signalLocation)
        loc = parms.signalLocation(jj);
        newImg = addSignalToLoc(loc, signalPart, newImg, parms);  
    end
    img = newImg;
end


if lines
    maxVal = max(img(:));
    minVal = min(img(:));
    meanVal = mean(img(:));
    if not (length(img(img<=meanVal)) == length(img(img>meanVal)))        
        disp("PROBLEMPROBLEMPROBLEM, use median..");
        meanVal = median(img(:));
        disp(length(img(img<=meanVal)) - length(img(img>meanVal)));
        disp(length(img(img==meanVal)));
        pause(1);
        if not (length(img(img<=meanVal)) == length(img(img>meanVal)))        
            error("median isn't the solution here..");
        end
        if not (length(img(img==meanVal)) == 0)
            error("let's stop here..");
        end
    end
    img(img<=meanVal) = minVal;
    img(img>meanVal) = maxVal;
end

if min(img(:) < 0)
    warning('Harmonics have negative sum, not realizable');
end

% figure; imagesc(img); colormap(gray); axis image

return;
