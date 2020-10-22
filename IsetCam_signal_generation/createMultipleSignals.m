function [newImg] = createMultipleSignals(img,parms)
% Add created signals to multiple locations
%
% Synopsis
%   [newImg] = createMultipleSignals(img,parms)
%
% Brief description
%  Replicates a harmonic image (img) to multiple locations in a new
%  image.
%
%  Expected: img is a harmonic with a mean of 1.
%  Output img has the harmonic replicated at signalLocal positions.
%
%  parms must have .signalLocation, .gridSize, .row, .col
%
% See also
%   imageHarmonic

newImg = ones(size(img));  % 
signalPart = img-1;   % img is a harmonic with mean 1.  We get the contrast.
for jj = 1:length(parms.signalLocation)
    loc = parms.signalLocation(jj);
    newImg = addSignalToLoc(loc, signalPart, newImg, parms);
end

end

