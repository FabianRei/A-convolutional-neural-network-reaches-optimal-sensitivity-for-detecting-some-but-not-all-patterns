function [img] = createMultipleSignals(parms,img, X)
% Add created signals to multiple locations
%   Detailed explanation goes here
    newImg = ones(size(X));
    signalPart = img-1;
    for jj = 1:length(parms.signalLocation)
        loc = parms.signalLocation(jj);
        newImg = addSignalToLoc(loc, signalPart, newImg, parms);  
    end
    img = newImg;
end

