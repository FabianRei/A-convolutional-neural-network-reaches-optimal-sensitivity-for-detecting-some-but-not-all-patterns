function newImg = addSignalToLoc(loc, signalPart, newImg, parms)
% Translates a signal, usually in the middle of the created scene,
% to a location specified in "loc". 
%
% Synopsis
%   newImg = addSignalToLoc(loc, signalPart, newImg, parms)
%
% Input
%   signalPart:  The signals that will be placed in the new iamge
%   loc:  Location where the new signal should be placed
%   newImg:  The image where you will add the signal
%   parms:   Struct that must include signalGridSize and gridZoom
%  
% Output
%   newImg:
%
% Description:
%
%  if signalGridSize is 3, the grid size corresponds to 3x3. 
%  location 2 would mean first row column 2
%  location 5 would  mean second row column 2 and so on.
%
% See also
%

% extract variables for shorter name
gs  = parms.signalGridSize;
row = parms.row;
col = parms.col;

gZoom  = parms.gridZoom;

% calculate the padding corresponding to the zoom chosen
paddingLeft = round((col-col/gZoom)/2);
paddingUp   = round((row-row/gZoom)/2);

% this calculates the coordinates for the center cut out of the signal. To
% move the signal to a place in the grid, it is not cut out and pasted,
% though. The whole signal is moved to the chosen location on the defined
% grid. If two big signals are put to grid locations next to each other, it
% might therefore be the case, that they overlap into their corresponding
% grid location rectangle
cuts = [round((((gs-1)/2)/gs)*row/gZoom  + 1 + paddingUp) ...
     round(((((gs-1)/2)+1)/gs)*row/gZoom + paddingUp) ...
     round((((gs-1)/2)/gs)*col/gZoom + 1 + paddingUp) ...
     round(((((gs-1)/2)+1)/gs)*col/gZoom + paddingUp)];

height = cuts(2)-cuts(1)+1;
width  = cuts(4)-cuts(3)+1;

% row/column location for the chosen grid location
% upper left part, where the signal will get extracted from
colGridLoc = mod(loc-1, gs);
rowGridLoc = (loc-1-colGridLoc)/gs;
upperLeftLoc = [rowGridLoc*width+1+paddingLeft colGridLoc*height+1+paddingUp];
upperLeftSignal = [cuts(1) cuts(3)];

% This is where the signal gets cut out. It is determined, how much of the
% the signal part can be copied and pasted (if moved to the upper left, for
% esample, parts left from or above the center cutout start cannot be 
% transfered. All parts to the right or bottom can, though)
upperCut = min(upperLeftLoc(1), upperLeftSignal(1))-1;
lowerCut = min(row-upperLeftLoc(1), row-upperLeftSignal(1));
leftCut = min(upperLeftLoc(2), upperLeftSignal(2))-1;
rightCut = min(col-upperLeftLoc(2), col-upperLeftSignal(2));

% cutting out, combining and creation of a new image
tempImg = newImg((upperLeftLoc(1)-upperCut):(upperLeftLoc(1)+lowerCut), ...
    (upperLeftLoc(2)-leftCut):(upperLeftLoc(2)+rightCut));

tempSignal = signalPart((upperLeftSignal(1)-upperCut):(upperLeftSignal(1)+lowerCut), ...
    (upperLeftSignal(2)-leftCut):(upperLeftSignal(2)+rightCut));

result = tempImg + tempSignal;
newImg(upperLeftLoc(1)-upperCut:upperLeftLoc(1)+lowerCut, ...
    upperLeftLoc(2)-leftCut:upperLeftLoc(2)+rightCut) = result;

end
