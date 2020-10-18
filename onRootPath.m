function rootPath = onRootPath()
% Return the path to the root iset directory
%
% This function must reside in the directory at the base of the Optimal
% Networks directory structure.  It is used to determine the location of
% various sub-directories.
% 
% Example:
%   fullfile(onRootPath,'data')

rootPath=which('onRootPath');

rootPath =fileparts(rootPath);

end
