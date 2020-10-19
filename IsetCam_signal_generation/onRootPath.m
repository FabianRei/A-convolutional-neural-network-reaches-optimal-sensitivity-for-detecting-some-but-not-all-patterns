function rootPath = onRootPath()
% Return the path to the root iset directory
%
% This function assumes that its file lies in the IsetCam_signal_generation
% directory. It will return the path of the repository, its parent dir.
% 
% Example:
%   fullfile(onRootPath,'data')

rootPath = which('onRootPath');

rootPath = fileparts(rootPath);
rootPath = fileparts(rootPath);

end
