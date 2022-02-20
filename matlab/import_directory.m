function [] = import_directory(path)
% Imports path to the current folder


maindir = matlab.desktop.editor.getActiveFilename; 
mydir  = maindir;
idcs   = strfind(mydir,'/');
newdir = mydir(1:idcs(end)-1);
maindir = newdir; % keep main path

restoredefaultpath % restore default folder for matlab    

% set up the path of fieldtrip
cd(path)
addpath(pwd)
cd(maindir)

end