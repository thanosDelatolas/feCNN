function [] = import_fieldtrip()
% Imports fieldtrip to the folder


maindir = matlab.desktop.editor.getActiveFilename; 
mydir  = maindir;
idcs   = strfind(mydir,'/');
newdir = mydir(1:idcs(end)-1);
maindir = newdir; % keep main path

restoredefaultpath % restore default folder for matlab    

% set up the path of fieldtrip
cd('/home/thanos/fieldtrip')
addpath(pwd)
ft_defaults
cd(maindir)

end

