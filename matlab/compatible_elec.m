function [electrodes, lay] = compatible_elec(labels, layout)
% Creates compitable electrodes with the input layout and with the labels
% labels ->  cell array
% layout -> the path to the layout (e.g. elec1010.lay)

import_fieldtrip();
cfg=[];
cfg.layout= layout;
lay=ft_prepare_layout(cfg);
% figure; ft_plot_layout(lay)
idx = ismember(labels, lay.label)';

if any(idx(:) == 0)
    error('Incompatible labels and layout')
end

electrodes = cell(size(labels));

for k=1:length(labels)
    label = labels{k};
    
    % find the index of the label
    idx = find(strcmp(lay.label, label));
    electrodes{k,1} = lay.pos(idx,1);
    electrodes{k,2} = lay.pos(idx,2);
end

electrodes = cell2mat(electrodes);

end

