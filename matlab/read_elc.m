function [sensors,sensors_label] = read_elc(filename)
filetype_pos = findstr('.',filename);
filetype_pos = filetype_pos(end);
filetype = filename(filetype_pos+1:end);

switch filetype
    case 'elc'
        fid = fopen(filename,'r');
        pos_flag = 1;
        
        while(pos_flag)
            InputText =textscan(fid,'%s',1,'delimiter','\n');
            
            if(strfind(InputText{1}{1},'NumberPositions=') == 1)
                N_sens = str2double(deblank(strrep(InputText{1}{1},'NumberPositions=','')));
            end
            
            if(strfind(InputText{1}{1},'Positions') == 1)
                pos_flag = 0;
            end
            
        end
        
        sensors = zeros(N_sens,3);
        
        for i = 1:N_sens
            InputText = textscan(fid,'%f %f %f',1,'delimiter','\n');
            sensors(i,:) = cell2mat(InputText);
        end
        
        textscan(fid,'%s',1,'delimiter','\n'); %dummy read for moving the pointer
        
        sensors_label = cell(N_sens,1); 
        for i = 1:N_sens
            InputText = textscan(fid,'%s',1,'delimiter','\n');
            sensors_label{i,1} = char(InputText{1});
        end
        
        fclose(fid);
    case 'pts'
        [x,y,z,N] = read_pts(filename);
        sensors = [x,y,z];
        sensors_label=[];
end