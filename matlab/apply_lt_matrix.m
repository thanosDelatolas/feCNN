function source_space_out=apply_lt_matrix(file_in, source_space)
% This function applies the linear transformation matrix (generated by fsl)
% and applies it to the source space. The source space must be a matrix
% with dimensions: dipoles x 3.

fid = fopen(file_in);
tline = fgetl(fid);

tf_matrix = zeros(4);
line=1;
while ischar(tline)
    line_array = str2double(strsplit(tline));
    tf_matrix(line,:)  = line_array(1:end-1);
    tline = fgetl(fid);
    line = line+1;
end
fclose(fid);

source_space_out = zeros(size(source_space));

R = tf_matrix(1:3,1:3);
t = tf_matrix(1:3,end);

for i=1:size(source_space,1)
  source_space_out(i,:) = (R*source_space(i,:)') + t;
end

end