function [ ret ] = decode( t, data ) 
%% Write to buffer
data = strjoin(arrayfun(@(x) num2str(x),data,'UniformOutput',false),',');
fwrite(t, data);

%% Read from buffer
result_img = fread(t);
load(char(strcat('../Autoencoder/result_',string(t.RemotePort),'.mat')),'result_img');
ret = result_img;
% received_data2_char = char(received_data2);
% result = str2num(received_data2_char');

%% Show Picture
% result_img = reshape(result,[3 128 128]);
% result_img = permute(result_img, [3 2 1]);
end
