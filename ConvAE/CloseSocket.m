% Close socket
data = 'disconnect'
fwrite(t, data)
fclose(t)
delete(t);
clear t
echotcpip('off');
return
