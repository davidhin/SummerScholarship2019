% Start socket
global t
t = tcpip('129.127.10.18', 8222, 'NetworkRole', 'client');
t.OutputBufferSize = 2048;
t.InputBufferSize = 1; % Change this
fopen(t);
return
