% Start socket
function [ ] = StartSocket(portNum)
    global t
    if nargin == 1
        t = tcpip('129.127.10.18', portNum, 'NetworkRole', 'client');
        t.OutputBufferSize = 2048;
        t.InputBufferSize = 1; % Change this
        fopen(t);
    else if nargin == 0
        t = tcpip('129.127.10.18', 8220, 'NetworkRole', 'client');
        t.OutputBufferSize = 2048;
        t.InputBufferSize = 1; % Change this
        fopen(t);
    end
    return
end
