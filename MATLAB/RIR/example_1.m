close all
clear
clc

c = 340;                    % Sound velocity (m/s)
fs = 8000;                 % Sample frequency (samples/s)
r = [2 1.5 2];              % Receiver position [x y z] (m)
s = [2 3.5 2];              % Source position [x y z] (m)
L = [5 4 6];                % Room dimensions [x y z] (m)
beta = 0.4;                 % Reverberation time (s)
n = 4096;                   % Number of samples

t = linspace(0,n/fs,n);

h = rir_generator(c, fs, r, s, L, beta, n);

figure
plot(t,h)
xlabel('t (seconds)')
ylabel('amplitude')
title('Room Impulse Response')
