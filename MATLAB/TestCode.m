clc
clear
close all

%Step 1: get room impulse response
M = 4001;
fs = 8000;    %sampling rate (voice frequency ranges from 300 - 3400 Hz)
[B,A] = cheby2(4,20,[0.1 0.7]);
Hd = dfilt.df2t([zeros(1,6) B],A);
hFVT = fvtool(Hd);  % Analyze the filter
set(hFVT, 'Color', [1 1 1])

H = filter(Hd,log(0.99*rand(1,M)+0.01).*sign(randn(1,M)).*exp(-0.002*(1:M)));
H = H/norm(H)*4;    % Room Impulse Response

figure

plot(0:1/fs:0.5,H);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Room Impulse Response');
set(gcf, 'Color', [1 1 1])

%step 2: get sample far-end(echoed signal)and filter it using room impulse response
mySig = audioread('Hello_Echoe.wav');
p8 = audioplayer(mySig,fs);
mySig = mySig(1:length(mySig));
dhat = filter(H,1,mySig);  %filter with room impulse response

figure

t = mySig/fs;
%plot(t,dhat);
plot(mySig);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Hello Echo Signal');
set(gcf, 'Color', [1 1 1])
pause(5)                        %wait for key press
disp('Playing Hello Echo Signal') %dhat is far-end
p8 = audioplayer(dhat,fs);
playblocking(p8);

d=dhat;

%step 3: pass the echoed signal to LMS filter
mu = 0.001;                     %sys parameters for LMS
W0 = zeros(1,2048);
del = 0.01;
lam = 0.98;
x = mySig;
x = x(1:length(W0)*floor(length(x)/length(W0)));
d = d(1:length(W0)*floor(length(d)/length(W0)));

%The FDAF filter, useful for identifying long impulse response construct the Frequency-Domain Adaptive Filter
hFDAF = dsp.FrequencyDomainAdaptiveFilter('Length',2048,'StepSize',mu,'LeakageFactor',1,'InitialPower',del,'AveragingFactor',lam);   %e is after the filter
[y,e] = hFDAF(x,d);
n = 1:length(e);
t = n/fs;

figure

pos = get(gcf,'Position');  % gcf = current figure handle
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)+85)])

subplot(2,1,1);
plot(t,d(n),'b');
axis([0 5 -1 1]);
ylabel('Amplitude');
title('Hello echo Signal');
subplot(2,1,2);
plot(t,e(n),'r');
axis([0 5 -1 1]);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Output of Acoustic Echo Canceller \mu =0.001');
set(gcf, 'Color', [1 1 1])
pause(5)                                        %wait for key press
disp('Playing mixed Speech Signal after filter mu =0.001')
p8 = audioplayer(e/max(abs(e)),fs);
playblocking(p8);

%step 3: pass the echoed signal to LMS filter
mu = 0.025;                     %sys parameters for LMS
W0 = zeros(1,2048);
del = 0.01;
lam = 0.98;
x = mySig;
x = x(1:length(W0)*floor(length(x)/length(W0)));
d = d(1:length(W0)*floor(length(d)/length(W0)));

%The FDAF filter, useful for identifying long impulse response Construct the Frequency-Domain Adaptive Filter
hFDAF = dsp.FrequencyDomainAdaptiveFilter('Length',2048,'StepSize',mu,'LeakageFactor',1,'InitialPower',del,'AveragingFactor',lam);   %e is after the filter
[y,e] = hFDAF(x,d);
n = 1:length(e);
t = n/fs;
%
figure
%
pos = get(gcf,'Position');  % gcf = current figure handle
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)+85)])

subplot(2,1,1);
plot(t,d(n),'b');
axis([0 5 -1 1]);
ylabel('Amplitude');
title('Hello echo Signal');
subplot(2,1,2);
plot(t,e(n),'r');
axis([0 5 -1 1]);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Output of Acoustic Echo Canceller \mu =0.025');
set(gcf, 'Color', [1 1 1])
pause(5)                                        %wait for key press
disp('Playing mixed Speech Signal after filter mu =0.025')
p8 = audioplayer(e/max(abs(e)),fs);
playblocking(p8);

%step 3: pass the echoed signal to LMS filter
mu = 0.09;                     %sys parameters for LMS
W0 = zeros(1,2048);
del = 0.01;
lam = 0.98;
x = mySig;
x = x(1:length(W0)*floor(length(x)/length(W0)));
d = d(1:length(W0)*floor(length(d)/length(W0)));

%The FDAF filter, useful for identifying long impulse response Construct the Frequency-Domain Adaptive Filter
hFDAF = dsp.FrequencyDomainAdaptiveFilter('Length',2048,'StepSize',mu,'LeakageFactor',1,'InitialPower',del,'AveragingFactor',lam);   %e is after the filter
[y,e] = hFDAF(x,d);
n = 1:length(e);
t = n/fs;
%
figure
%
pos = get(gcf,'Position');  % gcf = current figure handle
set(gcf,'Position',[pos(1), pos(2)-100,pos(3),(pos(4)+85)])

subplot(2,1,1);
plot(t,d(n),'b');
axis([0 5 -1 1]);
ylabel('Amplitude');
title('Hello echo Signal');
subplot(2,1,2);
plot(t,e(n),'r');
axis([0 5 -1 1]);
xlabel('Time [sec]');
ylabel('Amplitude');
title('Output of Acoustic Echo Canceller \mu =0.09');
set(gcf, 'Color', [1 1 1])
pause(5)                                        %wait for key press
disp('Playing mixed Speech Signal after filter mu =0.09')
p8 = audioplayer(e/max(abs(e)),fs);
playblocking(p8);