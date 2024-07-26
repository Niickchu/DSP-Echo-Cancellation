% Frequency Domain Adaptive Filter using Overlap-Save Method
clear
close all
clc

% Parameters
N = 256;  % FFT length
mu = 0.01;  % Step size
num_iterations = 1000;  % Number of iterations

% Generate some example input and desired signals
x = randn(1, num_iterations + N - 1);  % Input signal
d = filter([1, -0.5], 1, x);  % Desired signal (output of an unknown system)

% Initialization
W = zeros(1, N);  % Frequency domain filter weights
e = zeros(1, num_iterations);  % Error signal
X = zeros(1, N);  % Input buffer

for k = 1:num_iterations
    % Overlap-Save Method: Shift and load new input block
    X = [X(N/2 + 1:end), x(k + N/2 - 1:-1:k)];
    
    % FFT of the input block
    Xf = fft(X, N);
    
    % Compute output of the adaptive filter in frequency domain
    Yf = W .* Xf;
    
    % Inverse FFT to obtain time domain output
    y = ifft(Yf, N);
    
    % Compute error
    e(k) = d(k) - y(N/2 + 1);  % Only keep the last N/2 points as valid
    
    % FFT of the error signal (padded to length N)
    Ef = fft([zeros(1, N/2), e(k)], N);
    
    % Update filter weights in frequency domain
    W = W + mu * conj(Xf) .* Ef;
end

% Plot the results
figure;
subplot(3,1,1);
plot(x);
title('Input Signal');

subplot(3,1,2);
plot(d);
title('Desired Signal');

subplot(3,1,3);
plot(e);
title('Error Signal');
xlabel('Sample Index');
