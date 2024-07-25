mu = 0.001;                     % Step size
del = 0.01;                     % Initial power
lam = 0.98;                     % Averaging factor
blockLength = 1024;
lowpass_cutoff = 2000;

%%% Overall needs some fixing regarding strange noise in middle of audio


% Read the audio signals
sig_orig = audioread("mp3s/radio/CUT1-ORIG.mp3");
sig_echo = audioread("mp3s/radio/CUT1-ECHO.mp3");

% Ensure both signals are of the same length
min_length = min(length(sig_orig), length(sig_echo));
sig_orig_trimmed = sig_orig(1:min_length);
sig_echo_trimmed = sig_echo(1:min_length);

% Ensure the length is divisible by the block length
if mod(length(sig_orig_trimmed), blockLength) ~= 0
    % Zero-pad the signals to make their lengths divisible by blockLength
    padding_length = blockLength - mod(length(sig_orig_trimmed), blockLength);
    sig_orig_padded = [sig_orig_trimmed; zeros(padding_length, 1)];
    sig_echo_padded = [sig_echo_trimmed; zeros(padding_length, 1)];
else
    sig_orig_padded = sig_orig_trimmed;
    sig_echo_padded = sig_echo_trimmed;
end

%% testing filtering

% Create the frequency domain adaptive filter object
hFDAF = dsp.FrequencyDomainAdaptiveFilter('Length', blockLength, ...
                                          'StepSize', mu, ...
                                          'LeakageFactor', 1, ...
                                          'InitialPower', del, ...
                                          'AveragingFactor', lam);

% Compute the FFT of the padded signals
X = fft(sig_orig_padded);
D = fft(sig_echo_padded);

% Apply the adaptive filter
Y = hFDAF(X, D);

fs = 48000;

% Just to test
y = ifft(Y);
y = lowpass(y, lowpass_cutoff, fs);

[mse_values, snr_values, rmse_values, erle_values, time_vector] = similarity_windows(sig_orig_padded, real(y), sig_echo_padded, fs, 1, 0);
[mse_values2, snr_values2, rmse_values2, erle_values2, time_vector2] = similarity_windows(sig_orig_padded, sig_echo_padded, sig_echo_padded, fs, 1, 0);
[corr, lags, mse, ssim_index, spec_sim, cosine_sim, ERLE, ERLE_mean] = similarity_stats(sig_orig_padded, real(y));

time_signal = (0:length(sig_orig_padded)-1) / fs;

x_end = max(time_vector);
num_plots = 6;

subplot(num_plots, 1, 1);
plot(time_signal, sig_orig_padded)
title("Original Signal");
xlim([0, x_end]);

subplot(num_plots, 1, 2);
plot(time_signal, sig_echo_padded)
title("Echo Signal")
xlim([0, x_end]);

subplot(num_plots, 1, 3);
plot(time_signal, real(y))
title("Processed Signal")
xlim([0, x_end]);

subplot(num_plots, 1, 4);
plot(time_vector, snr_values);
title("SNR of Orig vs Processed")
xlim([0, x_end]);

subplot(6, 1, 5);
plot(time_vector2, snr_values2);
title("SNR of original vs echo")
xlim([0, x_end]);

subplot(num_plots, 1, 6);
plot(time_vector, erle_values);
title("ERLE")
xlim([0, x_end]);

p = audioplayer(real(y), fs);
playblocking(p);