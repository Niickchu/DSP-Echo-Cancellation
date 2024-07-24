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

subplot(5, 1, 1);
plot(sig_orig_trimmed);
title('Original Signal');
xlabel('Samples');
ylabel('Amplitude');
grid on;


subplot(5, 1, 2);
plot(sig_echo_trimmed);
title('Original Echoed Signal');
xlabel('Samples');
ylabel('Amplitude');
grid on;

subplot(5, 1, 3);
plot(real(y));
title('Cleaned Signal After Echo Cancellation');
xlabel('Samples');
ylabel('Amplitude');
grid on;

[corr1, lag1, mse1, ssim_index1, spec_sim1, cosine_sim1, ERLE1, ERLE_mean1] = test_similarity(sig_orig_padded, sig_echo_padded);
[corr2, lags2, mse2, ssim_index2, spec_sim2, cosine_sim2, ERLE2, ERLE_mean2] = test_similarity(sig_orig_padded, real(y));

fprintf("MSE before: %.4f, MSE after: %.4f\n", mse1, mse2);
fprintf("SSIM before: %.4f, SSIM after: %.4f\n", ssim_index1, ssim_index2);
fprintf("ERLE mean before: %.4f, ERLE mean after: %.4f\n", ERLE_mean1, ERLE_mean2);

subplot(5, 1, 4);
plot(ERLE1);

subplot(5, 1, 5);
plot(ERLE2);

figure;
subplot(2, 1, 1);
plot(corr1);
subplot(2, 1, 2);
plot(corr2);

p = audioplayer(real(y), fs);
playblocking(p);