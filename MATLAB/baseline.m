addpath("RIR\")

% Load audio files
male_speech = audioread("../Python/samples/male.wav");
female_speech = audioread("../Python/samples/female.wav");

% Room parameters
c = 340;                    % Sound velocity (m/s)
fs = 8000;                 % Sample frequency (samples/s)
r = [0.1 0.5 0.1];         % Receiver position [x y z] (m)
s = [1.5 1.5 1.5];         % Source position [x y z] (m)
L = [2 2 2];               % Room dimensions [x y z] (m)
beta = 0.08;               % Reverberation time (s)
n = 4096;                  % Number of samples

t = linspace(0,n/fs,n);
h = rir_generator(c, fs, r, s, L, beta, n);

% Create echoed signal
female_echoed = filter(h, 1, female_speech);
female_echoed = female_echoed(1:length(female_speech));

% Scale echoed signal
scale = sqrt(mean(female_speech.^2)) / sqrt(mean(female_echoed.^2));
female_echoed = female_echoed * scale;

% Pad signals
L = max([length(female_echoed), length(male_speech), length(female_speech)]);
female_echoed = [female_echoed; zeros(L - length(female_echoed), 1)];
female_speech = [female_speech; zeros(L - length(female_speech), 1)];
male_speech = [male_speech; zeros(L - length(male_speech), 1)];

figure;
plot(female_echoed);

% Adaptive filtering parameters
M_list = [32, 64, 128, 256, 512, 1024, 2048];
MU_list = [0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2];
mses = zeros(1, length(MU_list));
snrs = zeros(1, length(MU_list));
erles = zeros(1, length(MU_list));

for i = 1:length(MU_list)
    mu = MU_list(i);                     % Step size
    blockLength = 512;

    L = max([length(female_echoed), length(male_speech), length(female_speech)]);
    if mod(L, blockLength) ~= 0
        % Zero-pad the signals to make their lengths divisible by blockLength
        padding_length = blockLength - mod(L, blockLength);
        female_echoed_ = [female_echoed; zeros(padding_length, 1)];
        female_speech_ = [female_speech; zeros(padding_length, 1)];
        male_speech_ = [male_speech; zeros(padding_length, 1)];
    else
        female_echoed_ = female_echoed;
        female_speech_ = female_speech;
        male_speech_ = male_speech;
    end

    % Combine the male speech and female echoed signals
    near_end_signal = male_speech_ + female_echoed_;

    % Create the adaptive filter
    hFDAF = dsp.FrequencyDomainAdaptiveFilter('BlockLength', blockLength, ...
                                              'StepSize', mu, 'AveragingFactor', 0.90)
    
    % FFT of the signals
    D = fft(male_speech_);
    X = fft(near_end_signal);
    
    % Apply the adaptive filter
    Y = hFDAF(D, X);

    % Inverse FFT to get the time-domain signal and error
    y = ifft(Y);
    y = real(y);

    % Calculate MSE
    mse = mean((male_speech_ - y).^2);
    
    % Calculate SNR (in dB)
    signal_power = mean(male_speech_.^2);
    noise_power = mean((male_speech_ - y).^2);
    snr = 10 * log10(signal_power / noise_power);
    
    near_power = mean(near_end_signal.^2);
    processed_power = mean(y.^2);
    % Calculate ERLE (in dB)
    erle = 10 * log10(near_power / processed_power);  % Assuming signal_power is the desired signal power


    if i == 5
        figure;
        subplot(3, 1, 1);
        plot(y)
        subplot(3, 1, 2);
        plot(male_speech_)
        subplot(3, 1, 3);
        plot(female_echoed_)
        % p1 = audioplayer(y, fs);
        % playblocking(p1);
        % p1 = audioplayer(near_end_signal, fs);
        % playblocking(p1);
    end

    % Store MSE and SNR values
    mses(i) = mse;
    snrs(i) = snr;
    erles(i) = erle;
end

% Plot results
figure;

subplot(3, 1, 1);
plot(MU_list, mses);
title('MSE vs. Step Size (MU)');
xlabel('Step Size (MU)');
ylabel('Mean Squared Error (MSE)');

subplot(3, 1, 2);
plot(MU_list, snrs);
title('SNR vs. Step Size (MU)');
xlabel('Step Size (MU)');
ylabel('Signal-to-Noise Ratio (SNR)');

subplot(3, 1, 3);
plot(MU_list, erles);
title('ERLE vs. Step Size (MU)');
xlabel('Step Size (MU)');
ylabel('ERLE');

sgtitle("Step Size vs Fixed Block Size (512)")

[val, min_i] = min(mses);

mses = zeros(1, length(M_list));
snrs = zeros(1, length(M_list));
erles = zeros(1, length(M_list));

disp(MU_list(min_i))

for i = 1:length(M_list)
    mu = MU_list(min_i);                     % Step size
    blockLength = M_list(i);

    
    
    L = max([length(female_echoed), length(male_speech), length(female_speech)]);
    if mod(L, blockLength) ~= 0
        % Zero-pad the signals to make their lengths divisible by blockLength
        padding_length = blockLength - mod(L, blockLength);
        female_echoed_ = [female_echoed; zeros(padding_length, 1)];
        female_speech_ = [female_speech; zeros(padding_length, 1)];
        male_speech_ = [male_speech; zeros(padding_length, 1)];
    else
        female_echoed_ = female_echoed;
        female_speech_ = female_speech;
        male_speech_ = male_speech;
    end

    % Combine the male speech and female echoed signals
    near_end_signal = male_speech_ + female_echoed_;

    % Create the adaptive filter
    hFDAF = dsp.FrequencyDomainAdaptiveFilter('BlockLength', blockLength, ...
                                              'StepSize', mu, 'AveragingFactor', 0.90)
    
    % FFT of the signals
    D = fft(male_speech_);
    X = fft(near_end_signal);
    
    % Apply the adaptive filter
    [Y, e] = hFDAF(D, X);
    
    % Inverse FFT to get the time-domain signal and error
    y = ifft(Y);
    y = real(y);
    e = ifft(e);
    e = real(e);
    
    % Trimming to ensure the lengths match for error calculation
    % male_cut = male_speech_(1:length(y));
    % y = y(60000:end);
    % male_cut = male_cut(60000:end);
    
    % Calculate MSE
    mse = mean((male_speech_ - y).^2);
    
    % Calculate SNR (in dB)
    signal_power = mean(male_speech_.^2);
    noise_power = mean((male_speech_ - y).^2);
    snr = 10 * log10(signal_power / noise_power);
    
    near_power = mean(near_end_signal.^2);
    processed_power = mean(y.^2);
    % Calculate ERLE (in dB)
    erle = 10 * log10(near_power / processed_power);  % Assuming signal_power is the desired signal power
    % Store MSE and SNR values
    mses(i) = mse;
    snrs(i) = snr;
    erles(i) = erle;
end

% Plot results
figure;
sgtitle("Block Size vs Fixed Step Size (0.04)")

subplot(3, 1, 1);
plot(M_list, mses);
title('MSE vs. Step Size (MU)');
xlabel('Block Size (M)');
ylabel('Mean Squared Error (MSE)');

subplot(3, 1, 2);
plot(M_list, snrs);
title('SNR vs. Step Size (MU)');
xlabel('Block Size (M)');
ylabel('Signal-to-Noise Ratio (SNR)');

subplot(3, 1, 3);
plot(M_list, erles);
title('ERLE vs. Step Size (M)');
xlabel('Block Size (M)');
ylabel('ERLE');
