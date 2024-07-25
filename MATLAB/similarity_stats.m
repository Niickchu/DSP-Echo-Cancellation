function [corr, lags, mse, ssim_index, spec_sim, cosine_sim, ERLE, ERLE_mean] = similarity_stats(audio_orig_,audio_mod_)
%TEST_SIMILARITY Summary of this function goes here
%   Detailed explanation goes here
    %% CROSS-CORR

    audio_orig = audio_orig_ / max(abs(audio_orig_)); 
    audio_mod = audio_mod_ / max(abs(audio_mod_));

    [corr, lags] = xcorr(audio_orig, audio_mod);
    [max_corr, max_lag_index] = max(corr);
    max_lag = lags(max_lag_index);
    
    %% MSE
    mse = mean((audio_orig - audio_mod).^2);
    
    %% SSIM
    ssim_index = ssim(audio_mod, audio_orig);
    
    %% SPECTRAL SIM
    fft_orig = fft(audio_orig);
    fft_echo = fft(audio_mod);
    
    mag_orig = abs(fft_orig);
    mag_echo = abs(fft_echo);
    
    spec_sim = dot(mag_orig, mag_echo) / (norm(mag_orig) * norm(mag_echo));
    
    %% COSINE SIMILARITY
    
    cosine_sim = dot(audio_orig, audio_mod) / (norm(audio_orig) * norm(audio_mod));
    
    %% ERLE
    
    P_orig = audio_orig.^2;
    P_echo = audio_mod.^2;
    
    ERLE = 10*log10(P_orig./P_echo);
    ERLE_mean = 10*log10(mean(P_orig)/mean(P_echo));
end
