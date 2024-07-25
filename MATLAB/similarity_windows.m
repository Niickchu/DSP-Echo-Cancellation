function [mse_values, snr_values, rmse_values, erle_values, time_vector] = similarity_windows(original_signal,processed_signal,echoed_signal,fs,window_length, overlap_length)
    num_windows = floor((length(original_signal) - overlap_length) / (window_length - overlap_length));
    
    % Preallocate arrays for metrics
    mse_values = zeros(num_windows, 1);
    snr_values = zeros(num_windows, 1);
    rmse_values = zeros(num_windows, 1);
    erle_values = zeros(num_windows, 1);
    
    % Calculate metrics for each window
    for k = 1:num_windows
        start_index = (k - 1) * (window_length - overlap_length) + 1;
        end_index = start_index + window_length - 1;
    
        % Extract the current window
        original_window = original_signal(start_index:end_index);
        processed_window = processed_signal(start_index:end_index);
        echoed_window = echoed_signal(start_index:end_index);
    
        % Calculate MSE, mean square error
        mse_values(k) = mean((original_window - processed_window).^2);
    
        % Calculate RMSE, root MSE
        rmse_values(k) = sqrt(mse_values(k));
    
        % Calculate SNR, signal to noise ratio
        noise_power = mean((original_window - processed_window).^2);
        signal_power = mean(original_window.^2);
        snr_values(k) = 10 * log10(signal_power / noise_power);
    
        % Calculate ERLE, residual echo after processing
        P_original = mean(original_window.^2);
        residual_echo = echoed_window - processed_window;
        P_residual = mean(residual_echo.^2);
        erle_values(k) = 10 * log10(P_original / P_residual);
    end
    
    % Create time vector for plotting
    time_vector = (0:num_windows-1) * (window_length - overlap_length) / fs;
end

