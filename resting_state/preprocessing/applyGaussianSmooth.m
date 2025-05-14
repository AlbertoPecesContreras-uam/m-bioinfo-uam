%% FUNCTIONS
function out = applyGaussianSmooth(signal, sigma)
    % This function is able to apply a gaussian smoothness along all signal
    % points in order to remove abrupt picks.
    % arg_1 = signal (1 x n) double
    % arg_2 = sigma value for smoothness size effect (2)
    gaussian_kernel = fspecial('gaussian', [1, 2*sigma+1], sigma);
    out = conv(signal, gaussian_kernel, 'same');

end