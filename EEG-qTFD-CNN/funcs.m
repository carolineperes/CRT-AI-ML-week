


function y=preprocess(x, Fs)

H = fdesign.lowpass('N,Fc',4001, 30, Fs);
Hd = design(H,'window','window', @hamming);
y = filter(Hd, x);
y = resample(y, 64, Fs);

end

function y=tfd(x)
y = full_tfd(x, ...
    'sep', ... % type of kernel = separable
    { {127, 'hann'}, ... % doppler_window
      {63, 'hann'}}, ... % lag_window
    256, 128);  % Ntime (oversamplig), Nfreq (oversampling)
y = y(:, 8:119);
end

