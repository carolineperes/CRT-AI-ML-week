addpath("code")
addpath("qTFD")
addpath("qTFD/common/")
addpath("qTFD/full_TFDs/")
addpath("qTFD/utils")

clearvars;

raw = edfread("data/EDF_format/ID07_epoch1.edf");

% raw is a 3600x9 timetable
% each column represents one of the channels
% each row is an array with M samples for 1s


% create a vector with the seconds in the recording
rec_time = seconds(raw.("Record Time"));
T = length(rec_time); % total time in seconds (3600)


% convert timetable to continuous matrix
r = cell2mat(table2cell(raw));

% find the sampling frequency
[n_samples, n_channels] = size(r);
Fs = n_samples/T; % Fs = sampling frequency

% create time vector
t = linspace(0, n_samples-1, n_samples).'/Fs;


%% Bipolar channels
%F4-C4, C4-O2 (or C4-P4), F3-C3, C3-O1 (or C3-P3), 
% T4-C4, C4-Cz, Cz-C3 and C3-T3.
channels = raw.Properties.VariableNames;
BC = [ r(:,1)-r(:,2), ...  % F4-C4
       r(:,2)-r(:,4), ...  % C4-O2
       r(:,5)-r(:,6), ...  % F3-C3
       r(:,6)-r(:,8), ...  % C3-O1
       r(:,3)-r(:,2), ...  % T4-C4
       r(:,2)-r(:,9), ...  % C4-Cz
       r(:,9)-r(:,6), ...  % Cz-C3
       r(:,6)-r(:,7), ...  % C3-T3
    ];
n_bch = size(BC,2);


%% PRE-PROCESSING

% low-pass FIR zero-phase to 30Hz with Hamming window of 4001 samples
% then downsample to 64 Hz
% then get estim of deriv. via forward-finite difference of signal

t_5min = Fs*60*5; % num of points in 5 min of recording

i_seg = 10;
c_0 = r(i_seg*t_5min : (i_seg+1)*t_5min -1, :);

% figure(1); clf; plot(c_0)


H_lp = fdesign.lowpass('N,Fc',4001, 30, Fs);
Hd_lp = design(H_lp,'window','window', @hamming);
c_filt = filter(Hd_lp, c_0);
c = resample(c_filt, 64, Fs);
%c = c_filt;

% differential
% cfw = zeros(size(c));
h = 1/Fs;
% for i = 1:length(c)-1
%      cfw(i,:) = (c(i+1,:) - c(i,:))/h;
%  end
cfw = gradient(c,h);

% figure(2);clf; plot(c);
% hold on
% plot(cfw, 'r')    


%% qTFD - using full_tfd() from J O'Toole
% using separable kernel: k(t,f) = G(t)H(f)
% G(t) Doppler kernel with Hann window of length 127
% H(f) lag kernel with Hann window of length 63
% qTFD = matrix(256x128)

ch = 5;

r_0 = r(i_seg*t_5min : (i_seg+1)*t_5min -1, :);
% tf_r = full_tfd(r_0(:,ch), ...
%     'sep', ... % type of kernel = separable
%     { {127, 'hann'}, ... % doppler_window
%       {63, 'hann'}}, ... % lag_window
%     256, 128);  % Ntime (oversamplig), Nfreq (oversampling)
% 
% figure(1); clf; vtfd(tf_r,r_0(:,ch));
% 
% tf_c = full_tfd(c_0(:,ch), ...
%     'sep', ... % type of kernel = separable
%     { {127, 'hann'}, ... % doppler_window
%       {63, 'hann'}}, ... % lag_window
%     256, 128);  % Ntime (oversamplig), Nfreq (oversampling)
% 
% figure(2); clf; vtfd(tf_c,c_0(:,ch));


tf_cfw = full_tfd(cfw(:,ch), ...
    'sep', ... % type of kernel = separable
    { {127, 'hann'}, ... % doppler_window
      {63, 'hann'}}, ... % lag_window
    256, 128);  % Ntime (oversamplig), Nfreq (oversampling)

figure(3); clf; vtfd(tf_cfw,cfw(:,ch), 64);


%% POST-PROCESSING
% components 0-2Hz and 30-32Hz removed
% final matrix: 256 x 112

final = tf_cfw(:,8:119);
