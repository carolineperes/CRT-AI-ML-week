%% Process qTFD
% this script takes edf files in '../data/EDF_format/' (eeg signal)
% and converts it into 256x128x8x23 mat


addpath("code")
addpath("qTFD")
addpath("qTFD/common/")
addpath("qTFD/full_TFDs/")
addpath("qTFD/decimated_TFDs/")
addpath("qTFD/utils")

files = struct2table(dir('data/EDF_format'));
fnames = files.name;
n_files = size(files,1);

for i=1:n_files
    if files.isdir(i) == 0
        fname = fnames{i};
        [eeg_sig, qtfd, Fs] = shapedata(strcat("data/EDF_format/",fname));
        qtfd_log = log(abs(qtfd));
        fname = strcat(strtok(fname,'.'),'.mat');
        fname = strcat("data/MAT_format/", fname);
        disp(fname);
        save(fname,"eeg_sig","qtfd", "qtfd_log");
        disp('saved');
    end
end



%% Functions

function [eeg_sig, qtfd, Fs] = shapedata(filename)
r = edfread(filename);
T = size(r, 1);
r = cell2mat(table2cell(r));
[n_samples, ~] = size(r);
Fs = n_samples/T; % Fs = sampling frequency
% create time vector
t = linspace(0, n_samples-1, n_samples).'/Fs;
n_5min = Fs*60*5;
n_5min_downsampled = 64*60*5;
% convert channels to bipolar
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
n_sgms = 2*(T/(60*5))-1;
eeg_sig = zeros(n_5min_downsampled,n_bch,n_sgms);
qtfd = zeros(256,128,n_bch,n_sgms);


for i=1:n_bch
    ch = BC(:, i);
    segms = buffer(ch,n_5min, n_5min/2, 'nodelay');

    % each col of segms is one segment of 5 min
    for j=1:size(segms,2)
        x = preprocess(segms(:,j), Fs);
        y = tfd(x);
        eeg_sig(:,i,j) = x;
        qtfd(:,:,i, j) = y;    
    end
end



end

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
%y = y(:, 8:119);
end
