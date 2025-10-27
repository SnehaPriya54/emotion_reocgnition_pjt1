%% Dummy Data Generation for IoT Emotion Recognition
clear; clc; rng(1);

numSamples = 200;  % per class
emotionNames = ["Happy","Sad","Angry","Relaxed","Stressed"];
numClasses = numel(emotionNames);
N = numSamples * numClasses; % total samples

%% Labels as categorical column
labels = categorical(repelem(emotionNames, numSamples))';

%% PPG Features (4)
ppg_data = [
    70 + 5*randn(numSamples,1),  2+0.5*randn(numSamples,1),  50+5*randn(numSamples,1),  30+5*randn(numSamples,1);
    60 + 4*randn(numSamples,1),  3+0.6*randn(numSamples,1),  55+5*randn(numSamples,1),  28+5*randn(numSamples,1);
    90 + 6*randn(numSamples,1),  4+0.7*randn(numSamples,1),  45+5*randn(numSamples,1),  20+5*randn(numSamples,1);
    65 + 5*randn(numSamples,1),  1+0.4*randn(numSamples,1),  60+5*randn(numSamples,1),  35+5*randn(numSamples,1);
    85 + 7*randn(numSamples,1),  5+0.8*randn(numSamples,1),  40+5*randn(numSamples,1),  18+5*randn(numSamples,1)];

%% GSR Features (4)
gsr_data = [
    0.8+0.1*randn(numSamples,1), 0.05+0.01*randn(numSamples,1), 0.001+0.0002*randn(numSamples,1), 3+randn(numSamples,1);
    0.6+0.1*randn(numSamples,1), 0.04+0.01*randn(numSamples,1), 0.0008+0.0002*randn(numSamples,1), 2+randn(numSamples,1);
    1.0+0.15*randn(numSamples,1), 0.06+0.015*randn(numSamples,1), 0.0012+0.0003*randn(numSamples,1), 4+randn(numSamples,1);
    0.7+0.1*randn(numSamples,1), 0.045+0.01*randn(numSamples,1), 0.0009+0.0002*randn(numSamples,1), 2+randn(numSamples,1);
    0.95+0.15*randn(numSamples,1),0.055+0.015*randn(numSamples,1),0.0011+0.0003*randn(numSamples,1), 5+randn(numSamples,1)];

%% Temp Features (2)
temp_data = [
    36.8+0.2*randn(numSamples,1),  0.002+0.0003*randn(numSamples,1);
    36.5+0.2*randn(numSamples,1),  0.0018+0.0003*randn(numSamples,1);
    37.2+0.3*randn(numSamples,1),  0.0025+0.0004*randn(numSamples,1);
    36.6+0.2*randn(numSamples,1),  0.0017+0.0003*randn(numSamples,1);
    37.1+0.3*randn(numSamples,1),  0.0023+0.0004*randn(numSamples,1)];

%% Combine all features
X = [ppg_data, gsr_data, temp_data];

%% Check alignment
fprintf('X size: %d x %d\n', size(X));
fprintf('Labels size: %d x %d\n', size(labels));
disp("Unique labels:"); disp(unique(labels));

%% Train multi-class SVM using ECOC
disp('Training multi-class SVM...');
svm = fitcecoc(X, labels, ...
    'Learners', templateSVM('KernelFunction','rbf', ...
                             'Standardize',true, ...
                             'KernelScale','auto'), ...
    'Coding','onevsone');

%% Save for Simulink
saveLearnerForCoder(svm, 'svmMdl.mat');
disp('svmMdl.mat saved successfully.');


%% Save for Simulink
saveLearnerForCoder(svm, 'svmMdl.mat');
disp('svmMdl.mat saved successfully.');
