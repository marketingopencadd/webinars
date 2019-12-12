%% defining constants, loading song list

Fs=44100;               % sample frequency; this is default for most audio files, setting this guarantees there will be no exceptions
Nsec=20;                % sample size: 20 seconds
Nfreqs=256;             % number of spectrogram frequencies
warning('off','all')    % sometimes the audio reading function may throw warnings

% load('T2.mat')          % table with filename, title, artist and gender of a lot of songs
% F=cellfun(@(x) x{end},regexp(T.file,'\','split'),'UniformOutput',false);
% newPath='D:\Users\daniel.vieira\Desktop\music\musicas\';
% F=strcat({newPath},F);

load('songFiles.mat')
F=randsample(F,512);    % amostra pequena para prova de conceito
F(randperm(numel(F),10))

inds=rand(numel(F),1)<0.9;
dsTrain=imageDatastore(F(inds),'ReadFcn',@(x) readAudioSample(x,Fs,Nsec,Nfreqs),'FileExtensions','.mp3');
dsValid=imageDatastore(F(~inds),'ReadFcn',@(x) readAudioSample(x,Fs,Nsec,Nfreqs),'FileExtensions','.mp3');

%%

layers = [
    imageInputLayer([256 64 1],"Name","imageinput","Normalization","none")
    convolution2dLayer([7 7],48,"Name","conv_1","Padding","same","Stride",[2 2])
    leakyReluLayer("Name","relu_1")
    convolution2dLayer([5 5],48,"Name","conv_4","Padding","same")
    leakyReluLayer("Name","relu_6")
    dropoutLayer(0.1,"Name","dropout_1")
    maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([7 7],48,"Name","conv_2","Padding","same","Stride",[2 2])
    leakyReluLayer("Name","relu_2")
    convolution2dLayer([5 5],48,"Name","conv_5","Padding","same")
    leakyReluLayer("Name","relu_7")
    dropoutLayer(0.1,"Name","dropout_2")
    maxPooling2dLayer([5 5],"Name","maxpool_2","Padding","same","Stride",[2 2])
    transposedConv2dLayer([9 9],48,"Name","transposed-conv_1","Cropping","same","Stride",[2 2])
    leakyReluLayer("Name","relu_3")
    transposedConv2dLayer([9 9],48,"Name","transposed-conv_2","Cropping","same","Stride",[2 2])
    leakyReluLayer("Name","relu_4")
    transposedConv2dLayer([9 9],48,"Name","transposed-conv_3","Cropping","same","Stride",[2 2])
    leakyReluLayer("Name","relu_5")
    transposedConv2dLayer([9 9],48,"Name","transposed-conv_4","Cropping","same","Stride",[2 2])
    leakyReluLayer(0.1,"Name","leakyrelu")
    convolution2dLayer([1 1],1,"Name","conv_3","Padding","same")
    regressionLayer("Name","regressionoutput")];

%%
PatchesPerImage=8;
pdsTrain=randomPatchExtractionDatastore(dsTrain,dsTrain,[256 64],'PatchesPerImage',PatchesPerImage);
pdsValid=randomPatchExtractionDatastore(dsValid,dsValid,[256 64],'PatchesPerImage',PatchesPerImage);

%%
miniBatch=8;
maxEpochs=300;
numImgs=numel(dsTrain.Files);
iterPerEpoch=round(PatchesPerImage*numImgs./miniBatch);
validFreq=floor(0.5*iterPerEpoch);
validPat=10;
opts=trainingOptions('adam','MiniBatchSize',miniBatch,'MaxEpochs',maxEpochs,...
    'InitialLearnRate',1e-4,'L2Regularization',1e-4,'Epsilon',1e-8,...
    'ValidationData',pdsValid,'ValidationFrequency',validFreq,'ValidationPatience',validPat,...
    'Shuffle','every-epoch','Plots','training-progress');

%%

[net,trInfo]=trainNetwork(pdsTrain,layers,opts);
