%% defining constants, loading song list

Fs=44100;               % sample frequency; this is default for most audio files, setting this guarantees there will be no exceptions
Nsec=20;                % sample size: 20 seconds
warning('off','all')    % sometimes the audio reading function may throw warnings

load('T2.mat')          % table with filename, title, artist and gender of a lot of songs
T(randperm(height(T),10),:)

%% balancing data

cats=categories(T.Gender);              % 6 categories
numCats=numel(cats);                    %

% calculating number of files for each category
[g,gID]=findgroups(T.Gender);
N=splitapply(@numel,T.Gender,g);
table(gID,N)

% selecting a random amount of files for each category so that each
% category has an amount of files equal to the least frequent category
M=min(N);
inds=cell(1,numCats);
for z=1:numCats
    inds{z}=randsample(find(T.Gender==cats{z}),M);
end
inds=vertcat(inds{:});

% creating the datastore, using a custom read function to read the
% spectrogram of .mp3 files; look at the function code and comments
ds=imageDatastore(T.file(inds),'ReadFcn',@(x) readAudioSampleFcnSS(x,Fs,Nsec),'FileExtensions','.mp3','Labels',T.Gender(inds));

%% splitting and augmenting training, validation and test data

[dsValid,dsTest,dsTrain]=splitEachLabel(ds,0.1,0.1);

% the size returned by the read  function is [441 399]; the AlexNet needs
% [227 277]; so the augmentedImageDatastore normalizes the size; no other
% transformation is applid in the augmentation
augDsTrain=augmentedImageDatastore([227 227],dsTrain);
augDsValid=augmentedImageDatastore([227 227],dsValid);
augDsTest=augmentedImageDatastore([227 227],dsTest);

%% load AlexNet
% pretrained image classification CNN, one of the simplest models
% available; changing for something better like the Inception from Google
% might improve performance (and increase training time)
% each pretrained model requires a Support Package to be installed first

net=alexnet;
net.Layers

%% transfer learning: change last layers
% definition of the new layers; fullyConnected are the "standard" network
% layer that learn to perform a task; dropout layers randomly disable
% neurons before passing data on to the next layer to avoid overfitting
% during training; softmax and classification produce the output

newLayers=[fullyConnectedLayer(2048,'Name','myFullConn1');
    dropoutLayer(0.1,'Name','myDropout1');
    fullyConnectedLayer(512,'Name','myFullConn2');
    dropoutLayer(0.1,'Name','myDropout1');
    fullyConnectedLayer(numCats,'Name','myFullConnEnd');
    softmaxLayer('Name','myProb');
    classificationLayer('Name','myOutput')];

layers=[net.Layers(1:(end-numel(newLayers))); newLayers];
layers

%% training parameters
% sgdm is the 'standard' training algorithm, see the doc for others
% MiniBatchSize: how many inputs are taken in each batch of the training process; set it to be as high as your GPU can hold in memory
% InitialLearnRate: define the step size that the training algorithm takes to correct the weights in the neurons during training; may or may not change during training, therefore 'initial'
% Verbose: true to show the training process on the command window
% VerboseFrequency: how often to show stuff in the command window
% MaxEpochs: how many times, at most, to pass through the entire training dataset
% LearnRateSchedule: define if the learnRate will be constant or will drop every few epochs
% LearnRateDropPeriod and LearnRateDropFactor: if LearnRateSchedule is 'piecewise', learnRate will dropby LearnRateDropFactor every LearnRateDropPeriod epochs
% ValidationData: if empty, no validation; if set, the chosen datastore will be used for validation
% ValidationFrequency and ValidationPatience: if validation will be done, every ValidationFrequency iterations (mini batches) a pass through the validation data will be made, and after ValidationPatience validations without improvement training is stopped
% Shuffle: training data is reshuffled every epoch, or however it is set
% Plots: training process is shown graphically

% this list is not exhaustive, see the doc

opts=trainingOptions('sgdm','MiniBatchSize',8,'InitialLearnRate',1e-3,'Verbose',true,'VerboseFrequency',1,'MaxEpochs',500,...
'LearnRateDropPeriod',10,'LearnRateDropFactor',0.5,'LearnRateSchedule','piecewise',...
'ValidationData',augDsValid,'ValidationFrequency',37,'ValidationPatience',10,...
'Shuffle','every-epoch','Plots','training-progress');

%% training

[net,trInfo]=trainNetwork(augDsTrain,layers,opts);  % trains the network
save('netStyleClassification1.mat','net')         % and save it

%% evaluation: confusion matrix

[pred,score]=classify(net,augDsTest);   % apply the network to new data (this case the test set)

% evaluate performance through the confusion matrix
figure, cm=confusionchart(dsTest.Labels,pred,'RowSummary','row-normalized','ColumnSummary','column-normalized');

% average accuracy
accuracy=sum(diag(cm.NormalizedValues))./sum(cm.NormalizedValues(:))

% turns warnings back on
warning('on','all')