%% create image datastore


ds=imageDatastore('../foods2/','IncludeSubfolders',true,'LabelSource','foldernames','ReadFcn',@(x) myReadFcn(x));

%% filtering classes
[g,gID]=findgroups(ds.Labels);
NperCat=splitapply(@numel,ds.Labels,g);
table(gID,NperCat)

minN=min(NperCat);
sampleInds=[];
for k=1:numel(gID)
    sampleInds=[sampleInds;randsample(find(ds.Labels==gID(k)),minN)];
end
ds=imageDatastore(ds.Files(sampleInds),'IncludeSubfolders',true,'LabelSource','foldernames','ReadFcn',@(x) myReadFcn(x));
[g,gID]=findgroups(ds.Labels);
NperCat=splitapply(@numel,ds.Labels,g);
table(gID,NperCat)

%% shuffling, partitioning, balancing, augmenting
ds=shuffle(ds);
[validDs,testDs,trainDs]=splitEachLabel(ds,0.1,0.1);

aug=imageDataAugmenter('RandXReflection',true,'RandYReflection',false,'RandRotation',[-45 45]);
trainAds=augmentedImageDatastore([227 227],trainDs,'OutputSizeMode','centercrop','DataAugmentation',aug);
validAds=augmentedImageDatastore([227 227],validDs,'OutputSizeMode','centercrop','DataAugmentation',aug);
testAds=augmentedImageDatastore([227 227],testDs,'OutputSizeMode','centercrop');
numCats=numel(unique(ds.Labels));

%% Load AlexNet
net=alexnet;
net.Layers

%% Transfer Learning: change last layers
newLayers=[fullyConnectedLayer(2000,'Name','myFullConn1');
    reluLayer('Name','myRelu1');
    dropoutLayer(0.5,'Name','myDropout1');
    fullyConnectedLayer(numCats,'Name','myFullConnEnd');
    softmaxLayer('Name','myProb');
    classificationLayer('Name','myOutput')];

layers=[net.Layers(1:(end-numel(newLayers))); newLayers];


%% Training CNN

opts=trainingOptions('sgdm','Plots','training-progress','ExecutionEnvironment','gpu','MaxEpochs',500,'MiniBatchSize',128,...
    'Shuffle','every-epoch','ValidationData',validAds,'ValidationFrequency',11,'ValidationPatience',40,'InitialLearnRate',0.001);%,...

istrained=false;
if ~istrained
    net=trainNetwork(trainAds,layers,opts);
    %save('net3.mat','net')
else
    %load('net3.mat')
end

%% Testar CNN

predClasses=classify(net,testAds);
figure, confusionchart(testDs.Labels,predClasses,'RowSummary','row-normalized','ColumnSummary','column-normalized');
%M=plotConfMat(testDs.Labels,predClasses,'',hot(64));

