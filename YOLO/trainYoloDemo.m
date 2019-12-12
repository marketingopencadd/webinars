load('tempData.mat')
str=regexp(data.imageFilename,'\\frames\\train\\frame\d*.jpg','match');
str=cellfun(@(x) ['.' x{1}],str,'UniformOutput',false);
data.imageFilename=str;
data(:,3:5)=[];

for k=2:2
    for h=1:height(data)
        if ~isempty(data{h,k}{1})
            data{h,k}{1}=data{h,k}{1}(prod(data{h,k}{1}(:,3:4),2)>1500,:);
        end
    end
end

%% selecting anchor boxes by clustering

% allCarBoxes=vertcat(data.car{:});
% allBikeBoxes=vertcat(data.bike{:});
% allTruckBoxes=vertcat(data.truck{:});
% allBusBoxes=vertcat(data.bus{:});
% 
% allBoxes=vertcat(allCarBoxes,allBikeBoxes,allTruckBoxes,allBusBoxes);
allBoxes=vertcat(data.car{:});
aspectRatio=allBoxes(:,3)./allBoxes(:,4);
area=prod(allBoxes(:,3:4),2);

%%
% tol=0.02;
% meanIoU=zeros(10,1);
% kmedsOpts.MaxIter=25;
% for numAnchors=2:numel(meanIoU)
%     disp(numAnchors+"/"+numel(meanIoU))
%     [clusterAssignments,~,sumd]=kmedoids(allBoxes(:,3:4),numAnchors,'Distance',@iouDistanceMetric,'Options',kmedsOpts,'Algorithm','clara');
%     counts=accumarray(clusterAssignments, ones(length(clusterAssignments),1),[],@(x)sum(x)-1);
%     meanIoU(numAnchors)=mean(1-sumd./(counts));
% end
% 
% figure, plot((1:numel(meanIoU))',meanIoU)
% xlabel('numAnchors')
% ylabel('meanIoU')

%%
% numAnchors=find(diff(meanIoU)<tol,1,'first');
% if isempty(numAnchors)
%     numAnchors=20;
% else
%     numAnchors=numAnchors+1;
% end
numAnchors=15;
[clusterAssignments,anchorBoxes,sumd]=kmedoids(allBoxes(:,3:4),numAnchors,'Distance',@iouDistanceMetric);

figure, gscatter(area,aspectRatio,clusterAssignments)
xlabel('area')
ylabel('aspectRatio')
title([num2str(numAnchors) ' anchors'])
ax=gca;
ax.XScale='log';

%% transfer learning from ResNet50

imageSize=[224 224 3];
numClasses=width(data)-1;

resizedAnchors=floor(anchorBoxes.*[224 224]./[540 960]);

baseNetwork=resnet50;
featureLayer='activation_40_relu';
% baseNetwork=alexnet;
% featureLayer='relu5';
lgraph=yolov2Layers(imageSize,numClasses,resizedAnchors,baseNetwork,featureLayer);

%% splitting data

inds=rand(height(data),1)<0.1;
validData=data(inds,:);
trainData=data(~inds,:);

%% training

maxEpochs=5;
minibatch=16;
options = trainingOptions('adam',...
    'MiniBatchSize', minibatch,....
    'InitialLearnRate',1e-3,...
    'MaxEpochs',maxEpochs,...
    'CheckpointPath',tempdir,...
    'Shuffle','every-epoch',...
    'VerboseFrequency',1,...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.2,'LearnRateDropPeriod',2);

[detector,info]=trainYOLOv2ObjectDetector(trainData,lgraph,options);

%% evaluate performance

results=table('Size',[height(validData) 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});

for k=1:height(validData)
    try
        I=imread(validData.imageFilename{k});
        [bboxes,scores,labels]=detect(detector,I,'Threshold',0.02);
        [bboxes,scores,labels]=selectStrongestBboxMulticlass(bboxes,scores,labels,'RatioType','Min','OverlapThreshold',0.2);
        results.Boxes{k}=bboxes;
        results.Scores{k}=scores;
        results.Labels{k}=labels;
    catch
        
    end
end

%%
[ap,recall,precision]=evaluateDetectionPrecision(results,validData(:,2:end),0.02);
for k=1:1 %numel(recall)
    figure, plot(recall,precision)
    xlabel('recall')
    ylabel('precision')
    title({['Average precision: ' num2str(100*ap(k),2) '%'];[num2str(height(trainData)) ' training samples']})
end

%save('myYOLOdetector.mat','detector')

%% helping functions

function dist = iouDistanceMetric(boxWidthHeight,allBoxWidthHeight)
% Return the IoU distance metric. The bboxOverlapRatio function
% is used to produce the IoU scores. The output distance is equal
% to 1 - IoU.

% Add x and y coordinates to box widths and heights so that
% bboxOverlapRatio can be used to compute IoU.
boxWidthHeight = prefixXYCoordinates(boxWidthHeight);
allBoxWidthHeight = prefixXYCoordinates(allBoxWidthHeight);

% Compute IoU distance metric.
dist = 1 - bboxOverlapRatio(allBoxWidthHeight, boxWidthHeight);
end

function boxWidthHeight = prefixXYCoordinates(boxWidthHeight)
% Add x and y coordinates to boxes.
n = size(boxWidthHeight,1);
boxWidthHeight = [ones(n,2) boxWidthHeight];
end