function outImg = yolov2_detectFcn(in)

%   Copyright 2018-2019 The MathWorks, Inc.

% A persistent object yolov2Obj is used to load the YOLOv2ObjectDetector object.
% At the first call to this function, the persistent object is constructed and
% setup. When the function is called subsequent times, the same object is reused 
% to call detection on inputs, thus avoiding reconstructing and reloading the
% network object.
persistent yolov2Obj;

if isempty(yolov2Obj)
    yolov2Obj = coder.loadDeepLearningNetwork('myYOLOdetector.mat');
end

% pass in input
[bboxes,scores] = yolov2Obj.detect(in,'Threshold',0.02);
[bboxes,scores]=selectStrongestBbox(bboxes,scores,'RatioType','Min','OverlapThreshold',0.2);
inds=scores>max(scores)*0.75;

outImg=insertObjectAnnotation(in,'rectangle',bboxes(inds,:),repmat({'car'},[sum(inds) 1]));
