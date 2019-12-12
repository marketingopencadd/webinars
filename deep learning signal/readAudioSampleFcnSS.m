function [s,w,t]=readAudioSampleFcnSS(filename,FsDefault,Nsec)
% this function:
% reads the audio file specified in "filename",
% cuts a random sample of "Nsec" seconds (never picking from the start or end),
% resamples to the sampling rate of "FsDefault" (if not already sampled at this rate),
% calculates the spectrogram "s" of this sample, 
% repeats the spectrogram along 3 channels (to "emmulate" an RGB image),
% and returns the log(s^2+1) (as a  way of reducing the dynamc range)

% the spectrogram function requires many parameters, see its documentation

[y,Fs]=audioread(filename);
dur=size(y,1)/Fs;
randStart=(dur-2*Nsec)*rand()+Nsec;
randEnd=randStart+Nsec;

times=((1:size(y,1))/Fs)';
inds=times>=randStart & times<randEnd;

ySample=y(inds,:);
if Fs~=FsDefault
    ySample=resample(ySample,FsDefault,Fs);
end
y=ySample;
[s,w,t]=spectrogram(y(:,1),size(y,1)/(10*Nsec),[],linspace(0,2*pi,FsDefault/100),FsDefault/1000);
s=repmat(log(s.*conj(s)+1),[1 1 3]);
end