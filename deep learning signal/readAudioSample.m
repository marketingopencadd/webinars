function [s,w,t]=readAudioSample(filename,FsDefault,Nsec,NumFreqs)
info=audioinfo(filename);
t0=rand().*(info.Duration-Nsec);
tf=t0+Nsec;
interval=floor([t0 tf].*info.TotalSamples./info.Duration+[1 0]);
[y,Fs]=audioread(filename,interval);

if Fs~=FsDefault
    y=resample(y,FsDefault,Fs);
end

[s,w,t]=spectrogram(y(:,1),size(y,1)/(5*Nsec),[],linspace(0,FsDefault./2,NumFreqs),FsDefault);
s=log10(abs(s)+1);
if any(size(s)~=[NumFreqs 399])
    s=imresize(s,[NumFreqs 399]);
end
end