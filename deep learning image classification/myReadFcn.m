function I=myReadFcn(filename)
frame=imread(filename);
S=size(frame);
s=min(S(1:2));
I=imresize(frame,227./s);
end