function [R,d]=calcSunPath(L)
% L - latitude de interesse, em graus
% R(k,h,:) - vetor de dire��o do sol no dia k e minuto h
% R � NaN quando o sol est� abaixo do horizonte
% d - datas de 1 ano inteiro, come�ando no equin�cio de mar�o
tilt=23.44;
d=(datetime(2018,3,20):days(1):datetime(2019,3,19))';
q=-tilt.*sin(linspace(0,2*pi,numel(d)))';

A=[1 0 -sin(L.*pi/180) 0;
    0 1 cos(L.*pi/180) 0;
    1 0 0 -cos(L.*pi/180);
    0 1 0 -sin(L.*pi/180)];
B=[-sin((L+q').*pi/180);
    cos((L+q').*pi/180);
    zeros(1,numel(q));
    zeros(1,numel(q))];
gama=A\B;
x0=gama(1,:)';
z0=gama(2,:)';

h=linspace(0,2*pi,24*60);
r0=reshape([x0 zeros(numel(x0),1) z0],[numel(x0) 1 3]);
w=reshape([0 1 0],1,1,3);
v=reshape([-sin(L.*pi/180) 0 cos(L.*pi/180)],1,1,3);

rho=cos(q.*pi/180);
s=rho.*sin(h);
t=rho.*cos(h);

r=@(s,t) r0+s.*v+t.*w;
R=r(s,t);
R(repmat(R(:,:,3)<0,[1 1 3]))=NaN;
end

