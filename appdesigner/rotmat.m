function rm=rotmat(u,theta)
% função auxiliar para definir matrizes de rotação arbitrárias a partir
% de um vetor representando o eixo de rotação e um ângulo de rotação
rm=[cos(theta)+u(1)^2*(1-cos(theta)) u(1)*u(2)*(1-cos(theta))-u(3)*sin(theta) u(1)*u(3)*(1-cos(theta))+u(2)*sin(theta);
    u(2)*u(1)*(1-cos(theta))+u(3)*sin(theta) cos(theta)+u(2)^2*(1-cos(theta)) u(2)*u(3)*(1-cos(theta))-u(1)*sin(theta);
    u(3)*u(1)*(1-cos(theta))-u(2)*sin(theta) u(3)*u(2)*(1-cos(theta))+u(1)*sin(theta) cos(theta)+u(3)^2*(1-cos(theta))];
end