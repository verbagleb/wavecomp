function [ x ] = det_solve(A, detA)
%Solver for a homogenous equation with det and b=[1;1;...;1]
%   Detailed explanation goes here

    assert(size(A,1)==size(A,2));
    N_bands=size(A,1);
    DA=repmat(A,1,1,N_bands);
    x=sym('x',[N_bands,1]);
    for i=1:N_bands
        DA(:,i,i)=ones(N_bands,1);
        x(i)=det(DA(:,:,i))/detA;
    end

end

