N=21;
A1=[
    1.0251851     0.021408467   -0.5557318    -0.16318431   0.15553209    0.016790447   

    ];
A2=[
   0.26073162    -0.1322251    0.041222103   -0.025522432  0.045175682   0.0048769349  
    ];

A1(numel(A1)+1:max(numel(A1),numel(A2)))=0;
A2(numel(A2)+1:max(numel(A1),numel(A2)))=0;
A=[A1;A2];
B=2*A([2 1],:).*(-1).^(0:size(A,2)-1).*[1; -1];
A=1*A;

fprintf('%d\t',sum(A(1,:)~=0));
for i=1:sum(A(1,:)~=0)
    fprintf('%-11.8g',A(1,i));
end
fprintf('\n');
fprintf('%d\t',sum(A(2,:)~=0));
for i=1:sum(A(2,:)~=0)
    fprintf('%-11.8g',A(2,i));
end
fprintf('\n');

fprintf('%d\t',sum(B(1,:)~=0));
for i=1:sum(B(1,:)~=0)
    fprintf('%-11.8g',B(1,i));
end
fprintf('\n');
fprintf('%d\t',sum(B(2,:)~=0));
for i=1:sum(B(2,:)~=0)
    fprintf('%-11.8g',B(2,i));
end
fprintf('\n');

eps=1e-5;
Af=[A(:,end:-1:1).*[1; 1j] A.*[1; -1j]];
Bf=[B(:,end:-1:1).*[1; -1j] B.*[1; 1j]];
check_ar = conv(Af(1,:),Bf(1,:))+conv(Af(2,:),Bf(2,:));
true_ar = [zeros(1,(numel(check_ar)-1)/2) 2 zeros(1,(numel(check_ar)-1)/2)];
if all(abs(check_ar-true_ar)<eps)
    disp('Checked');
else
    disp('Mistake!');
end

syms x;
Ax=Af*exp(1j*pi*x).^(-size(A,2)+1/2:size(A,2)-1/2).';
Bx=Bf*exp(1j*pi*x).^(-size(B,2)+1/2:size(B,2)-1/2).';

% figure();
subplot(2,1,1);
fplot(real(Ax),[0, 1]);
hold on;
plot([0 1],[0 0],'--');
hold off;
legend({'ФНЧ', 'ФВЧ'});
title(num2str(N));    

subplot(2,1,2);
fplot(real(Bx),[0 1]);
hold on;
plot([0 1],[0 0],'--');
hold off;
legend({'ФНЧ', 'ФВЧ'});
saveas(gcf,['Frequency response/2even_', num2str(N), '.bmp']);