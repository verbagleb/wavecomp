dim=3;
N=33;
% k1=1.6963;k2=1.7026;k3=1.8351;
% k1=1.7004;k2=1.6734;k3=2.0341;
% k1=1.9;k2=1.55;k3=1.9;
k1=1;k2=1;k3=1;
Ac={[
    0.359464443    0.273207975    0.090923083   -0.017111497   -0.020655305   -0.006096479
    ]/k1 [
    0.000000000    0.441427011    0.000000000   -0.091786495    0.000000000   -0.033213505
    ]/k2 [
    0.364020790   -0.270305548    0.090652806    0.014833323   -0.022663202    0.005472226
    ]/k3 };
Bc={[
    1.275162298    0.915602491    0.173141227   -0.176727733   -0.097733965    0.020173943    0.035118304   -0.011132269   -0.000051428    0.004028280
    ]*k1 [
    0.000000000   -1.064028238   -0.000310989    0.332705268   -0.000293826   -0.006884400    0.000188861    0.003715017    0.000017162   -0.001344302
    ]*k2 [
    1.288653424   -0.903985317    0.180660307    0.182157596   -0.093777827   -0.019231312    0.034159499    0.010145980    0.000046871   -0.003671385
    ]*k3 };
assym = [0 1 0];
odd = true;

clear A B;
for i=1:dim
    A(i,1:numel(Ac{i}))=Ac{i};
    B(i,1:numel(Bc{i}))=Bc{i};
end

fprintf('\n');
for j=1:dim
    fprintf('%d\t',numel(Ac{j}));
    for i=1:numel(Ac{j})
        fprintf('%13.9g',Ac{j}(i));
    end
    fprintf('\n');
end
fprintf('\n');
for j=1:dim
    fprintf('%d\t',numel(Bc{j}));
    for i=1:numel(Bc{j})
        fprintf('%13.9g',Bc{j}(i));
    end
    fprintf('\n');
end
fprintf('\n');

eps=1e-5;
coef = (-1j).^assym;
Af=[A(:,end:-1:odd+1).*coef' A.*coef.'];
Bf=[B(:,end:-1:odd+1).*coef.' B.*coef'];
check_ar = 0;
for j=1:dim
    check_ar = check_ar + conv(Af(j,:),Bf(j,:));
end
true_ar = [zeros(1,(numel(check_ar)-1)/2) dim zeros(1,(numel(check_ar)-1)/2)];
if all(abs(check_ar-true_ar)<eps)
    disp('Checked');
else
    disp('Mistake!');
end

syms x;
border = 1/2 + odd*1/2;
Ax=Af*exp(1j*pi*x).^(-size(A,2)+border:size(A,2)-border).';
Bx=Bf*exp(1j*pi*x).^(-size(B,2)+border:size(B,2)-border).';

figure(1);
subplot(2,1,1);
fplot(real(Ax),[0, 1]);
hold on;
plot([0 1],[0 0],'--');
hold off;
legend({'ФНЧ', 'ФСЧ', 'ФВЧ'});
title(num2str(N));    

subplot(2,1,2);
fplot(real(Bx),[0 1]);
hold on;
plot([0 1],[0 0],'--');
hold off;
legend({'ФНЧ', 'ФСЧ', 'ФВЧ'});
saveas(gcf,['Frequency response/3_', num2str(N), '.bmp']);

figure(2)
fplot(real(Ax).'*real(Bx),[0, 1]);