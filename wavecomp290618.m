N_coef=2;
N_bands=3;
asymm=[0 1 0];   %антисимметричность фильтра каждой полосы
      %условия для на произодные прямых фильтров в заданных точках
cond_direct.filter     = [ 1   1   2   3   3];
cond_direct.derivative = [ 0   0   0   0   0];
cond_direct.point      = [ 0   1   1/2 0   1];
cond_direct.value      = [ 1   0   1   0   1];
      %условия для на произодные обратных фильтров в заданных точках
cond_inverse.filter     = [ 1   3];
cond_inverse.derivative = [ 0   0];
cond_inverse.point      = [ 1   0];
cond_inverse.value      = [ 0   0];
%_______________________________________________________________%

assert(numel(asymm)==N_bands);
zero = N_coef*N_bands;
left = N_bands*(N_bands-1)/2;
if (mod(N_bands,2)==1)
    N_coef_total=N_coef*N_bands+sum(~asymm);
    N_coef_det=(zero-left)/N_bands;
    fprintf('Коэффициентов всего: %d\nУравнений для детерминанта: %d\n',...
        N_coef_total,N_coef_det);
else
    error('Ошибка. Для чётных N_bands не готово');
    N_coef_total=N_coef*N_bands
end
N_direct_cond=size(cond_direct.filter,2);
fprintf('Условий на спектр прямых фильтров: %d\n', N_direct_cond);
N_inverse_cond=size(cond_inverse.filter,2);
fprintf('Условий на спектр обратных фильтров: %d\n', N_inverse_cond);
if (N_coef_total~=N_coef_det+N_direct_cond+N_inverse_cond)
    error('Ошибка в количестве условий');
end
fprintf('Также наложено условие на неравенство нулю коэффициента при нулевой степени детерминанта\n\n');

syms z;
h=[sym('h',[N_bands,1]).*(~asymm.') sym('h',[N_bands,N_coef])];%./repmat(1j.^asymm.',1,N_coef+1);
if (N_coef_total~=numel(symvar(h)))
    error('Ошибка составления матрицы коэффициентов. Возможно, некоторые имена совпадают');
end
hz=[h(:,end:-1:2).*(-1).^repmat(asymm.',1,N_coef) h].*repmat(z.^(-N_coef:N_coef),N_bands,1);
H=sym('H',[N_bands,N_bands]);
for n=1:N_bands
    H(n,:)=sum(hz(:,n:N_bands:end),2).';
end

dH=det(H)*z^zero;
[det_coef,det_comp]=coeffs(dH,z);
    %det_coef(det_comp==z^zero) - коэффициент при нулевой степени
    %det_coef(det_comp==z^left) - коэффициент при наименшьей степени
eqns=sym('eq',[N_coef_total,1]);
for i=1:N_coef_det
    eqns(i)=det_coef(det_comp==z^(zero-N_bands*i))==0;
end

syms x;
hx=subs( sum(hz,2),z,exp(1j*pi*x)); 
for i=1:N_direct_cond
    eqns(N_coef_det+i)=subs(diff(hx(cond_direct.filter(i)),cond_direct.derivative(i)),cond_direct.point(i))==cond_direct.value(i);
end

% Искуственное решение вместо Kht=H\ones(N_bands,1);
DH=repmat(H,1,1,N_bands);
det0=det_coef(det_comp==z^zero);
Kht=sym('Kht',[N_bands,1]);
for i=1:N_bands
    DH(:,i,i)=ones(3,1);
    Kht(i)=det(DH(:,:,i))/det0;
end
%

Khtx=subs(Kht,z,exp(1j*pi*x)); 
for i=1:N_inverse_cond
    eqns(N_coef_det+N_direct_cond+i)=subs(diff(Khtx(cond_inverse.filter(i)),cond_inverse.derivative(i)),cond_inverse.point(i))==cond_inverse.value(i);
end

eqns(end+1)=det_coef(det_comp==z^zero)~=0;
s=solve(eqns,'ReturnConditions', true);

if (~isempty(s.parameters))
    error('Ошибка. Наложенные условия неоднозначно определяют решение (решение содержит параметры)');
end
sh=rmfield(s,{'parameters','conditions'});
n_solutions=numel(sh.h1);
if (n_solutions==0)
    error('Ошибка. Система не имеет решений');
end

for i_solution=1:n_solutions    
    
    sol_str=sprintf('Решение %d/%d',i_solution,n_solutions);
    fprintf('%s:\n',sol_str);
     
    hs=subs(h,sh);
    hs=hs(i_solution:n_solutions:end,:);
    disp('Коэффициенты прямого фильтра (правая часть):');
    disp(hs);
    
    hxs=subs(hx,sh);
    hxs=hxs(i_solution:n_solutions:end,:);
    
    Kzero = N_coef*(N_bands-1);
    Hs=subs(H,sh);
    Hs=Hs(i_solution:n_solutions:end,:);
    %Kh=Hs\ones(N_bands,1);
    Kh=subs(Kht,sh);
    Kh=Kh(i_solution:n_solutions:end,:);
    %
    Kh_shifted=Kh*z^Kzero;
    kh=zeros(N_bands,(N_bands-1)*(2*N_coef-N_bands+2)/2+1); %максимальный размер
    for i=1:N_bands
        Kh_pol=sym2poly(Kh_shifted(i));
        kh(i,1:numel(Kh_pol)-Kzero)=Kh_pol(end-Kzero:-1:1);
    end
    disp('Коэффициенты обратного фильтра (правая часть):');
    disp(kh);
    
    figure();
    subplot(2,1,1);
    fplot(real(hxs),[0, 1]);
    title(sol_str);    
    
    Khx=subs(Kh,z,exp(1j*pi*x));
    subplot(2,1,2);
    fplot(real(Khx),[0 1]);
    hold on;
    plot([0 1],[0 0],'--');
    hold off;
end