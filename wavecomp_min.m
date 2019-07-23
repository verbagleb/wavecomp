clear;
restrictReal = true;
restrictDouble = true;
fracOutput = false;

% N_coef=6;
% N_bands=2;
% odd = 1;
% shift = [0 1];
% asymm = [0 0];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter      = [ 1   1   2   2   1   2   1   2 ];
% cond_direct.derivative  = [ 0   0   0   0   0   0   0   0 ];
% cond_direct.point       = [ 0   1   1   0   0.875 0.25 0.75 0.125 ];
% cond_direct.value       = [ 1   0   1   0   0   0   0   0 ];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [ ];
% cond_inverse.derivative = [ ];
% cond_inverse.point      = [ ];
% cond_inverse.value      = [ ];
% N_extra_cond = 0;

% N_coef=6;
% N_bands=2;
% odd = 0;
% shift = [0 0];
% asymm = [0 1];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter      = [ 1      2           ];
% cond_direct.derivative  = [ 0      0           ];
% cond_direct.point       = [ 0      1           ];
% cond_direct.value       = [ 1      1           ];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [  ];
% cond_inverse.derivative = [  ];
% cond_inverse.point      = [  ];
% cond_inverse.value      = [  ];
% N_extra_cond = 5;
% h_init=[0.6222862  	-0.103542  	-0.0622286 	0.041691   	0.0098639  	-0.0080704 	0.4229736  	-0.1409912 	-0.0422974 	0.0302124  	0.0046997  	-0.0038452 	];

% N_coef=2;
% N_bands=4;
% odd = 0;
% shift = [0 0 0 0];
% asymm = [0 1 0 1];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter     = [ 1   1    2   2      3    3     4   4];
% cond_direct.derivative = [ 0   2    0   0      0    0     0   2];
% cond_direct.point      = [ 0   0    1/3 1      2/3  0     1   1];
% cond_direct.value      = [ 1   0    1   0      1    0     1   0];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [  ];
% cond_inverse.derivative = [  ];
% cond_inverse.point      = [  ];
% cond_inverse.value      = [  ];

N_coef=2;
N_bands=3;
odd = 1;
shift = [0 0 0];
asymm = [0 1 0];   %антисимметричность фильтра каждой полосы
      %условия для на произодные прямых фильтров в заданных точках
cond_direct.filter     = [ 1   1   2   3   3];
cond_direct.derivative = [ 0   0   0   0   0];
cond_direct.point      = [ 0   1   1/2 0   1];
cond_direct.value      = [ 1   0   1   0   1];
      %условия для на произодные обратных фильтров в заданных точках
cond_inverse.filter     = [ 1   3   1   1   3   3   1   1   3   3   2 ];
cond_inverse.derivative = [ 0   0   0   0   2   2   2   2   1   ];
cond_inverse.point      = [    ];
cond_inverse.value      = [    ];
N_extra_cond = 2;
h_init=[1/2 1/4 0 1/2 0 1/2 -1/4 0];

% N_coef=2;
% N_bands=5;
% odd = 1;
% shift = [0 0 0 0 0];
% asymm = [0 1 0 1 0];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter     = [ 1   1    1   2       2       3   3   3      4        4       5   5   5];
% cond_direct.derivative = [ 0   0    2   0       1       0   0   0      0        1       0   0   2];
% cond_direct.point      = [ 0   1    0   1/3     1/3     1/2 0   1      2/3      2/3     0   1   0];
% cond_direct.value      = [ 1   0    0   1       0       1   0   0      1        0       0   1   0];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [  ];
% cond_inverse.derivative = [  ];
% cond_inverse.point      = [  ];
% cond_inverse.value      = [  ];

%_______________________________________________________________%

assert(numel(asymm)==N_bands);
assert(numel(shift)==N_bands);

if odd
    h=[sym('h',[N_bands,1]).*(~asymm.') sym('h',[N_bands,N_coef+1])];
else
    h=sym('h',[N_bands,N_coef+1]);
end
h=h(:,1:end-1);
% пользовательские ограничения числа коэффициентов %
% h(1,end)=0;
% h(1,end-1)=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if restrictReal
    assume(h,'real');
else
    assume(h,'clear');
end

syms z;
if odd
    hf=[h(:,end:-1:2).*(+1j).^repmat(asymm.',1,N_coef) h.*(-1j).^repmat(asymm.',1,N_coef+1)];
    hz=hf.*repmat(z.^(-N_coef:N_coef),N_bands,1);
else
    hf=[h(:,end:-1:1).*(+1j).^repmat(asymm.',1,N_coef) h.*(-1j).^repmat(asymm.',1,N_coef)];
    hz=hf.*repmat(z.^(-N_coef+0.5:N_coef-0.5),N_bands,1);
end
H=sym('H',[N_bands,N_bands]);
for n=1:N_bands
    H(n,:)=sum(hz(:,n:N_bands:end),2).';
end
for n=1:N_bands
    H(:,n)=circshift(H(:,n),shift(n));
end

if odd
    zero = N_coef*N_bands;
else
    zero = (N_coef-0.5)*N_bands;
end

dH=det(H)*z^zero;
[det_coef,det_comp]=coeffs(dH,z);
det0=det_coef(det_comp==z^zero); % коэффициент при нулевой степени
if isempty(det0)
    error('Определитель не содержит нулевой степени при данном выборе чётности, симметрии и смещения');
end

N_coef_total=numel(symvar(h));
N_coef_det=(numel(det_comp)-1)/2;

fprintf('Коэффициентов всего: %d\nУравнений для детерминанта: %d\n',...
    N_coef_total,N_coef_det);
N_direct_cond=size(cond_direct.filter,2);
fprintf('Условий на спектр прямых фильтров: %d\n', N_direct_cond);
N_inverse_cond=size(cond_inverse.filter,2);
fprintf('Условий на спектр обратных фильтров: %d\n', N_inverse_cond);
fprintf('Дополнительных условий: %d\n', N_extra_cond);
if (N_coef_total~=N_coef_det+N_direct_cond+N_inverse_cond+N_extra_cond)
    error('Ошибка в количестве условий');
end
fprintf('Также наложено условие на неравенство нулю коэффициента при нулевой степени детерминанта\n\n');

% Условия на определитель
eqns=sym('eq',[N_coef_total,1]);
eqns(1:N_coef_det)=det_coef(1:N_coef_det).'==0;
ineq_vect=@(u)double(subs(det_coef(1:N_coef_det).',symvar(h),u))/1j;
nonlincond = @(u)deal([],ineq_vect(u));

% Условия на прямые фильтры
syms x;
hx=subs( sum(hz,2),z,exp(1j*pi*x)); 
for i=1:N_direct_cond
    eqns(N_coef_det+i)=subs(diff(hx(cond_direct.filter(i)),cond_direct.derivative(i)),cond_direct.point(i))==cond_direct.value(i);
end
[Aeq,beq]=equationsToMatrix(eqns(N_coef_det+(1:N_direct_cond)),symvar(h));
Aeq=double(Aeq);
beq=double(beq);

% Искуственное решение вместо Kht=H\ones(N_bands,1);
DH=repmat(H,1,1,N_bands);
Kht=sym('Kht',[N_bands,1]);
for i=1:N_bands
    DH(:,i,i)=ones(N_bands,1);
    Kht(i)=det(DH(:,:,i))/det0;
end

% Условия на обратные фильтры
Khtx=subs(Kht,z,exp(1j*pi*x)); 
for i=1:N_inverse_cond
    eqns(N_coef_det+N_direct_cond+i)=subs(diff(Khtx(cond_inverse.filter(i)),cond_inverse.derivative(i)),cond_inverse.point(i))==cond_inverse.value(i);
end
% func ?

% Дополнительные условия
% eqns(N_coef_det+N_direct_cond+N_inverse_cond+1)=[hf(1,:) []]*[[] hf(2,:)].'==0;
eqns=eqns(1:end-N_extra_cond);

% Условие на наличие свободного члена определителя
eqns(end+1)=det_coef(det_comp==z^zero)~=0;

% s=solve(eqns,'ReturnConditions', true);
% sh=rmfield(s,{'parameters','conditions'});

% if (~isempty(s.parameters))
%     error('Ошибка. Наложенные условия неоднозначно определяют решение (решение содержит параметры)');
% end
% n_solutions=numel(sh.h1_1);
% if (n_solutions==0)
%     error('Ошибка. Система не имеет решений');
% end

int_h = int((hx(1)*hx(3))^2,[-1 1]);
int_f=@(u)double(subs(int_h,symvar(h),u));
% int_f=@(u) integral(@(y)real(double(subs(subs(abs(prod(hx)),symvar(h),u),y))),-1,1);

[h_m, fval] =fmincon(int_f, h_init,[],[],Aeq,beq,[],[],nonlincond);
disp('Значение интеграла:');
disp(fval);
% shf = fieldnames(sh);
% for i=1:numel(shf)
%     sh=setfield(sh, shf{i}, subs(getfield(sh, shf{i}),s.parameters,[parmin parmin]));
% end

% if restrictDouble
%     for i=1:numel(shf)
%         sh=setfield(sh, shf{i}, real(double(getfield(sh, shf{i}))));
%     end
% end

% for i_solution=1:n_solutions    
    
%     sol_str=sprintf('Решение %d/%d',i_solution,n_solutions);
    sol_str=sprintf('Локальный минимум');
    fprintf('%s:\n',sol_str);
     
%     hs=subs(h,sh);
%     hs=hs(i_solution:n_solutions:end,:);
    hs = subs(h, symvar(h), h_m);
    disp('Коэффициенты прямого фильтра (правая часть):');
    if fracOutput 
        disp(real(hs));
    else
%         disp(double(hs));%./repmat(-1j.^asymm.',1,size(hs,2))));
        for j=1:size(hs,1)
            for i=1:size(hs,2)
                fprintf('%-14.8g',double(hs(j,i)));
            end
            fprintf('\n');
        end
    end
    
%     hxs=subs(hx,sh);
%     hxs=hxs(i_solution:n_solutions:end,:);
    hxs = subs(hx, symvar(h), h_m);
    
    if odd
        Kzero = N_coef*(N_bands-1);
        Klength = (N_bands-1)*(2*N_coef-N_bands+2)/2+1; %максимальный размер
    else
        Kzero = (N_coef-0.5)*(N_bands-1);
        Klength = (N_bands-1)*(2*N_coef-N_bands+1)/2+0.5; %максимальный размер
    end
%     Hs=subs(H,sh);
%     Hs=Hs(i_solution:n_solutions:end,:);
    Hs = subs(H, symvar(h), h_m);
%     Kh=Hs\ones(N_bands,1);
%     Kh=subs(Kht,sh);
%     Kh=Kh(i_solution:n_solutions:end,:);
    Kh = subs(Kht, symvar(h), h_m);
    %
    Kh_shifted=Kh*z^Kzero;
    kh=zeros(N_bands,Klength);
    for i=1:N_bands
        Kh_pol=sym2poly(Kh_shifted(i));
        kh(i,1:floor(numel(Kh_pol)-Kzero))=Kh_pol(floor(numel(Kh_pol)-Kzero):-1:1);
    end
    disp('Коэффициенты обратного фильтра (правая часть):');
    if fracOutput
        disp(real(sym(kh)./repmat(1j.^asymm.',1,size(kh,2))));
    else
        kh_out=kh./repmat(1j.^asymm.',1,size(kh,2));
%         disp(kh_out);
        for j=1:size(kh_out,1)
            for i=1:size(kh_out,2)
                fprintf('%-14.8g',kh_out(j,i));
            end
            fprintf('\n');
        end
    end
    
    figure();
    subplot(2,1,1);
    fplot(real(hxs),[0, 1]);
    hold on;
    plot([0 1],[0 0],'--');
    hold off;
    title(sol_str);    
    
    Khx=subs(Kh,z,exp(1j*pi*x));
    subplot(2,1,2);
    fplot(real(Khx),[0 1]);
    hold on;
    plot([0 1],[0 0],'--');
    hold off;
% end