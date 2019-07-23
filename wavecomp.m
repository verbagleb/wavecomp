clear all;
numerical = false;  %symbolic if false
% if symbolic
    hardSolution = true;    % разделение системы на линейную и нелинейную части
                            % (желательно использовать)
    restrictReal = true;    % отбросить комплексные решения
    restrictDouble = true;  % использовать численное приближение решения на выводе и для получения обратного фильтра
    % if not
        fracOutput = false; % вывод в виде символьных выражений
% if numerical
    iter_max=100;           % максимальное число итераций численного метода решения
    eps = 1e-3;             % погрешность метрики сравнения полученных решения
    n_iter = 5;             % количество начальных точек

% В число N_coef входят коэффициенты строго справа от 0
% В число N_corr входят коэффициенты строго справа от 0. 0 - полная

% N_coef=6;
% N_bands=2;
% odd = 1;
% shift = [0 1];
% asymm = [0 0];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter      = [ 1   1   2   2   1   2   1   2 ];
% cond_direct.derivative  = [ 0   0   0   0   2   2   4   4 ];
% cond_direct.point       = [ 0   1   1   0   1   0   1   0 ];
% cond_direct.value       = [ 1   0   1   0   0   0   0   0 ];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [ ];
% cond_inverse.derivative = [ ];
% cond_inverse.point      = [ ];
% cond_inverse.value      = [ ];
% N_extra_cond = 0;
% direct = true;
% N_corr = 0;
% aa = 0.00;

% N_coef=4;
% N_bands=2;
% odd = 0;
% shift = [0 0];
% asymm = [0 1];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter      = [ 1   1   2   2        ];
% cond_direct.derivative  = [ 0   2   0   1        ];
% cond_direct.point       = [ 0   0   1   0        ];
% cond_direct.value       = [ 1   0   1   0        ];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [  ];
% cond_inverse.derivative = [  ];
% cond_inverse.point      = [  ];
% cond_inverse.value      = [  ];
% N_extra_cond = 1;

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

% N_coef=5;
% N_bands=3;
% odd = 1;
% shift = [0 0 0];
% asymm = [0 1 0];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter     = [ 1  2   3   ];
% cond_direct.derivative = [ 0  0   0   ];
% cond_direct.point      = [ 1  .5  0   ];
% cond_direct.value      = [ 0  1   0   ];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [ 1   1   3   3   1   1   3   3   2 ];
% cond_inverse.derivative = [ 0   0   0   0   2   2   2   2   1 ];
% cond_inverse.point      = [ 0   1   0   1   0   1   0   1   0 ];
% cond_inverse.value      = [ 1   0   0   1   0   0   0   0   0 ];
% N_extra_cond = 1;
% aa = 0.09;

% N_coef=3;
% N_bands=3;
% odd = 1;
% shift = [0 0 0];
% asymm = [0 1 0];   %антисимметричность фильтра каждой полосы
%       %условия для на произодные прямых фильтров в заданных точках
% cond_direct.filter     = [ 1   1   2   3   3,  1  2  2  3 ];
% cond_direct.derivative = [ 0   0   0   0   0,  2  1  1  2 ];
% cond_direct.point      = [ 1   0   1/2 0   1,  1  0  1  0 ];
% cond_direct.value      = [ 0   1   1   0   1,  0  0  0  0 ];
%       %условия для на произодные обратных фильтров в заданных точках
% cond_inverse.filter     = [ ];
% cond_inverse.derivative = [ ];
% cond_inverse.point      = [ ];
% cond_inverse.value      = [ ];
% N_extra_cond = 0;
% direct = false;
% N_corr = 0;
% aa = 0.01;

N_coef=5;
N_bands=3;
odd = 1;
shift = [0 0 0];
asymm = [0 1 0];   %антисимметричность фильтра каждой полосы
      %условия для на произодные прямых фильтров в заданных точках
cond_direct.filter     = [ 1   1   2   3   3,  1  1  2  2  3  3  3   ];
cond_direct.derivative = [ 0   0   0   0   0,  2  2  1  1  2  2  4   ];
cond_direct.point      = [ 1   0   1/2 0   1,  0  1  0  1  0  1  0   ];
cond_direct.value      = [ 0   1   1   0   1,  0  0  0  0  0  0  0   ];
      %условия для на произодные обратных фильтров в заданных точках
cond_inverse.filter     = [ ];
cond_inverse.derivative = [ ];
cond_inverse.point      = [ ];
cond_inverse.value      = [ ];
N_extra_cond = 0;
direct = false;
N_corr = 1;
aa = 0.001;

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
% h(2,3)=0;
% h(3,end)=0;
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
eqns_det=sym('eq',[N_coef_det,1]);
if ~numerical
    eqns_det =              det_coef(1:N_coef_det).'==0;
else
    eq_det_vect=@(u)(subs(  det_coef(1:N_coef_det).',   symvar(h),u))/1j;
end

% Условия на прямые фильтры
syms x;
hx=subs( sum(hz,2),z,exp(1j*pi*x)); 
eqns_direct=sym('eq',[N_direct_cond,1]);
eq_direct_sym=sym('eq_d_s',[N_direct_cond,1]);
for i=1:N_direct_cond
    if ~numerical
        eqns_direct(i)   = subs(diff(hx(cond_direct.filter(i)),cond_direct.derivative(i)),cond_direct.point(i))==cond_direct.value(i);
    else
        eq_direct_sym(i) = subs(diff(hx(cond_direct.filter(i)),cond_direct.derivative(i)),cond_direct.point(i)) - cond_direct.value(i);
    end
end
if numerical
    eq_direct_vect=@(u) subs(eq_direct_sym, symvar(h),u);
end

% Искуственное решение вместо Kht=H\ones(N_bands,1);
Kh = det_solve(H, det0);

eq_inverse_sym=sym('eq_i_s',[N_inverse_cond,1]);
% Условия на обратные фильтры
Khtx=subs(Kh,z,exp(1j*pi*x)); 
eqns_inverse=sym('eq',[N_inverse_cond,1]);
for i=1:N_inverse_cond
    if ~numerical
        eqns_inverse(i)   = subs(diff(Khtx(cond_inverse.filter(i)),cond_inverse.derivative(i)),cond_inverse.point(i))==cond_inverse.value(i);
    else
        eq_inverse_sym(i) = subs(diff(Khtx(cond_inverse.filter(i)),cond_inverse.derivative(i)),cond_inverse.point(i))-cond_inverse.value(i);
    end
end
if numerical
    eq_inverse_vect=@(u) subs(eq_inverse_sym, symvar(h), u);
end

% Дополнительные условия
% eqns(N_coef_det+N_direct_cond+N_inverse_cond+1)=[hf(1,:) []]*[[] hf(2,:)].'==0;
% int_h = int(prod(hx)^2,[-1 1]); % интеграл произведения
% eqns(N_coef_det+N_direct_cond+N_inverse_cond+1) = int_h ==1;
if odd
    Kzero = N_coef*(N_bands-1);
    Klength = (N_bands-1)*(2*N_coef-N_bands+2)/2+1; %максимальный размер
else
    Kzero = (N_coef-0.5)*(N_bands-1);
    Klength = (N_bands-1)*(2*N_coef-N_bands+1)/2+0.5; %максимальный размер
end
Kh_shifted=Kh*z^Kzero;
kh=sym('kh',[N_bands,Klength]);
for i=1:N_bands
    Kht_pol=coeffs(Kh_shifted(i),z,'all');
    kh(i,1:floor(numel(Kht_pol)-Kzero))=Kht_pol(floor(numel(Kht_pol)-Kzero):-1:1);
    khf(i,:)=Kht_pol(end:-1:1);
end

if N_corr==0
    if direct
        N_corr = N_coef;
    else
        N_corr = Klength - odd;
    end
end

eqns_extra=sym('eq',[N_extra_cond,1]);
if N_extra_cond
    if odd
        if ~numerical
            if direct
                eqns_extra    = (h(1,1)*h(end,1)+2*h(1,2:N_corr+1)*h(end,2:N_corr+1).')^2 == aa^2 * hf(1,:)*hf(1,:).' * hf(end,:)*hf(end,:).';
            else
                eqns_extra    = (kh(1,1)*kh(end,1)+2*kh(1,2:N_corr+1)*kh(end,2:N_corr+1).')^2 == aa^2 * khf(1,:)*khf(1,:).' * khf(end,:)*khf(end,:).';
            end
        else
            if direct
                eq_extra_vect = @(u) subs( abs(h(1,1)*h(end,1)+2*h(1,2:N_corr+1)*h(end,2:N_corr+1).') - aa * sqrt(hf(1,:)*hf(1,:).' * hf(end,:)*hf(end,:).'), symvar(h), u);
            else
                eq_extra_vect = @(u) subs( abs(kh(1,1)*kh(end,1)+2*kh(1,2:N_corr+1)*kh(end,2:N_corr+1).') - aa * sqrt(khf(1,:)*khf(1,:).' * khf(end,:)*khf(end,:).'), symvar(h), u);
            end
        end
    else
        if ~numerical
            if direct
                eqns_extra    = (2*h(1,1:N_corr)*h(3,1:N_corr).')^2 == aa^2 * hf(1,:)*hf(1,:).' * hf(3,:)*hf(3,:).';
            else
                eqns_extra    = (2*kh(1,1:N_corr)*kh(3,1:N_corr).')^2 == aa^2 * khf(1,:)*khf(1,:).' * khf(3,:)*khf(3,:).';
            end
        else
            if direct
                eq_extra_vect = @(u) subs( abs(2*h(1,1:N_corr)*h(3,1:N_corr).') - aa * sqrt(hf(1,:)*hf(1,:).' * hf(3,:)*hf(3,:).'), symvar(h), u);
            else
                eq_extra_vect = @(u) subs( abs(2*kh(1,1:N_corr)*kh(3,1:N_corr).') - aa * sqrt(khf(1,:)*khf(1,:).' * khf(3,:)*khf(3,:).'), symvar(h), u);
            end
        end
    end
else
    if ~numerical
    else
        eq_extra_vect = @(u)[];
    end
end

if numerical
    eq_vect = @(u)double([eq_det_vect(u); eq_direct_vect(u); eq_inverse_vect(u); eq_extra_vect(u)]);
    options=optimset('MaxIter', iter_max, 'Display', 'notify', 'useparallel' , false);
else
    eqns=   [eqns_det;eqns_direct;eqns_inverse;eqns_extra];
    eqns_l=  eqns_direct;
    eqns_nl=[eqns_det;eqns_inverse;eqns_extra];
end

h_sol_all = {};
if numerical
    for i_iter = 1:n_iter
        fprintf("Iteration %d/%d:\n", i_iter, n_iter);
        h_init = (2*rand(size(symvar(h)))-1)/1;
        h_sol=fsolve(eq_vect, h_init, options);
        eq_flag = false;
        for h_sol_i = h_sol_all
            h_d = max(abs(h_sol_i{1}-h_sol));
            if h_d < eps
                eq_flag = true;
                break;
            end
        end
        if eq_flag
            continue;
        else
            h_sol_all{end+1}=h_sol;
        end
        sol_str=sprintf('Частное решение %d', i_iter);
        fprintf('%s:\n',sol_str);
        hs = subs(h, symvar(h), h_sol);
        disp('Коэффициенты прямого фильтра (правая часть):');
        for j=1:size(hs,1)
            for i=1:size(hs,2)
                fprintf('%15.9f',double(hs(j,i)));
            end
            fprintf('\n');
        end
        fprintf('Корреляция прямых фильтров:\n    %g\n', ...
             sqrt(double(...
             (hs(1,1)*hs(3,1)+2*hs(1,2:end)*hs(3,2:end).')^2 / ...
             (hs(1,1)^2+2*hs(1,2:end)*hs(1,2:end).') / ...
             (hs(3,1)^2+2*hs(3,2:end)*hs(3,2:end).') )));
        fprintf('Частичная неортогональность прямых фильтров:\n    %g\n\n', ...
             sqrt(double(...
             (hs(1,1)*hs(3,1)+2*hs(1,2)*hs(3,2))^2 / ...
             (hs(1,1)^2+2*hs(1,2:end)*hs(1,2:end).') / ...
             (hs(3,1)^2+2*hs(3,2:end)*hs(3,2:end).') )));

        hxs = subs(hx, symvar(h), h_sol);
             
        
        if odd
            Kzero = N_coef*(N_bands-1);
            Klength = (N_bands-1)*(2*N_coef-N_bands+2)/2+1; %максимальный размер
        else
            Kzero = (N_coef-0.5)*(N_bands-1);
            Klength = (N_bands-1)*(2*N_coef-N_bands+1)/2+0.5; %максимальный размер
        end
        Hs = subs(H, symvar(h), h_sol);
        Khs = subs(Kh, symvar(h), h_sol);

        Khs_shifted=Khs*z^Kzero;
        khs=zeros(N_bands,Klength);
        for i=1:N_bands
            Kh_pol=sym2poly(Khs_shifted(i));
            khs(i,1:floor(numel(Kh_pol)-Kzero))=Kh_pol(floor(numel(Kh_pol)-Kzero):-1:1);
        end

        disp('Коэффициенты обратного фильтра (правая часть):');
        kh_out=khs./repmat(1j.^asymm.',1,size(khs,2));
        for j=1:size(kh_out,1)
            for i=1:size(kh_out,2)
                fprintf('%15.9f',kh_out(j,i));
            end
            fprintf('\n');
        end
        
        fprintf('Корреляция обратных фильтров:\n    %g\n', ...
             sqrt(double(...
             (khs(1,1)*khs(3,1)+2*khs(1,2:end)*khs(3,2:end).')^2 / ...
             (khs(1,1)^2+2*khs(1,2:end)*khs(1,2:end).') / ...
             (khs(3,1)^2+2*khs(3,2:end)*khs(3,2:end).') )));        
        fprintf('Частичная неортогональность обратных фильтров:\n    %g\n\n', ...
             (sqrt(double(...
             khs(1,1)*khs(3,1)+2*khs(1,2)*khs(3,2))^2 / ...
             (khs(1,1)^2+2*khs(1,2:end)*khs(1,2:end).') / ...
             (khs(3,1)^2+2*khs(3,2:end)*khs(3,2:end).') )))

        figure();
        subplot(2,1,1);
        fplot(real(hxs),[0, 1]);
        hold on;
        plot([0 1],[0 0],'--');
        hold off;
        title(sol_str);    

        Khx=subs(Khs,z,exp(1j*pi*x));
        subplot(2,1,2);
        fplot(real(Khx),[0 1]);
        hold on;
        plot([0 1],[0 0],'--');
        hold off;
    end
      
else
    if ~hardSolution
        s=solve(eqns,'ReturnConditions', true);
        s_h=rmfield(s,{'parameters','conditions'});
        if (~isempty(s.parameters))
            cond_s = solve(s.conditions,'ReturnConditions', true);  % ошибка: находит решения для набора условий 
                                                                    % для разных решений как для системы
            if ~isempty(cond_s.parameters)
                error('Ошибка. Наложенные условия неоднозначно определяют решение (решение содержит параметры)');
            else
                cond_s_u = rmfield(cond_s,{'parameters','conditions'});
                s_h = subs_each(s_h,cond_s_u);
            end
        end
    else
        %%% синтаксис имён: 
        % параметрическое решение с условиями: spc_value_parameters
        % параметрическое решение: s_value_parameters
        % параметры: p_parameters
        % условия на parameters: c_parameters
        spc_h_u = solve(eqns_l,symvar(h),'ReturnConditions', true);  % одно решение с параметрами
        s_h_u = rmfield(spc_h_u,{'parameters','conditions'});        % h=h(u)
        p_u = spc_h_u.parameters;
        c_u = spc_h_u.conditions;
        
        assume(c_u);
        eqns_u = subs(eqns_nl,s_h_u);                               % eqns_nl(h(u))
        spc_u_t = solve(eqns_u,p_u,'ReturnConditions', true);       % может быть много решений с параметрами
        s_u_t = rmfield(spc_u_t,{'parameters','conditions'});       % u=u(t)
        p_t = spc_u_t.parameters;
        c_t = spc_u_t.conditions;
        f_u = fieldnames(s_u_t);
        
        s_u = s_u_t;
        for i=1:numel(f_u)    % для каждой переменной u
            s_u = setfield(s_u, f_u{i}, []);
        end
        expr_u_t = subs(p_u, s_u_t);   % матрица число_решений х число_параметров
        for j=1:size(expr_u_t, 1)      % для каждого решения для u
            eqns_tj = c_t(j);
            p_tj = symvar(expr_u_t(j,:));
            if ~isempty(p_tj)
                spc_tj = solve(eqns_tj, p_tj,'ReturnConditions', true);
                s_tj = rmfield(spc_tj,{'parameters','conditions'});
                if ~isempty(spc_tj.parameters)
                    error('Ошибка. Наложенные условия неоднозначно определяют решение (решение содержит параметры)');
                else
                    for i=1:numel(f_u)    % для каждой переменной u
                        expr_ui_current = getfield(s_u, f_u{i});
                        expr_ui_t = getfield(s_u_t, f_u{i});
                        expr_ui_tj = expr_ui_t(j);
                        expr_uij = subs(expr_ui_tj, s_tj);
                        s_u = setfield(s_u, f_u{i}, [expr_ui_current; expr_uij]);
                    end
                end
            else
                    for i=1:numel(f_u)    % для каждой переменной u
                        expr_ui_current = getfield(s_u, f_u{i});
                        expr_ui_t = getfield(s_u_t, f_u{i});
                        expr_ui_tj = expr_ui_t(j);
                        s_u = setfield(s_u, f_u{i}, [expr_ui_current; expr_ui_tj]);
                    end
            end
        end
        s_h = subs_each(s_h_u, s_u);
        assert(~numel(symvar(subs(h,s_h))))
    end

    if restrictDouble
        shf = fieldnames(s_h);
        for i=1:numel(shf)
            s_h=setfield(s_h, shf{i}, real(double(getfield(s_h, shf{i}))));
        end
    end
    h_all=subs(h(:).',s_h);
    det0s_all=subs(det0,s_h);

    h_flag = zeros(size(h_all,1),1); % показывает уникальность среди предыдущих
    for i_h = 1:size(h_all,1)
        h_flag(i_h)=~any(all(h_all(1:i_h-1,:)==h_all(i_h*ones(1,i_h-1),:),2)) && ...
                det0s_all(i_h)~=0;
        continue;
    end
    n_solutions=numel(s_h.h1);
    n_solutions_uniq = sum(h_flag);
    if (n_solutions==0)
        error('Ошибка. Система не имеет решений');
    end
    
    i_solution_uniq=0;
    for i_solution=1:n_solutions    
        
        if ~h_flag(i_solution)
            continue;
        else
            i_solution_uniq=i_solution_uniq+1;
        end
        sol_str=sprintf('Решение %d/%d',i_solution_uniq,n_solutions_uniq);
        fprintf('%s:\n',sol_str);

        det0s=det0s_all(i_solution);
     
        hs=subs(h, s_h);      % матрица коэффициентов
        hs=hs(i_solution:n_solutions:end,:);
        disp('Коэффициенты прямого фильтра (правая часть):');
        if fracOutput 
            disp(real(hs));
        else
            for j=1:size(hs,1)
                for i=1:size(hs,2)
                    fprintf('%15.9f',double(hs(j,i)));
                end
                fprintf('\n');
            end
        end

        fprintf('Корреляция прямых фильтров:\n    %g\n', ...
             sqrt(double(...
             (hs(1,1)*hs(end,1)+2*hs(1,2:end)*hs(end,2:end).')^2 / ...
             (hs(1,1)^2+2*hs(1,2:end)*hs(1,2:end).') / ...
             (hs(end,1)^2+2*hs(end,2:end)*hs(end,2:end).') )));
        fprintf('Частичная неортогональность прямых фильтров:\n    %g\n\n', ...
             sqrt(double(...
             (hs(1,1)*hs(end,1)+2*hs(1,2)*hs(end,2))^2 / ...
             (hs(1,1)^2+2*hs(1,2:end)*hs(1,2:end).') / ...
             (hs(end,1)^2+2*hs(end,2:end)*hs(end,2:end).') )));
      
        hxs=subs(hx,s_h);    % h(x)
        hxs=hxs(i_solution:n_solutions:end,:);
        
        if odd
            Kzero = N_coef*(N_bands-1);
            Klength = (N_bands-1)*(2*N_coef-N_bands+2)/2+1; %максимальный размер
        else
            Kzero = (N_coef-0.5)*(N_bands-1);
            Klength = (N_bands-1)*(2*N_coef-N_bands+1)/2+0.5; %максимальный размер
        end
        Hs=subs(H,s_h);    % матрица системы
        Hs=Hs(i_solution:n_solutions:end,:);
        %  Kh=Hs\ones(N_bands,1);
        Khs = det_solve(Hs, det0s);

        Khs_shifted=Khs*z^Kzero;
        khs=zeros(N_bands,Klength);
        for i=1:N_bands
            Kh_pol=sym2poly(Khs_shifted(i));
            khs(i,1:floor(numel(Kh_pol)-Kzero))=Kh_pol(floor(numel(Kh_pol)-Kzero):-1:1);
        end
        disp('Коэффициенты обратного фильтра (правая часть):');
        if fracOutput && ~restrictDouble
            disp(real(sym(khs)./repmat(1j.^asymm.',1,size(khs,2))));
        else
            kh_out=khs./repmat(1j.^asymm.',1,size(khs,2));
            for j=1:size(kh_out,1)
                for i=1:size(kh_out,2)
                    fprintf('%15.9f',kh_out(j,i));
                end
                fprintf('\n');
            end
        end
        
        fprintf('Корреляция обратных фильтров:\n    %g\n', ...
             sqrt(double(...
             (khs(1,1)*khs(end,1)+2*khs(1,2:end)*khs(end,2:end).')^2 / ...
             (khs(1,1)^2+2*khs(1,2:end)*khs(1,2:end).') / ...
             (khs(end,1)^2+2*khs(end,2:end)*khs(end,2:end).') )));        
        fprintf('Частичная неортогональность обратных фильтров:\n    %g\n\n', ...
             sqrt(double(...
             (khs(1,1)*khs(end,1)+2*khs(1,2)*khs(end,2))^2 / ...
             (khs(1,1)^2+2*khs(1,2:end)*khs(1,2:end).') / ...
             (khs(end,1)^2+2*khs(end,2:end)*khs(end,2:end).') )));


    figure();
    subplot(2,1,1);
    fplot(real(hxs),[0, 1]);
    hold on;
    plot([0 1],[0 0],'--');
    hold off;
    title(sol_str);    

    Khx=subs(Khs,z,exp(1j*pi*x));
    subplot(2,1,2);
    fplot(real(Khx),[0 1]);
    hold on;
    plot([0 1],[0 0],'--');
    hold off;
    
    end
    
end