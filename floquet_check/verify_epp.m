function verify_epp(N, M)
% VERIFY_EPP  Independent marginal-stability check of the Floquet base.
%
% For a set of tabulated critical points (E, k_c, Ta_c) of the base, and
% for BOTH wall conditions
%     epp = +1 : cylinders oscillating IN PHASE  (co-oscillating)
%     epp = -1 : cylinders oscillating in OPPOSITION (counter-oscillating)
% this script
%   (1) evaluates the dominant Floquet multiplier |mu| at the tabulated
%       Ta_c  (a marginal point must give |mu| = 1),
%   (2) refines Ta to marginality by fzero and reports the deviation
%       from the tabulated value.
% The operator is the one of mode_multipliers3.m (verbatim), the
% monodromy matrix being integrated by fixed-step RK4 over one period
% (M steps), at N Chebyshev points.  Defaults: N = 50, M = 100, the
% production resolution of the base.
%
% Output: verify_epp_N<N>_M<M>.csv next to this file, columns
%   E, k, Ta_tab, epp, absmu_at_Ta_tab, Ta_star, dev_pct, absmu_star, arg_over_pi
%
% Usage (batch):  matlab -batch "cd('<this folder>'); verify_epp(50,100)"
if nargin < 1, N = 50; end
if nargin < 2, M = 100; end
om = 50.0; S = 1.0; eps = 0.14;                  % gamma = 5, UCM, d/R1 = 0.14
% (E, k_c, Ta_c) from the base; 'sign' = sign of the raw E_*.csv file
pts = [
    0.0008   5.0  179.1245   -1
    0.0033   4.9  163.0770   +1
    0.0171  10.6   82.1466   -1
    0.1286  15.6   26.1722   +1
    0.7274  13.4   31.6522   -1
    1.7163  16.8   45.6353   -1
   10.0     18.6   64.8884   +1 ];
here = fileparts(mfilename('fullpath'));
out = fullfile(here, sprintf('verify_epp_N%d_M%d.csv', N, M));
fid = fopen(out, 'w');
fprintf(fid, 'E,k,Ta_tab,raw_sign,epp,absmu_at_Ta_tab,Ta_star,dev_pct,absmu_star,arg_over_pi,seconds\n');
fprintf('N = %d, M = %d, om = %g, S = %g, eps = %g\n', N, M, om, S, eps);
for j = 1:size(pts, 1)
    E = pts(j,1); kk = pts(j,2); Ta = pts(j,3); rs = pts(j,4);
    Rg = Ta / sqrt(eps);
    for pp = [+1, -1]
        t0 = tic;
        mu0 = floq_eigs(Rg, kk, om, E, S, N, M, pp, eps);
        a0 = max(abs(mu0));
        Rstar = NaN; mus = NaN; arg = NaN;
        try
            f = @(y) log(max(abs(floq_eigs(y, kk, om, E, S, N, M, pp, eps))));
            Rstar = fzero(f, [0.80*Rg, 1.20*Rg]);
        catch
            try, Rstar = fzero(f, Rg); catch, Rstar = NaN; end
        end
        if ~isnan(Rstar)
            mu = floq_eigs(Rstar, kk, om, E, S, N, M, pp, eps);
            [~, ix] = max(abs(mu)); m1 = mu(ix); mus = abs(m1); arg = angle(m1)/pi;
        end
        Tstar = Rstar * sqrt(eps); dev = 100*(Tstar - Ta)/Ta; sec = toc(t0);
        fprintf(fid, '%.6g,%.4f,%.6f,%+d,%+d,%.6f,%.6f,%.4f,%.6f,%.4f,%.1f\n', ...
            E, kk, Ta, rs, pp, a0, Tstar, dev, mus, arg, sec);
        fprintf('E=%-8.4g k=%-5.1f Ta_tab=%8.3f raw%+d  epp=%+d  |mu|(Ta_tab)=%.4f  Ta*=%8.3f  dev=%+7.2f%%  arg/pi=%+.3f  (%.0f s)\n', ...
            E, kk, Ta, rs, pp, a0, Tstar, dev, arg, sec);
    end
end
fclose(fid);
fprintf('written: %s\n', out);
end

% ---------------------------------------------------------------------
function mu = floq_eigs(Re, k, om, E, S, N, M, pp, eps)
    T = 2*pi/om;
    [x, DM] = chebdif(N, 4); s = 2; D1 = s .* DM(:, :, 1); D2 = s^2 .* DM(:, :, 2); D4 = s^4 .* DM(:, :, 4);
    i = sqrt(-1); I = eye(N); Z = zeros(N);
    RR = (1 - S) * E; gamma = sqrt(om / 2);
    C1 = (om * E)^2 + 1; C2 = (om * RR)^2 + 1; C3 = om * (E - RR);
    beta = sqrt(sqrt(C1 / C2) + C3 / C2); zeta = sqrt(sqrt(C1 / C2) - C3 / C2);
    xx = 0.5 * (1 + x); xxx = 0.5 * (1 - x); epp = pp;
    C4 = epp .* cos(beta * gamma) + cosh(zeta * gamma);
    C7 = om * S * E; C8 = ((2 * om * E)^2) + 1; C9 = Re * E; C10 = om * Re * E * E;
    V11 = cos(beta * gamma * xx) .* cosh(zeta * gamma * xxx) + epp .* cos(beta * gamma * xxx) .* cosh(zeta * gamma * xx);
    V22 = sin(beta * gamma * xx) .* sinh(zeta * gamma * xxx) + epp .* sin(beta * gamma * xxx) .* sinh(zeta * gamma * xx);
    V1 = V11 ./ C4; V2 = V22 ./ C4;
    dV1 = D1 * V1; dV2 = D1 * V2;
    T1 = (S / C1) .* dV1 - (C7 / C1) .* dV2; T2 = (C7 / C1) * dV1 + (S / C1) * dV2;
    T3 = -(C9 / C8) .* T1 .* V1 + 2 * (C10 / C8) .* T1 .* V2 + 2 * (C10 / C8) .* T2 .* V1 + (C9 / C8) .* T2 .* V2;
    T4 = (C9 / C8) .* T1 .* dV1 - 2 * (C10 / C8) .* T1 .* dV2 - 2 * (C10 / C8) .* T2 .* dV1 - (C9 / C8) * T2 .* dV2;
    T5 = -2 * (C10 / C8) .* T1 .* V1 - (C9 / C8) .* T1 .* V2 - (C9 / C8) .* T2 .* V1 - 2 * (C10 / C8) .* T2 .* V2;
    T6 = 2 * (C10 / C8) .* T1 .* dV1 + (C9 / C8) .* T1 .* dV2 + (C9 / C8) .* T2 .* dV1 - 2 * (C10 / C8) .* T2 .* dV2;
    T7 = -C9 .* T1 .* V1 - C9 .* T2 .* V2; T8 = C9 .* T1 .* dV1 + C9 .* T2 .* dV2;

    B = zeros(8*N);
    B(3:N-2,:) = [D2(3:N-2,1:N)-(k^2)*I(3:N-2,1:N), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:)];
    B(N+2:2*N-1,:) = [Z(2:N-1,1:N), I(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N)];
    for q = 3:8
        B((q-1)*N+1:q*N, (q-1)*N+1:q*N) = E*I;
    end
    c1 = [I(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:)];
    c2 = [D1(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:)];
    c3 = [I(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:)];
    c4 = [D1(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:)];
    c5 = [Z(1,:),I(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:)];
    c6 = [Z(N,:),I(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:)];
    C = [c1;c2;c3;c4;c5;c6];
    r = [1,2,N-1,N,N+1,2*N]; ki = [3:N-2,N+2:2*N-1,2*N+1:8*N];
    G = -pinv(C(:,r))*C(:,ki);
    BB = B(ki,ki) + B(ki,r)*G;
    invBB = inv(BB);
    n = 8*N-6;

    function Q = Qmat(t)
        Vi = V1 .* cos(om * t) + V2 .* sin(om * t);
        Ti = T1 .* cos(om * t) + T2 .* sin(om * t);
        Tii = (eps .* T3 + T4) .* cos(2 * om * t) + (eps .* T5 + T6) .* sin(2 * om * t) + (eps .* T7 + T8);
        Vc = diag(Vi); dVs = diag(D1*Vi); Tc = diag(Ti); dTs = diag(D1*Ti); Tcc = diag(Tii); dTss = diag(D1*Tii);
        A = zeros(8*N);
        A(3:N-2,:) = [(1-S)*(D4(3:N-2,1:N)-2*(k^2)*D2(3:N-2,1:N)+(k^4)*I(3:N-2,1:N)), ...
            -2*(k^2)*eps*Re*Vc(3:N-2,3:N-2)*I(3:N-2,:), -(k^2)*D1(3:N-2,1:N), Z(3:N-2,1:N), ...
            -i*(k^3)*I(3:N-2,1:N)-i*k*D2(3:N-2,1:N), Z(3:N-2,1:N), Z(3:N-2,1:N), (k^2)*D1(3:N-2,1:N)];
        A(N+2:2*N-1,:) = [-Re*dVs(2:N-1,2:N-1)*I(2:N-1,1:N)-eps*Re*Vc(2:N-1,2:N-1)*I(2:N-1,1:N), ...
            (1-S)*(D2(2:N-1,1:N)-(k^2)*I(2:N-1,1:N)), Z(2:N-1,1:N), D1(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), i*k*I(2:N-1,1:N), Z(2:N-1,1:N)];
        A(2*N+1:3*N,:) = [2*S*D1, Z, -I, Z, Z, Z, Z, Z];
        A(3*N+1:4*N,:) = [E*Re*Tc*D1-E*Re*dTs*I+eps*E*Re*Tc*I, S*D1, E*Re*dVs*I-eps*E*Re*Vc*I, -I, Z, Z, Z, Z];
        A(4*N+1:5*N,:) = [i*(1/k)*S*D2+i*k*S*I, Z, Z, Z, -I, Z, Z, Z];
        A(5*N+1:6*N,:) = [-E*Re*dTss*I+2*eps*E*Re*Tcc*I, 2*E*Re*Tc*D1-2*eps*E*Re*Tc*I, Z, 2*E*Re*dVs*I-2*eps*E*Re*Vc*I, Z, -I, Z, Z];
        A(6*N+1:7*N,:) = [i*(1/k)*E*Re*(Tc*D2+eps*Tc*D1-(eps^2)*Tc*I), i*k*S*I, Z, Z, E*Re*dVs*I-eps*E*Re*Vc*I, Z, -I, Z];
        A(7*N+1:8*N,:) = [-2*S*D1-2*eps*S*I, Z, Z, Z, Z, Z, Z, -I];
        AA = A(ki,ki) + A(ki,r)*G;
        Q = AA * invBB;
    end

    % exponential (Magnus, midpoint) stepping for Phi' = Q(t) Phi,
    % Phi(0) = I, over one period: Phi <- expm(h Q(t+h/2)) Phi.
    % Unconditionally stable, exact for piecewise-constant Q; the
    % explicit schemes blow up on this stiff (D^4) operator.
    h = T / M; Phi = eye(n);
    for m = 1:M
        t = (m-1)*h;
        Phi = expm(h * Qmat(t + h/2)) * Phi;
    end
    mu = eig(Phi);
end

function [x, DM] = chebdif(N, M)
I = eye(N); L = logical(I);
n1 = floor(N/2); n2 = ceil(N/2);
k = (0:N-1)'; th = k*pi/(N-1);
x = sin(pi*(N-1:-2:1-N)'/(2*(N-1)));
T = repmat(th/2, 1, N);
DX = 2*sin(T'+T).*sin(T'-T);
DX = [DX(1:n1,:); -flipud(fliplr(DX(1:n2,:)))]; DX(L) = ones(N,1);
C = toeplitz((-1).^k);
C(1,:) = C(1,:)*2; C(N,:) = C(N,:)*2; C(:,1) = C(:,1)/2; C(:,N) = C(:,N)/2;
Z = 1./DX; Z(L) = zeros(N,1);
D = eye(N);
for ell = 1:M
    D = ell*Z.*(C.*repmat(diag(D),1,N) - D);
    D(L) = -sum(D');
    DM(:,:,ell) = D;
end
end
