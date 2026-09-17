function diag_epp()
% DIAG_EPP  Why does the exponential stepping blow up when N grows?
% For one tabulated point, compare at N = 12, 16, 24:
%   - the spectrum of Q(t) at t = 0 (largest real part, norm),
%   - max|mu| over one period by expm-midpoint (M = 100, 400),
%   - max|mu| by ode45 (RelTol 1e-3), the scheme of the original code.
om = 50.0; S = 1.0; eps = 0.14; T = 2*pi/om;
cases = [0.0033 4.9 163.0770; 0.1286 15.6 26.1722];
for c = 1:size(cases,1)
    E = cases(c,1); kk = cases(c,2); Ta = cases(c,3); Re = Ta/sqrt(eps);
    fprintf('\n=== E=%g k=%g Ta=%g  epp=+1 ===\n', E, kk, Ta);
    for N = [12 16 24]
        Qf = make_Q(Re, kk, om, E, S, N, +1, eps);
        Q0 = Qf(0); ev = eig(Q0);
        [mr, im] = max(real(ev));
        fprintf('N=%2d  n=%3d  ||Q||=%.2e  max Re(eig Q(0)) = %+.3e  (Im %+.2e)  #Re>0: %d\n', ...
            N, size(Q0,1), norm(Q0,1), mr, imag(ev(im)), sum(real(ev) > 1e-6));
        for M = [100 400]
            h = T/M; Phi = eye(size(Q0,1));
            for m = 1:M, Phi = expm(h*Qf((m-0.5)*h)) * Phi; end
            fprintf('      expm M=%3d : max|mu| = %.4g\n', M, max(abs(eig(Phi))));
        end
        if N <= 24
            n = size(Q0,1); t0 = tic;
            opts = odeset('RelTol', 1e-3, 'AbsTol', 1e-6);
            rhs = @(t, F) reshape(Qf(t) * reshape(F, n, n), [], 1);
            [~, Fo] = ode45(rhs, [0 T], reshape(eye(n), [], 1), opts);
            Phi = reshape(Fo(end,:), n, n);
            fprintf('      ode45 RelTol 1e-3 : max|mu| = %.4g   (%.0f s)\n', max(abs(eig(Phi))), toc(t0));
        end
    end
end
end

function Qf = make_Q(Re, k, om, E, S, N, pp, eps)
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
    V1 = V11 ./ C4; V2 = V22 ./ C4; dV1 = D1 * V1; dV2 = D1 * V2;
    T1 = (S / C1) .* dV1 - (C7 / C1) .* dV2; T2 = (C7 / C1) * dV1 + (S / C1) * dV2;
    T3 = -(C9 / C8) .* T1 .* V1 + 2 * (C10 / C8) .* T1 .* V2 + 2 * (C10 / C8) .* T2 .* V1 + (C9 / C8) .* T2 .* V2;
    T4 = (C9 / C8) .* T1 .* dV1 - 2 * (C10 / C8) .* T1 .* dV2 - 2 * (C10 / C8) .* T2 .* dV1 - (C9 / C8) * T2 .* dV2;
    T5 = -2 * (C10 / C8) .* T1 .* V1 - (C9 / C8) .* T1 .* V2 - (C9 / C8) .* T2 .* V1 - 2 * (C10 / C8) .* T2 .* V2;
    T6 = 2 * (C10 / C8) .* T1 .* dV1 + (C9 / C8) .* T1 .* dV2 + (C9 / C8) .* T2 .* dV1 - 2 * (C10 / C8) .* T2 .* dV2;
    T7 = -C9 .* T1 .* V1 - C9 .* T2 .* V2; T8 = C9 .* T1 .* dV1 + C9 .* T2 .* dV2;
    B = zeros(8*N);
    B(3:N-2,:) = [D2(3:N-2,1:N)-(k^2)*I(3:N-2,1:N), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:), Z(3:N-2,:)];
    B(N+2:2*N-1,:) = [Z(2:N-1,1:N), I(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N)];
    for q = 3:8, B((q-1)*N+1:q*N, (q-1)*N+1:q*N) = E*I; end
    c1 = [I(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:)]; c2 = [D1(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:)];
    c3 = [I(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:)]; c4 = [D1(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:)];
    c5 = [Z(1,:),I(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:),Z(1,:)]; c6 = [Z(N,:),I(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:),Z(N,:)];
    C = [c1;c2;c3;c4;c5;c6];
    r = [1,2,N-1,N,N+1,2*N]; ki = [3:N-2,N+2:2*N-1,2*N+1:8*N];
    G = -pinv(C(:,r))*C(:,ki);
    BB = B(ki,ki) + B(ki,r)*G; invBB = inv(BB);
    Qf = @(t) Qof(t);
    function Q = Qof(t)
        Vi = V1 .* cos(om * t) + V2 .* sin(om * t); Ti = T1 .* cos(om * t) + T2 .* sin(om * t);
        Tii = (eps .* T3 + T4) .* cos(2 * om * t) + (eps .* T5 + T6) .* sin(2 * om * t) + (eps .* T7 + T8);
        Vc = diag(Vi); dVs = diag(D1*Vi); Tc = diag(Ti); dTs = diag(D1*Ti); Tcc = diag(Tii); dTss = diag(D1*Tii);
        A = zeros(8*N);
        A(3:N-2,:) = [(1-S)*(D4(3:N-2,1:N)-2*(k^2)*D2(3:N-2,1:N)+(k^4)*I(3:N-2,1:N)), -2*(k^2)*eps*Re*Vc(3:N-2,3:N-2)*I(3:N-2,:), -(k^2)*D1(3:N-2,1:N), Z(3:N-2,1:N), -i*(k^3)*I(3:N-2,1:N)-i*k*D2(3:N-2,1:N), Z(3:N-2,1:N), Z(3:N-2,1:N), (k^2)*D1(3:N-2,1:N)];
        A(N+2:2*N-1,:) = [-Re*dVs(2:N-1,2:N-1)*I(2:N-1,1:N)-eps*Re*Vc(2:N-1,2:N-1)*I(2:N-1,1:N), (1-S)*(D2(2:N-1,1:N)-(k^2)*I(2:N-1,1:N)), Z(2:N-1,1:N), D1(2:N-1,1:N), Z(2:N-1,1:N), Z(2:N-1,1:N), i*k*I(2:N-1,1:N), Z(2:N-1,1:N)];
        A(2*N+1:3*N,:) = [2*S*D1, Z, -I, Z, Z, Z, Z, Z];
        A(3*N+1:4*N,:) = [E*Re*Tc*D1-E*Re*dTs*I+eps*E*Re*Tc*I, S*D1, E*Re*dVs*I-eps*E*Re*Vc*I, -I, Z, Z, Z, Z];
        A(4*N+1:5*N,:) = [i*(1/k)*S*D2+i*k*S*I, Z, Z, Z, -I, Z, Z, Z];
        A(5*N+1:6*N,:) = [-E*Re*dTss*I+2*eps*E*Re*Tcc*I, 2*E*Re*Tc*D1-2*eps*E*Re*Tc*I, Z, 2*E*Re*dVs*I-2*eps*E*Re*Vc*I, Z, -I, Z, Z];
        A(6*N+1:7*N,:) = [i*(1/k)*E*Re*(Tc*D2+eps*Tc*D1-(eps^2)*Tc*I), i*k*S*I, Z, Z, E*Re*dVs*I-eps*E*Re*Vc*I, Z, -I, Z];
        A(7*N+1:8*N,:) = [-2*S*D1-2*eps*S*I, Z, Z, Z, Z, Z, Z, -I];
        AA = A(ki,ki) + A(ki,r)*G;
        Q = AA * invBB;
    end
end

function [x, DM] = chebdif(N, M)
I = eye(N); L = logical(I); n1 = floor(N/2); n2 = ceil(N/2);
k = (0:N-1)'; th = k*pi/(N-1); x = sin(pi*(N-1:-2:1-N)'/(2*(N-1)));
T = repmat(th/2, 1, N); DX = 2*sin(T'+T).*sin(T'-T);
DX = [DX(1:n1,:); -flipud(fliplr(DX(1:n2,:)))]; DX(L) = ones(N,1);
C = toeplitz((-1).^k); C(1,:) = C(1,:)*2; C(N,:) = C(N,:)*2; C(:,1) = C(:,1)/2; C(:,N) = C(:,N)/2;
Z = 1./DX; Z(L) = zeros(N,1); D = eye(N);
for ell = 1:M
    D = ell*Z.*(C.*repmat(diag(D),1,N) - D); D(L) = -sum(D'); DM(:,:,ell) = D;
end
end
