% exact solution for 1D model

function exact_1d(xI, kxI, init_s)
    tic;

    c1 = sqrt(1 - init_s);
    c2 = sqrt(init_s);

    N = 2;
    L = 512;
    M = 9192;                    

    mass = 2000;
    sigmax = 1.0;
    fID = 1;

    dt = 0.5;
    Nstep = 20000; %ceil(7.5 / (k(1) / mass) / dt); % 5000;
    tgraph = 100;

    enable_plot = false;
    % grids
    x0 = linspace(-L/2, L/2, M)';
    dx = x0(2) - x0(1);
    dkx = 2 * pi / M / dx;
    kx0 = (-M/2:M/2-1)' * dkx;
    % construct TU on k grid
    T = kx0.^2 / 2 / mass;
    TU = exp(-1i * dt * T);
    TU = fftshift(TU);
    % construct VU
    VU = zeros(M,N,N);
    Hs = zeros(M,N,N);
    evas = zeros(M,N,N);
    evts = zeros(M,N,N);
    for j=1:M
        %{
        % TULLY 1
        A = 0.01;
        B = 1.6;
        C = 0.005;
        D = 1.0;

        H = zeros(2,2);
        x = x0(j);
        if x > 0.0 
            H(1,1) = A * (1 - exp(-B * x));
        else 
            H(1,1) = -A * (1 - exp(B * x));
        end
        H(2,2) = -H(1,1);
        H(1,2) = C * exp(-D * x * x);
        H(2,1) = H(1,2);
        %}

        % TULLY 3
        A = 6e-4;
        B = 0.1;
        C = 0.9;

        H = zeros(2,2);
        x = x0(j);
        H(1,1) = A;
        H(2,2) = -A;
        if x < 0.0
            H(1,2) = B * exp(C * x);
        else
            H(1,2) = B * (2 - exp(-C * x));
        end
        H(2,1) = H(1,2);

        VU(j,:,:) = expm(-1i * dt / 2 * H);
        Hs(j,:,:) = H;

        [evt, eva] = eig(H);
        if j > 1
            phase1 = evts(j-1,1,1) * evt(1,1) + evts(j-1,2,1) * evt(2,1);
            phase2 = evts(j-1,1,2) * evt(1,2) + evts(j-1,2,2) * evt(2,2);
            if phase1 < 0.0
                evt(:,1) = -evt(:,1);
            end
            if phase2 < 0.0
                evt(:,2) = -evt(:,2);
            end
        end
        evas(j,:,:) = eva;
        evts(j,:,:) = evt;
    end
    % Initial wavefunction -- Gaussian wavepacket on adiabats
    psiad0 = zeros(M,N);
    psiad0(:,1) = c1 * exp(1i*(kxI*x0)) .* exp(-(x0-xI).^2/sigmax^2);
    psiad0(:,2) = c2 * exp(1i*(kxI*x0)) .* exp(-(x0-xI).^2/sigmax^2);
    psiad0 = psiad0 / sqrt(sum(sum(abs(psiad0).^2)));
    % Convert to diabat
    psi0 = zeros(M,N);
    psi0(:,1) = evts(:,1,1) .* psiad0(:,1) + evts(:,1,2) .* psiad0(:,2);
    psi0(:,2) = evts(:,2,1) .* psiad0(:,1) + evts(:,2,2) .* psiad0(:,2);
    % psim & psi_k_m -- for plot
    psiad0_k(:,1) = fftshift(fft(psiad0(:,1)));
    psiad0_k(:,2) = fftshift(fft(psiad0(:,2)));
    psiad0_k = psiad0_k / sqrt(sum(sum(abs(psiad0_k).^2)));
    psiad_max = max(max(max(abs(psiad0).^2)));
    psiad_k_max = max(max(max(abs(psiad0_k).^2)));
    psi0_k(:,1) = fftshift(fft(psi0(:,1)));
    psi0_k(:,2) = fftshift(fft(psi0(:,2)));
    psi0_k = psi0_k / sqrt(sum(sum(abs(psi0_k).^2)));
    psi_max = max(max(max(abs(psi0).^2)));
    psi_k_max = max(max(max(abs(psi0_k).^2)));
    % propagate WF
    psi = psi0;
    for t=0:Nstep-1
        % exp(-iVdt/2) * |Psi> in diab repr
        psi_k(:,1) = VU(:,1,1).*psi(:,1) + VU(:,1,2).*psi(:,2);
        psi_k(:,2) = VU(:,2,1).*psi(:,1) + VU(:,2,2).*psi(:,2);
        % exp(-iTdt) * psi
        psi_k(:,1) = TU .* fft(psi_k(:,1));
        psi_k(:,2) = TU .* fft(psi_k(:,2));
        % exp(-iVdt/2) * psi
        psi_k(:,1) = ifft(psi_k(:,1));
        psi_k(:,2) = ifft(psi_k(:,2));
        psi(:,1) = VU(:,1,1).*psi_k(:,1) + VU(:,1,2).*psi_k(:,2);
        psi(:,2) = VU(:,2,1).*psi_k(:,1) + VU(:,2,2).*psi_k(:,2);
        % analysis & report
        if mod(t,tgraph) == 0
            % analysis in diab
            psi_k(:,1) = fftshift(fft(psi(:,1)));
            psi_k(:,2) = fftshift(fft(psi(:,2)));
            psi_k = psi_k / sqrt(sum(sum(abs(psi_k).^2)));
            
            norm_k1 = sum(abs(psi_k(:,1)).^2) + 1e-16;
            norm_k2 = sum(abs(psi_k(:,2)).^2) + 1e-16;

            p1x = sum(abs(psi_k(:,1)).^2 .* kx0) / norm_k1;
            p2x = sum(abs(psi_k(:,2)).^2 .* kx0) / norm_k2;

            KE = sum((abs(psi_k(:,1)).^2 + abs(psi_k(:,2)).^2) .* (kx0.^2) / 2 / mass);
            PE = sum( conj(psi(:,1)) .* Hs(:,1,1) .* psi(:,1) + conj(psi(:,1)) .* Hs(:,1,2) .* psi(:,2) + conj(psi(:,2)) .* Hs(:,2,1) .* psi(:,1) + conj(psi(:,2)) .* Hs(:,2,2) .* psi(:,2) );

            % analysis in adiab
            psiad(:,1) = conj(evts(:,1,1)) .* psi(:,1) + conj(evts(:,2,1)) .* psi(:,2);
            psiad(:,2) = conj(evts(:,1,2)) .* psi(:,1) + conj(evts(:,2,2)) .* psi(:,2);
            psiad = psiad / sqrt(sum(sum(abs(psiad).^2)));

            psiad_k(:,1) = fftshift(fft(psiad(:,1)));
            psiad_k(:,2) = fftshift(fft(psiad(:,2)));
            psiad_k = psiad_k / sqrt(sum(sum(abs(psiad_k).^2)));

            normad_k1 = sum(abs(psiad_k(:,1)).^2) + 1e-16;
            normad_k2 = sum(abs(psiad_k(:,2)).^2) + 1e-16;

            pad1x = sum(abs(psiad_k(:,1)).^2 .* kx0) / normad_k1;
            pad2x = sum(abs(psiad_k(:,2)).^2 .* kx0) / normad_k2;

            % output
            if t == 0
                fprintf(fID, '# EXACT ADIAB\n');
                fprintf(fID, '# xI = %8.4f kxI = %8.4f sigmax = %8.4f A = %8.4f init_s = %8.4f c1 = %8.4f c2 = %8.4f \n', ...
                                    xI, kxI, sigmax, A, init_s, c1, c2);
                fprintf(fID, '# L = %8.4f M = %8d dt = %8.4f Nstep = %8d tgraph = %8d\n', ...
                                    L, M, dt, Nstep, tgraph);
                fprintf(fID, '#%9s%16s%16s%16s%16s%16s%16s%16s\n', ...
                                't', 'n0t', 'n1t', 'n0r', 'n1r', 'px0', 'px1', 'Etot');
            end

            fprintf(fID, '#%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f\n', ...
                        t*dt, ...
                        sum(sum(abs(psiad_k(M/2+1:M,1).^2))), ...
                        sum(sum(abs(psiad_k(M/2+1:M,2).^2))), ...
                        sum(sum(abs(psiad_k(1:M/2,1).^2))), ...
                        sum(sum(abs(psiad_k(1:M/2,2).^2))), ...
                        p1x, p2x, ...
                        KE+PE ...
                        );
            % plot
            if enable_plot == true
                subplot(1,2,1);
                plot(x0,abs(psiad(:,1)).^2, '-b', ...
                     x0,abs(psiad(:,2)).^2, '--r');
                ylim([0, psiad_max]);
                title('Real space Adiab');

                subplot(1,2,2);
                plot(kx0,abs(psiad_k(:,1)).^2, '-b', ...
                     kx0,abs(psiad_k(:,2)).^2, '--r');
                ylim([0, psiad_k_max]);
                title('Mom space Adiab');

                drawnow;
            end

            % check end
            act_region = [-15.0,15.0];
            wall_idx = round((act_region - x0(1)) / dx);
            if sum(sum(abs(psiad(wall_idx(1):wall_idx(2), :)).^2)) < 1e-4
                fprintf('#checked end\n');
                break;
            end
        end
    end
    fprintf(fID, ' %16.10f%16.10f%16.10f%16.10f%16.10f%16.10f%16.10f\n', ...
                kxI, ...
                sum(sum(abs(psiad_k(M/2+1:M,1).^2))), ...
                sum(sum(abs(psiad_k(M/2+1:M,2).^2))), ...
                sum(sum(abs(psiad_k(1:M/2,1).^2))), ...
                sum(sum(abs(psiad_k(1:M/2,2).^2))), ...
                p1x, p2x ...
                );
    fprintf('# '); toc;
end
