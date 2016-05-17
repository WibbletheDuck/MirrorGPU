function [ dist, resXVT, savedX, savedV ] = mirror_gpu()
% mirror_gpu()
% Externally-scriptable version of test-particles-in-a-mirror-B-field
% simulation.  Comes with functions (below) to build distribution and
% construct the field, as well as various support functions.  By default,
% will fall back to CPU processing if compatible GPUs are not present.

    q = 1;
    m = 1;
    nt = 10000;   % # timesteps
    dt = .1;     % step length
    qE = 0;
    qmt2 = q/m*dt/2;
    
    B0 = 1; % Magnetic field base is 50 uT
    v0 = 0.00989179273; % likewise velocity base in PSL is equivalent to 25 eV
    r0 = 0.337212985; % based on Larmour radius w/ above, length base is ~0.337 m
    t0 = 0.714477319; % based on B, Larmour period ~714 ns

    target_length = 5000;   % in km
    target_z = -target_length/r0;  % negative because we're launching upwards
    long_enough = 500;
    mirror_ratio = 5;
    saved_steps = 500;

    % So Bsim=Breal/50uT, vsim=vreal/25 eV, and xsim=xreal/0.337m
    % So a 100x100x1000 simulation extent is a 33.7x33.7x337m volume
    % So dt ~71.4ns, and 1000 timesteps is 71us
    
    length_factor = target_z^2/(mirror_ratio-1); % assumes 'end point' is z=0

    x_range = 0;
    y_range = 0;
    z_range = target_z;
    v_range = [25 484 1125];%[25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, ...
    %256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, ...
    %841, 900, 961, 1024, 1089, 1156, 1225]; % linear in v
    t_dtheta = 3*pi/256; % delta for co-latitude
    t_domega = 0.001; % delta for solid angle in steradians
    %p_range = 0:pi/7:pi; %0:pi/15:pi/2;
    
    v_distrib = build_distrib(v0, x_range, y_range, z_range, v_range, t_dtheta, t_domega);
       
%    size(v_distrib)

    N_part = size(v_distrib,2);
    [ 'Simulating ' num2str(N_part) ' particles over maximum ' num2str(long_enough) ' timesteps...' ]
    N_ts = nt+2;

    % distributed() is dumb, and requires the chunking dimension to be 
    % the last one.
    v_sdivdist = distributed(v_distrib);
    
    disp('Start')
    tic

    % spmd (single program, multiple data) is a more generalized 
    % multithreaded methodology than parfor, and allows use of 
    % distributed/codistributed functionality to split up arrays
    spmd

        v_localdist = getLocalPart(v_sdivdist);
        N_dpart = size(v_localdist, 2);
        chunk_inds = globalIndices(v_sdivdist,2);

        d = gpuDevice();
        disp( [ 'Running ' num2str(N_dpart) ' particles (' num2str(chunk_inds(1)) ':' num2str(chunk_inds(end)) ') in Lab ' num2str(labindex) ' on GPU ' num2str(d.Index) '.' ] )

        % Pre-allocate result arrays on GPU
        gm_X = nan([ 3, saved_steps, N_dpart ], 'double', 'gpuArray'); % x,y,z
        gm_V = nan([ 3, saved_steps, N_dpart ], 'double', 'gpuArray'); % vx,vy,vz

        % result is x,y,z,vx,vy,vz,t,ts1,ts2
        gm_result = zeros([ 3, 3, N_dpart ], 'double', 'gpuArray'); % x,y,z

        d = gpuDevice();
        t_tmem = d.TotalMemory;
        t_umem = t_tmem-d.AvailableMemory;
        disp( [ 'Memory Used: ' num2str(t_umem/1e9) '/' num2str(t_tmem/1e9) 'GB (' num2str(t_umem/t_tmem*100) '%)' ]);
        
        active_indices = 1:N_dpart;
        length(active_indices)  

        % Get B at initial positions
        % permute() lets us slot a (3,N) data peg into a (3,M,N) hole
        gm_X(:,end-1,:) = permute(v_localdist(1:3,:),[1 3 2]);
        gm_V(:,end-1,:) = permute(v_localdist(4:6,:),[1 3 2]);
        
        % Recall all arrays are (dimension, timestep, particles)
        gm_B_x = squeeze(arrayfun(@bxcalc,gm_X(1,end-1,:),gm_X(3,end-1,:),length_factor));
        gm_B_y = squeeze(arrayfun(@bycalc,gm_X(2,end-1,:),gm_X(3,end-1,:),length_factor));
        gm_B_z = squeeze(arrayfun(@bzcalc,gm_X(3,end-1,:),length_factor));
        
        % Calculate 2nd position with Boris Mover 
        gm_v_mh = squeeze(gm_V(:,end-1,:));
        gm_v_minus = gm_v_mh + qE;

        gm_t_vec = tcalc(gm_B_x,gm_B_y,gm_B_z,qmt2);
        gm_s_vec = scalc(gm_t_vec);
        size(gm_v_minus)
        size(gm_t_vec)
        gm_v_prime = gm_v_minus + cross(gm_v_minus,gm_t_vec,1);
        gm_v_plus = gm_v_minus + cross(gm_v_prime,gm_s_vec,1);

        gm_V(:,end,:) = 0.5 .* (gm_v_mh + gm_v_plus + qE);
        gm_X(:,end,:) = gm_X(:,end-1,:) + gm_V(:,end-1,:) .* dt;

        tstep = 1;
        % Loop until all particles are done, or we've 
        % done an absurd number of timesteps.
        while ~isempty(active_indices) && (tstep <= long_enough)
            tstep = tstep + 1;
            
            % shift saved-data matrices down one row
            gm_X(:,1:end-1,:) = gm_X(:,2:end,:);
            gm_V(:,1:end-1,:) = gm_V(:,2:end,:);
                        
            if labindex == 1 && mod(tstep,100) == 0
                display(['Step ' num2str(tstep) ', ' num2str(length(active_indices)) ... 
                    ' particles active, min/max z = ' num2str(min(gm_X(3,end-1,active_indices))) '/' num2str(max(gm_X(3,end-1,active_indices))) '.'])
            end
            % Recall all arrays are (dimension, timestep, particles)
            gm_B_x = squeeze(arrayfun(@bxcalc,gm_X(1,end-1,active_indices),gm_X(3,end-1,active_indices),length_factor));
            gm_B_y = squeeze(arrayfun(@bycalc,gm_X(2,end-1,active_indices),gm_X(3,end-1,active_indices),length_factor));
            gm_B_z = squeeze(arrayfun(@bzcalc,gm_X(3,end-1,active_indices),length_factor));

            gm_v_minus = squeeze(gm_V(:,end-1,active_indices) + qE);    % half-step due to E-field
            
            gm_t_vec = tcalc(gm_B_x,gm_B_y,gm_B_z,qmt2);
            gm_s_vec = scalc(gm_t_vec);
            gm_v_prime = gm_v_minus + cross(gm_v_minus,gm_t_vec);   % these calculate the
            gm_v_plus = gm_v_minus + cross(gm_v_prime,gm_s_vec);    % B-field effects
            
            gm_V(:,end,active_indices) = gm_v_plus + qE;  % second half-step from E-field
            gm_X(:,end,active_indices) = gm_X(:,end-1,active_indices) + gm_V(:,end-1,active_indices) .* dt;
            
            % check if next z-pos passes the target plane
            strike_indices = active_indices(gm_X(3,end,active_indices) > 0); 
            if ~isempty(strike_indices)
                display([ 'Timestep ' num2str(tstep) ': ' num2str(length(strike_indices)) ' strikes.' ]);
                % interpolate absolute strike time?
                gm_result(:,1,strike_indices) = squeeze(gm_X(:,end,strike_indices));
                gm_result(:,2,strike_indices) = squeeze(gm_V(:,end,strike_indices));
                gm_result(:,3,strike_indices) = repmat([ tstep-1 ; tstep ; tstep*dt*t0 ],[1 length(strike_indices)]);
                active_indices = active_indices(~ismember(active_indices,strike_indices));
            end
        end

        t_codist_result = codistributor1d(3, codistributor1d.unsetPartition, [3, 3, N_part]);
        t_codist_saved = codistributor1d(3, codistributor1d.unsetPartition, [3, saved_steps, N_part]);
        
        % gather() to copy from GPU RAM to Main Memory
        r_divres = codistributed.build(gather(gm_result), t_codist_result, 'noCommunication');
        r_divsavX = codistributed.build(gather(gm_X), t_codist_saved, 'noCommunication');
        r_divsavV = codistributed.build(gather(gm_V), t_codist_saved, 'noCommunication');        
        
    end % spmd

    % gather() again to recombine distributed arrays
    r_result = gather(r_divres);
    r_savX = gather(r_divsavX);
    r_savV = gather(r_divsavV);

    toc
    'Stop'
    
    % results to output variables
    dist = v_distrib;
    resXVT = r_result;
    savedX = r_savX;
    savedV = r_savV;
    
end

function n = bxcalc(x,z,L_z2)
    n = -2*x*z/L_z2;
end

function n = bycalc(y,z,L_z2)
    n = -2*y*z/L_z2;
end

function n = bzcalc(z,L_z2)
    n = (1+z^2/L_z2);
end

function n = threenorm(x,y,z)

    n = sqrt(x^2+y^2+z^2);

end
                      
function n = tcalc(bx,by,bz,c)

    n = c*[ bx by bz ].';

end

function n = scalc(t)

    n = 2*t./(1 + t.^2);

end

function d = build_distrib(v0, x_range, y_range, z_range, v_range, t_dtheta, t_domega)
    % Build particle distribution

    % initial positions x y z
    % initial velocities v theta phi (magnitude, azimuth, elevation)
    %   mag 25:2000 eV, azi 0:pi, el 0:pi/2
    % input as [ x y z v theta phi ] columns in v_distrib_raw

    t_range = 0+t_dtheta:t_dtheta:pi/2-t_dtheta; % range of thetas, discard first (pole) and last (plane)
    
    angle_list = [ 0 0 ];
    for i=1:length(t_range)
        theta = t_range(i);
        for omega=0:2*pi/round(2*pi*sin(theta)*t_dtheta/t_domega):2*pi
            angle_list = [ angle_list ; theta omega ];
        end
    end
    
    v_distrib_raw = zeros(6,length(x_range)*length(y_range)*length(z_range)*length(v_range)*length(angle_list));
    vdr_ind = 1;
    for i=1:length(x_range)
        for j=1:length(y_range)
            for k=1:length(z_range)
                for l=1:length(angle_list)
                    for m=1:length(v_range)
                        v_distrib_raw(:,vdr_ind) = [ x_range(i) y_range(j) z_range(k) v_range(m) angle_list(l,1) angle_list(l,2) ];
                        vdr_ind = vdr_ind + 1;
                    end
                end
            end
        end
    end

    % Transform v_mag,theta,phi to v_x, v_y, v_z
    v_distrib = v_distrib_raw;
    t_v = sqrt(3.913903e-6*v_distrib_raw(4,:))/v0; % number is 2/(m_e*c^2) in eV^-1
    v_distrib(4,:) = t_v .* cos(v_distrib_raw(6,:)) .* cos(v_distrib_raw(5,:));
    v_distrib(5,:) = t_v .* cos(v_distrib_raw(6,:)) .* sin(v_distrib_raw(5,:));
    v_distrib(6,:) = t_v .* sin(v_distrib_raw(6,:));

    d = v_distrib;
end

function d = test_distrib()

    test_array = [  1 3 8 ;
                4 7 2 ;
                9 1 7 ;
                4 6 2 ;
                1 1 1 ];
            
    test_v = [ 0 2 1.22 ;
           0 1.9 1.22 ;
           0 2.1 1.22 ;
           0 2 1.12 ;
           0 2 1.32 ];
       
    d = shiftdim([ test_array test_v ],1);
end

function [ Xg, Yg, Zg, Bx, By, Bz ] = build_field()
    % Field Initialization
    L_xyz = 100; 
    n_xyz = 100;
    k = pi/L_xyz;
    x0 = 51; y0 = 51; z0 = 51;

    % Generate the grid of the magnetic field
    [X, Y, Z] = meshgrid(1:100, 1:100, 1:1000);
    B_xgrid = zeros(100,100,1000); B_ygrid = zeros(100,100,1000); B_zgrid = zeros(100,100,1000);
    %B_maggrid = zeros(100,100,100);
    for ii = 1:100
        for jj = 1:100
            for kk = 1:1000
    %            B_xgrid(ii,jj,kk) = 0;
    %            B_ygrid(ii,jj,kk) = 0;
    %            B_zgrid(ii,jj,kk) = -1;
                % 2.85966 factor normalizes field so max magnitude is 1
                B_xgrid(ii,jj,kk) = -(5/8) * k * sin(k*(kk-z0)) * (ii - x0) / 2.85966;
                B_ygrid(ii,jj,kk) = -(5/8) * k * sin(k*(kk-z0)) * (jj - y0) / 2.85966;
                B_zgrid(ii,jj,kk) = 5 * (1 - .5*(1 + 1/8*k^2 * ((ii - x0)^2 + (jj - y0)^2)) * cos(k*(kk-z0))) / 2.85966;
             %   B_maggrid(ii,jj,kk) = sqrt(B_xgrid(ii,jj,kk)^2 + B_ygrid(ii,jj,kk)^2 + B_zgrid(ii,jj,kk)^2);
            end
        end
    end

    Xg = X; Yg = Y; Zg = Z;
    Bx = B_xgrid; By = B_ygrid; Bz = B_zgrid;
end

function ok = selectGPUDeviceForLab()

    persistent hasGPU;

    if isempty( hasGPU )
        devIdx = mod(labindex-1,gpuDeviceCount())+1;
        try
            dev = gpuDevice( devIdx );
            hasGPU = dev.DeviceSupported;
        catch %#ok
            hasGPU = false;
        end
    end
    ok = hasGPU;
    
end
