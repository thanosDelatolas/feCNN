function [u_eLORETA,filt] = eLORETA(varargin)

b     = varargin{1};
L     = varargin{2};
alpha = varargin{3};
[~,n] = size(L);

if length(varargin)<4 
    filt = mkfilt_eloreta(L, alpha);
else
    filt = varargin{4};
end

loc_dof    = 3;
u_eLORETA  = zeros(n,1);
n_loc      = n/loc_dof;

for i=1:n_loc
    start_ind = (i-1)*loc_dof+1;
    end_ind   = start_ind+(loc_dof-1);
    ind       = start_ind:end_ind;
    
    u_eLORETA(ind) = filt(:,ind)'*b;
end
