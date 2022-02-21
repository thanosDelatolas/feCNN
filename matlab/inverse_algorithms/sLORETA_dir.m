function [u_sLORETA,s] = sLORETA_dir(b,L,alpha)
%sLORETA function
%L: leadfield
%b: data
%alpha: regularization parameter (usual value 25)
[m,n] = size(L);
n_loc = n/3;

s  = zeros(n_loc,1);

u_mns = MNE(b,L,alpha);

E = ((L*L' + alpha * eye(m)) \ L);
%Now compute the F-scores
loc_dof = 3;
for i=1:n_loc
    start_ind = (i-1)*loc_dof+1;
    end_ind = start_ind+(loc_dof-1);
    ind = start_ind:end_ind;
    for ii=1:loc_dof
        R = L(:,ind(ii))'  *  E(:,ind(ii));
        s(ind(ii)) = -u_mns(ind(ii))'*(R \ u_mns(ind(ii)));%not sure about -
    end
end

G    = L*L';
invG = inv(G + alpha * eye(m));

loc_dof    = 3;
u_sLORETA  = zeros(n,1);
n_dof      = n/loc_dof;

for i=1:n_dof
    start_ind = (i-1)*loc_dof+1;
    end_ind   = start_ind+(loc_dof-1);
    ind       = start_ind:end_ind;
    lf        = L(:,ind);
    
    filt = zeros(size(lf,2),m);
    for ii=1:size(lf,2)
        filt(ii,:) = pinv(sqrt(lf(:,ii)' * invG * lf(:,ii))) * lf(:,ii)' * invG;
    end
    u_sLORETA(ind) = filt * b;
end
