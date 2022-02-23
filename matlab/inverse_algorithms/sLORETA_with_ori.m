function s = My_sLORETA(b,L,alpha)

% Dmne is the dipole that compute for MNE
% invS is inverse of standarized matrix S
%
% WE want to find the standarized dipole
% We know how to compute the standardized current density power:
%
%   Dmne' * invS * Dmne => Cholesky on invS =
% = Dmne' * Sch'*Sch * Dmne = (Dmne'*Sch') * (Dmne*Sch) =
% = sDmne' * sDmne, wheare the sDmne is the standarized dipole



% The dipole d
% amplitude of d = norm(d,2) = sqrt(d'*d) =>
% power = d'*d


[m n] = size(L);
n_loc = n/3;

s = zeros(n,1);

u_mns = MNE(b,L,alpha);

E = ((L*L' + alpha * eye(m)) \ L);
%Now compute the F-scores
loc_dof = 3;
for i=1:n_loc
    start_ind = (i-1)*loc_dof+1;
    end_ind = start_ind+(loc_dof-1);
    ind = start_ind:end_ind;
    
    %     R = L(:,ind)'  *  E(:,ind);
    %     s(i) = u_mns(ind)'*(R \ u_mns(ind)); % standardized current density power
    %%     s(i) = sqrt(s(i));                   % standardized current density amplitude
    %
    
    invS = inv( L(:,ind)' * E(:,ind) );
    Sch = chol(invS);
    s(ind) = Sch*u_mns(ind); % standardized  dipole
    
    
    %     s(i) = s(ind)'*s(ind);   % standardized current density power
    %     s(i) =  norm(s(ind),2);       % standardized current density amplitude
    % %    s(i) = sqrt(s(i));       % standardized current density amplitude
    
end



end

