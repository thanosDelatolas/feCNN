function u_reg = WMNE(b_noisy,L,alpha,weighting,fMRI,beta)

[m n] = size(L);
n_loc = n/3;


switch weighting
    case 'L2'
        weight_vec = sqrt(sum(L.^2,1))';
    case 'L1'
        weight_vec = sum(abs(L),1)';
    case 'Linf'
        weight_vec = sqrt(max(abs(L),[],1))';
    case 'L22'
        aux = sum(L.^2,1)';
        aux = reshape(aux,3,[]);
        aux = sqrt(sum(aux,1));
        aux = repmat(aux,3,1);
        weight_vec = aux(:);
    case 'L21'
        % take the L2 norm of the rows of a 3-Lead and then
        % the L1 norm over the 1-Lead
        loc_dof = 3;
        for i=1:n_loc
            start_ind = (i-1)*loc_dof+1;
            end_ind = start_ind+(loc_dof-1);
            LeadHere = L(:,start_ind:end_ind);
            LeadHereL2 = sqrt(sum(LeadHere.^2,2));
            Wloc = sum(abs(LeadHereL2));
            weight_vec(start_ind:end_ind) = Wloc;
        end
    case 'L2inf'
        % take the L2 norm of the rows of a 3-Lead and then
        % the Linf norm over the 1-Lead
        loc_dof = 3;
        for i=1:n_loc
            start_ind = (i-1)*loc_dof+1;
            end_ind = start_ind+(loc_dof-1);
            LeadHere = L(:,start_ind:end_ind);
            LeadHereL2 = sqrt(sum(LeadHere.^2,2));
            Wloc = max(LeadHereL2);
            weight_vec(start_ind:end_ind) = Wloc;
        end
    case 'fMRI'
        v = fMRI(:) + 0.1;
        V = [v v v]';
        weight_vec = V(:);
    case 'Liu'
        fMRIbin = (fMRI > 0.5 * max(fMRI(:)));
        fMRIlist = fMRIbin(:);
        % beta is scaling facto related to the square of the expected
        % source amplitude
        v = ((1 - fMRIlist)*beta + fMRIlist); 
        V = [v v v]';
        weight_vec = V(:);
    otherwise
        error('invalid weighting')
end



Winv = spdiags(1./weight_vec(:),0,n,n);
u_reg =  Winv *(L' * (((L * (Winv*Winv)) * L' + alpha * eye(m,m)) \ b_noisy));
u_reg = Winv * u_reg;

end