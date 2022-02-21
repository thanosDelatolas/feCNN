function u_reg = MNE(b_noisy,L,alpha)

[m n] = size(L);
u_reg =  L' * ((L*L' + alpha * eye(m,m)) \ b_noisy); %Tikhonov regularization

end