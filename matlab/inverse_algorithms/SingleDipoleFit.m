function [dip,best_loc] = SingleDipoleFit(L,f)

% simple single dipole fit, assumes iid noise
[m,n] = size(L);
n_loc = n/3;
ind = reshape(1:n,3,[])';
goal_fun = zeros(n_loc,1);
s_dip = zeros(n,1);
for i=1:n_loc
    lead = L(:,ind(i,:));
    s_dip(ind(i,:)) = lead \ f;
    goal_fun(i) = norm(lead*s_dip(ind(i,:)) - f);
end

[~, best_loc] = min(goal_fun);
dip = zeros(size(s_dip));
dip(ind(best_loc,:)) = s_dip(ind(best_loc,:));

end