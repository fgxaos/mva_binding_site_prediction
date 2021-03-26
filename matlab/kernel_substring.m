function K = kernel_substring(s,t,p,lam)
    % Computes the Gram matrix of the substring kernel 
    % with the given input data.
    % Inputs:
    %   s: first string
    %   t: second string
    %   p: length of the subsequence to consider
    %   lam: lambda parameter in the substring kernel

    s = s{1};
    t = t{1};
    n = strlength(s);
    m = strlength(t);
    B = zeros(p,n,m);
    K = zeros(p,1);
    K(1) = 0;
    for i = 1 : n
        for j = 1 : m
            if s(i) == t(j)
                B(1,i,j)=lam^2;
                K(1) = K(1)+B(1,i,j);
            end
        end
    end
    K(1) = K(1)/(lam^2);
    for l = 2 : p
        K(l) = 0;
        S = zeros(n,m);
        for i = 2 : n
            for j = 2 : m
                S(i, j) = B(l-1, i, j) +lam*(S(i-1, j)+S(i, j-1))-(lam^2)*S(i-1,j-1);
                if s(i) == t(j)
                    B(l, i, j) = (lam^2)*S(i-1, j-1);
                    K(l) = K(l)+B(l,i,j);
                end
            end
        end
        K(l) = K(l)/(lam^(2*l));
    end
end

