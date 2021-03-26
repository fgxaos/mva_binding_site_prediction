% Load the dataset to consider
X = readtable('../data/Xtr0.csv');
a = string(X{1,'seq'});
b = string(X{2,'seq'});
p = 4;  % length of the substring
n = 101;    % size of each sequence
f = waitbar(0,'Processing lines');
N = 2000;   % number of elements in the dataset
% Initialize Gram matrix
K = zeros(p,N,N);
tic
for i=1:N
    for j=i:N
        a = string(X{i,'seq'});
        b = string(X{j,'seq'});
        k = kernel_substring(a,b,4,0.5);
        K(:,i,j) = k;
        K(:,j,i) = k;
    end
    waitbar(i/N,f);
end
toc
close(f)
save("gram_matrices/gram_4_substr", "K")