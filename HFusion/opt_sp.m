function [x,M] = opt_sp(A, b, lambda, alpha_p, T)
  ndim = size(A,2);
  B = zeros(ndim-1,ndim);
  C = zeros(ndim-1,ndim);
  for i = 1:(ndim-1)
      B(i,i) = -1;
      B(i,i+1) = 1;
  end
  for i = 1:(ndim-T)
      C(i,i) = -1;
      C(i,i+T) = 1;
  end
  H = B'*B;
  I = C'*C;
  M = (1-alpha_p)*lambda*H + alpha_p*lambda*I;
  Ml = A'*A+ (1-alpha_p)*lambda*H + alpha_p*lambda*I;
  Mr = A'*b;
  x = Ml\Mr;
end
