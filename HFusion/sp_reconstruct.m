function [recon_events, recon_error,recon_params, M]=sp_reconstruct(A, y, lambdas, events, alphas)
  sphat = zeros(length(events),length(lambdas),length(alphas));
  T = 51;
  recon_params = zeros(2,length(lambdas),length(alphas));
  for i=1:length(lambdas)
      for j=1:length(alphas)
          lambda_s = lambdas(i);
          alpha_p = alphas(j);
          [sphat(:,i,j), M] = opt_sp(A,y,lambda_s, alpha_p, T);
          recon_params(:,i,j) = [lambda_s, alpha_p];
      end
  end
  recon_events = sphat;
  recon_error = sqrt(mean((recon_events - repmat(events,1,length(lambdas),length(alphas))).^2,1));
end
