function [ Out, Out_LSQ ] = hfusion( events, lambdas, alpha, config_rep_dur, config_rep_over)
    % Count report configuration.
    i = 1;

    config_rds = length(config_rep_dur);
    config_frq = length(config_rep_over);


    for j_rd = 1:config_rds
        mu_rd = config_rep_dur(j_rd);
      for j_or = 1:config_frq
         rate = config_rep_over(j_or);

         fprintf('looping %d %d\n',j_rd,j_or);

         % generate reports
         reports = createData(events, mu_rd, rate);

         % generate characteristic linear system
         % corresponding to the reports
         [A,y]=rep_constraint_equations_full(reports,events);

         % Out will store the experimental results
         Out(i).muvars = [mu_rd,rate];
         Out(i).A = A;
         Out(i).y = y;

        % Reconstruct sequence by H-FUSION
        [reconstracted_events, reconstruction_error, reconstruction_param, M] = sp_reconstruct(A, y, lambdas, events, alpha);
        Out(i).x_reconstr = reconstracted_events;
        Out(i).x_error =  reconstruction_error;
        Out(i).Matrix = M;
        Out(i).sp_params = reconstruction_param;
        [Out(i).error,Out(i).minIdx] = min(reconstruction_error);


        [reconstracted_events, reconstruction_error] = lsq_reconstruct(A, y, events);

        Out_LSQ(i).x_reconstr = reconstracted_events;
        Out_LSQ(i).error =  reconstruction_error;

        Out_LSQ(i).muvars = [mu_rd,rate];
        Out_LSQ(i).A = A;
        Out_LSQ(i).y = y;

        i = i+1;
       end
    end
end
