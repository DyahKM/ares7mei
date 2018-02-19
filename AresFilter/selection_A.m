function [AUC, Out_A] = selection_A(Out, events, Out_Cmp)
    stops = [0:0.1:0.8,0.9:0.01:1];
    len = length(stops);
    AUC_AG = zeros(len,1);
    AUC_AL = zeros(len,1);
    final_stop = 0;

    Out_A_save = struct([]);
    for i = 1:len
        Out_A_save(i).Out_A = Out;
    end
    
    AvgRMSE = zeros(1,len);
    
    parfor i = 1:len
        Out_A_save(i).Out_A = annihilating(Out, events, stops(i), 10);
        [AUC_AG(i),AUC_AL(i)] = TPH_HFUSE_plot(Out_Cmp, Out_A_save(i).Out_A);
        
        errors = zeros(1,length(Out));
        for k = 1:length(Out)
            errors(k) = Out_A_save(i).Out_A(k).error;
        end
        
        AvgRMSE(i) = mean(errors);
    end
    
    Out_A = Out_A_save(len).Out_A;
    for i = 1:len
      if AUC_AG(i)-AUC_AL(i) == max(AUC_AG-AUC_AL)
%       if AvgRMSE(i) == min(AvgRMSE)
        final_stop = stops(i);
        Out_A = Out_A_save(i).Out_A;
        AUC.idx = i;
        AUC.Out_A = Out_A;
        break;
      end
    end
    
    AUC.AvgRMSE = AvgRMSE;
    AUC.AUC_AL = AUC_AL;
    AUC.AUC_AG = AUC_AG;
    AUC.final_stop = final_stop; 
end