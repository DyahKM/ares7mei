% Optimized main.m with reduced computational load
% Purpose: Convert monthly reports to daily estimates, selecting best filter index (ratio) automatically

addpath('HFusion')
addpath('AresFilter')
addpath('data')
clc; clear;

%% 1. Define monthly reports (Jan 2020 - Nov 2024)
reports = [
    1, 31, 43787; % Jan 2020
    32, 60, 40427; % Feb 2020 (29 days, leap year)
    61, 91, 39398; % Mar 2020
    92, 121, 25369; % Apr 2020
    122, 152, 19049; % May 2020
    153, 182, 26593; % Jun 2020
    183, 213, 26878; % Jul 2020
    214, 244, 23179; % Aug 2020
    245, 274, 24317; % Sep 2020
    275, 305, 20644; % Oct 2020
    306, 335, 21124; % Nov 2020
    336, 366, 16747; % Dec 2020
    367, 397, 35053; % Jan 2021
    398, 425, 31532; % Feb 2021 (28 days)
    426, 456, 39848; % Mar 2021
    457, 486, 38948; % Apr 2021
    487, 517, 34749; % May 2021
    518, 547, 37652; % Jun 2021
    548, 578, 25075; % Jul 2021
    579, 609, 30670; % Aug 2021
    610, 639, 39631; % Sep 2021
    640, 670, 42452; % Oct 2021
    671, 700, 45619; % Nov 2021
    701, 731, 44017; % Dec 2021
    732, 762, 66025; % Jan 2022
    763, 790, 48431; % Feb 2022 (28 days)
    791, 821, 56857; % Mar 2022
    822, 851, 52675; % Apr 2022
    852, 882, 54903; % May 2022
    883, 912, 66734; % Jun 2022
    913, 943, 63507; % Jul 2022
    944, 974, 67759; % Aug 2022
    975, 1004, 69874; % Sep 2022
    1005, 1035, 69597; % Oct 2022
    1036, 1065, 68806; % Nov 2022
    1066, 1096, 56626; % Dec 2022
    1097, 1127, 83897; % Jan 2023
    1128, 1155, 73983; % Feb 2023 (28 days)
    1156, 1186, 77314; % Mar 2023
    1187, 1216, 56881; % Apr 2023
    1217, 1247, 80216; % May 2023
    1248, 1277, 69611; % Jun 2023
    1278, 1308, 78302; % Jul 2023
    1309, 1339, 85188; % Aug 2023
    1340, 1369, 78406; % Sep 2023
    1370, 1400, 83093; % Oct 2023
    1401, 1430, 77180; % Nov 2023
    1431, 1461, 63072; % Dec 2023
    1462, 1492, 74898; % Jan 2024
    1493, 1521, 65049; % Feb 2024 (29 days, leap year)
    1522, 1552, 67988; % Mar 2024
    1553, 1582, 66322; % Apr 2024
    1583, 1613, 75951; % May 2024   
    1614, 1643, 65180; % Jun 2024
    1644, 1674, 74885; % Jul 2024
    1675, 1705, 71239; % Aug 2024
    1706, 1735, 70057; % Sep 2024
    1736, 1766, 77470; % Oct 2024
    1767, 1796, 70135; % Nov 2024
];

num_days = reports(end, 2);
events = zeros(num_days, 1);

%% 2. Setup parameters - REDUCED FOR SPEED
% Reduce the number of lambdas (regularization parameters)
lambdas = [0.1, 1, 5, 10]; % Reduced from [0.1, 1, 5, 10, 20]
alpha = 0.5;
gama = 0.5;
iterationTime = 2; % Reduced from 4

% Reduce configuration space
config_rep_dur = [15, 30, 60]; % Reduced from [15, 30, 60]
config_rep_over = [1, 7, 15]; % Reduced from [1, 7, 15]
xdim = length(config_rep_dur);
ydim = length(config_rep_over);

% Reduce number of ratios to evaluate
stops = [1:5]; % Reduced from 1:20 to just 5 values

save('TB_disaggregation_setup.mat', 'reports', 'events', 'lambdas', 'alpha', 'gama', 'iterationTime', ...
    'config_rep_dur', 'config_rep_over', 'xdim', 'ydim');

%% 3. First Phase: Initial reconstruction using H-Fusion
fprintf('Starting H-Fusion reconstruction...\n');
tic;
[A, y] = rep_constraint_equations_full(reports, events);
Out_original = struct();
Out_original(1).muvars = [0, 0];
Out_original(1).A = A;
Out_original(1).y = y;
[recon_events, recon_error, reconstruction_param, M] = sp_reconstruct(A, y, lambdas, events, alpha);
Out_original(1).x_reconstr = recon_events(:, 1, 1);
Out_original(1).x_error = recon_error;
Out_original(1).Matrix = M;
Out_original(1).sp_params = reconstruction_param;
[Out_original(1).error, Out_original(1).minIdx] = min(recon_error(:));
phase1_time = toc;
fprintf('Phase 1 completed in %.2f seconds\n', phase1_time);
save('TB_phase1_original.mat', 'A', 'y', 'Out_original', 'recon_events', 'recon_error', 'reconstruction_param', 'M');

fprintf('Running H-Fusion with different report configurations...\n');
tic;
[Out, Out_LSQ] = hfusion(events, lambdas, alpha, config_rep_dur, config_rep_over);
Out = [Out_original, Out];
hfusion_time = toc;
fprintf('H-Fusion completed in %.2f seconds\n', hfusion_time);
save('TB_phase1_combined.mat', 'Out', 'Out_LSQ');

%% 4. Second Phase: ARES Filter with Automatic Ratio Selection - USING REDUCED SET
fprintf('Starting ARES filter processing...\n');
RMSE = zeros(length(stops), iterationTime);
costs = zeros(length(stops), 1);

fprintf('[ARES] Evaluating selected ratios...\n');
for i = 1:length(stops)
    ratio = stops(i);
    fprintf('  → Evaluating ratio %d of %d (value: %d)\n', i, length(stops), ratio);
    tic;
    [Inc_A, ~, ~] = iteration(Out, events, iterationTime, gama, ratio);
    ArgRMSE = mean(Inc_A.ActError, 1);
    RMSE(i,:) = ArgRMSE;
    costs(i) = ArgRMSE(end) + gama * ratio;
    ratio_time = toc;
    fprintf('  → Ratio %2d: Final RMSE = %.4f | Cost = %.4f | Time: %.2f seconds\n', ratio, ArgRMSE(end), costs(i), ratio_time);
end

[min_cost, opt_idx] = min(costs);
opt_ratio = stops(opt_idx);
fprintf('\n[ARES Result Summary]\n');
fprintf('  → Selected optimal ratio = %d\n', opt_ratio);
fprintf('  → Final RMSE = %.4f\n', RMSE(opt_idx,end));
fprintf('  → Final cost = %.4f (RMSE + γ·Ratio = %.4f + %.4f)\n\n', min_cost, RMSE(opt_idx,end), gama * opt_ratio);

% Run ARES one more time with optimal ratio
fprintf('Running final iteration with optimal ratio = %d\n', opt_ratio);
tic;
[Inc_A, Out_A1, Out_AL] = iteration(Out, events, iterationTime, gama, opt_ratio);
final_time = toc;
fprintf('Final iteration completed in %.2f seconds\n', final_time);
save('TB_ARES_results.mat', 'Inc_A', 'Out_A1', 'Out_AL', 'opt_ratio', 'min_cost');
%%
daily_estimates = Out_AL(1).x_reconstr;
daily_estimates_constrained = enforce_monthly_integer_constraints(daily_estimates, reports);
save('TB_daily_estimates.mat', 'daily_estimates', 'daily_estimates_constrained', 'reports');

% Output results
dates = (1:length(daily_estimates_constrained))';
daily_output = [dates, daily_estimates_constrained];
csvwrite('TB_daily_estimates.csv', daily_output);

% Plot results
figure;
plot(daily_estimates_constrained, 'LineWidth', 1.5);
title('Estimated Daily TB Cases (Integer Constraint Enforced)');
xlabel('Day (1 = Jan 1, 2020)');
ylabel('Number of TB Cases');
grid on;
saveas(gcf, 'TB_daily_estimates_integer.png');

% Verification
monthly_sums = zeros(size(reports,1), 1);
for j = 1:size(reports,1)
    monthly_sums(j) = sum(daily_estimates_constrained(reports(j,1):reports(j,2)));
end

original_sums = zeros(size(reports,1), 1);
for j = 1:size(reports,1)
    original_sums(j) = sum(daily_estimates(reports(j,1):reports(j,2)));
end

max_error_before = max(abs(original_sums - reports(:,3))) / max(reports(:,3)) * 100;
max_error_after = max(abs(monthly_sums - reports(:,3))) / max(reports(:,3)) * 100;

fprintf('Maximum constraint error before adjustment: %.6f%%\n', max_error_before);
fprintf('Maximum constraint error after adjustment: %.6f%%\n', max_error_after);

if all(daily_estimates_constrained == round(daily_estimates_constrained))
    fprintf('All daily estimates are integers as required for TB case counts.\n');
else
    fprintf('WARNING: Not all daily estimates are integers!\n');
end

% Save verification data
verification_data = struct();
verification_data.monthly_sums = monthly_sums;
verification_data.original_sums = original_sums;
verification_data.max_error_before = max_error_before;
verification_data.max_error_after = max_error_after;
verification_data.all_integers = all(daily_estimates_constrained == round(daily_estimates_constrained));
save('TB_verification.mat', 'verification_data');

% Plot monthly comparison
figure;
bar([reports(:,3), monthly_sums, original_sums]);
title('Monthly Reports vs. Sum of Daily Estimates');
xlabel('Month Index');
ylabel('Number of TB Cases');
legend('Original Monthly Reports', 'Constrained Daily Sums', 'Unconstrained Daily Sums');
grid on;
saveas(gcf, 'TB_monthly_verification.png');

% Save monthly comparison
monthly_comparison = [reports(:,3), monthly_sums, original_sums, reports(:,3) - monthly_sums, reports(:,3) - original_sums];
csvwrite('TB_monthly_comparison.csv', monthly_comparison);

% Plot detailed view of first months
figure;
months_to_show = min(6, size(reports,1));
end_day = reports(months_to_show, 2);
plot(1:end_day, daily_estimates_constrained(1:end_day), 'LineWidth', 1.5);
title('Daily TB Cases - First 6 Months');
xlabel('Day');
ylabel('Number of TB Cases');
grid on;
hold on;
for j = 1:months_to_show
    if j > 1
        xline(reports(j,1)-0.5, '--', 'LineWidth', 1);
    end
    month_midpoint = (reports(j,1) + reports(j,2))/2;
    text(month_midpoint, max(daily_estimates_constrained(1:end_day))*0.9, ...
        sprintf('Month %d', j), 'HorizontalAlignment', 'center');
end
hold off;
saveas(gcf, 'TB_daily_detail_first_months.png');

% Save detail plot data
detail_plot_data = struct();
detail_plot_data.months_to_show = months_to_show;
detail_plot_data.end_day = end_day;
detail_plot_data.daily_data = daily_estimates_constrained(1:end_day);
save('TB_detail_plot_data.mat', 'detail_plot_data');

%%
% Output results
dates = (1:length(daily_estimates_constrained))';
daily_output = [dates, daily_estimates_constrained];
csvwrite('TB_daily_estimates.csv', daily_output);

% Verification
monthly_sums = zeros(size(reports,1), 1);
for j = 1:size(reports,1)
    monthly_sums(j) = sum(daily_estimates_constrained(reports(j,1):reports(j,2)));
end

original_sums = zeros(size(reports,1), 1);
for j = 1:size(reports,1)
    original_sums(j) = sum(daily_estimates(reports(j,1):reports(j,2)));
end

max_error_before = max(abs(original_sums - reports(:,3))) / max(reports(:,3)) * 100;
max_error_after = max(abs(monthly_sums - reports(:,3))) / max(reports(:,3)) * 100;

fprintf('Maximum constraint error before adjustment: %.6f%%\n', max_error_before);
fprintf('Maximum constraint error after adjustment: %.6f%%\n', max_error_after);

if all(daily_estimates_constrained == round(daily_estimates_constrained))
    fprintf('All daily estimates are integers as required for TB case counts.\n');
else
    fprintf('WARNING: Not all daily estimates are integers!\n');
end

% Save verification data
verification_data = struct();
verification_data.monthly_sums = monthly_sums;
verification_data.original_sums = original_sums;
verification_data.max_error_before = max_error_before;
verification_data.max_error_after = max_error_after;
verification_data.all_integers = all(daily_estimates_constrained == round(daily_estimates_constrained));
save('TB_verification.mat', 'verification_data');

% Plot monthly comparison
figure;
bar([reports(:,3), monthly_sums, original_sums]);
title('Monthly Reports vs. Sum of Daily Estimates');
xlabel('Month Index');
ylabel('Number of TB Cases');
legend('Original Monthly Reports', 'Constrained Daily Sums', 'Unconstrained Daily Sums');
grid on;
saveas(gcf, 'TB_monthly_verification.png');

% Save monthly comparison
monthly_comparison = [reports(:,3), monthly_sums, original_sums, reports(:,3) - monthly_sums, reports(:,3) - original_sums];
csvwrite('TB_monthly_comparison.csv', monthly_comparison);

% Plot detailed view of first months
figure;
months_to_show = min(6, size(reports,1));
end_day = reports(months_to_show, 2);
plot(1:end_day, daily_estimates_constrained(1:end_day), 'LineWidth', 1.5);
title('Daily TB Cases - First 6 Months');
xlabel('Day');
ylabel('Number of TB Cases');
grid on;
hold on;
for j = 1:months_to_show
    if j > 1
        xline(reports(j,1)-0.5, '--', 'LineWidth', 1);
    end
    month_midpoint = (reports(j,1) + reports(j,2))/2;
    text(month_midpoint, max(daily_estimates_constrained(1:end_day))*0.9, ...
        sprintf('Month %d', j), 'HorizontalAlignment', 'center');
end
hold off;
saveas(gcf, 'TB_daily_detail_first_months.png');

% Save detail plot data
detail_plot_data = struct();
detail_plot_data.months_to_show = months_to_show;
detail_plot_data.end_day = end_day;
detail_plot_data.daily_data = daily_estimates_constrained(1:end_day);
save('TB_detail_plot_data.mat', 'detail_plot_data');

% Fixed function implementation
function constrained_estimates = enforce_monthly_integer_constraints(daily_estimates, reports)
    % This function adjusts daily estimates so that:
    % 1. They sum exactly to the monthly reports
    % 2. All daily values are integers (suitable for case counts)
    
    constrained_estimates = daily_estimates;
    
    % Process each month
    for j = 1:size(reports, 1)
        start_day = reports(j, 1);
        end_day = reports(j, 2);
        target_sum = reports(j, 3);
        
        % Get the daily estimates for this month
        month_days = constrained_estimates(start_day:end_day);
        
        % First, round to integers while preserving the overall structure
        month_days_int = round(month_days);
        
        % Calculate the current sum
        current_sum = sum(month_days_int);
        
        % Adjust to match target (distribute the difference)
        difference = target_sum - current_sum;
        
        if difference ~= 0
            % Sort days by fractional part to determine which to adjust
            [~, idx] = sort(abs(month_days - month_days_int), 'descend');
            
            % Adjust days with highest fractional part
            for k = 1:min(abs(difference), length(idx))
                if difference > 0
                    month_days_int(idx(k)) = month_days_int(idx(k)) + 1;
                else
                    month_days_int(idx(k)) = month_days_int(idx(k)) - 1;
                end
            end
        end
        
        % Ensure no negative values (TB cases can't be negative)
        month_days_int(month_days_int < 0) = 0;
        
        % Re-balance if necessary after ensuring non-negativity
        while sum(month_days_int) ~= target_sum
            if sum(month_days_int) < target_sum
                % Need to add more
                [~, idx] = sort(month_days, 'descend');
                remaining_diff = target_sum - sum(month_days_int);
                for k = 1:min(remaining_diff, length(idx))
                    month_days_int(idx(k)) = month_days_int(idx(k)) + 1;
                end
            else
                % Need to subtract more
                [~, idx] = sort(month_days_int, 'descend');
                remaining_diff = sum(month_days_int) - target_sum;
                for k = 1:min(remaining_diff, length(idx))
                    if month_days_int(idx(k)) > 0
                        month_days_int(idx(k)) = month_days_int(idx(k)) - 1;
                    end
                end
            end
        end
        
        % Update the constrained estimates for this month
        constrained_estimates(start_day:end_day) = month_days_int;
    end
end
