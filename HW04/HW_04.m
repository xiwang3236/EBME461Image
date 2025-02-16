% 1. Initial Setup
clc; clear;
live = imread('live_new.tif');    % Reference image (with contrast)
mask = imread('mask_new.tif');    % Floating image (no contrast)
[m, n] = size(live);

% Initial difference before registration
diff_before = live - mask;

% 2. Parameter Settings for Testing
param = struct();
param.scaling = 1;

% Different parameters to test
tolFun_values = [1e-1, 1e-3, 1e-5];  % Function tolerance
tolX_values = [1e-1, 1e-3, 1e-5];    % Parameter tolerance
scaling_values = [1, 10, 100];        % Scaling factors
max_iter_values = [100, 200, 500];    % Maximum iterations



% 3. Testing Different Parameter Combinations
for tf = tolFun_values
    for tx = tolX_values
        for sc = scaling_values
            for mi = max_iter_values
                % Update parameters
                param.scaling = sc;
                
                % Setup cost function
                costF = @(t) SSE(live, mask, t(1), t(2), param);
                
                % Create the output function with additional parameters
                outfun = @(x, optimVals, state) regOutFun(x, optimVals, state, live, mask, param);
                
                % Set up optimization options with the modified output function
                options = optimset('Display', 'iter', ...
                                 'TolFun', tf,      % Use tf from loop
                                 'TolX', tx,        % Use tx from loop
                                 'MaxIter', mi,     % Use mi from loop
                                 'OutputFcn', outfun);
                
                % Start timing
                tic;
                
                % Run optimization with costF (not outfun)
                [t_optimal, fval, exitflag, output] = fminsearch(costF, [0,0], options);
                
                % End timing
                computation_time = toc;
                
                % Store results
                results(result_idx).TolFun = tf;
                results(result_idx).TolX = tx;
                results(result_idx).Scaling = sc;
                results(result_idx).MaxIter = mi;
                results(result_idx).FinalCost = fval;
                results(result_idx).Iterations = output.iterations;
                results(result_idx).Tx = t_optimal(1);
                results(result_idx).Ty = t_optimal(2);
                results(result_idx).Time = computation_time;
                results(result_idx).Success = (exitflag == 1);
                
                result_idx = result_idx + 1;
            end
        end
    end
end

% 4. Convert results to table for better visualization
results_table = struct2table(results);

% 5. Find best result (lowest cost)
[~, best_idx] = min([results.FinalCost]);
best_params = results(best_idx);

% 6. Apply best parameters for final registration
param.scaling = best_params.Scaling;
options = optimset('Display', 'iter', ...
                  'TolFun', best_params.TolFun, ...
                  'TolX', best_params.TolX, ...
                  'MaxIter', best_params.MaxIter, ...
                  'OutputFcn', @regOutFun);

[final_t, final_cost] = fminsearch(@(t) SSE(live, mask, t(1), t(2), param), [0,0], options);

% 7. Create final registered image using best parameters
[X, Y] = meshgrid(1:n, 1:m);
X_regis = X + final_t(1);
Y_regis = Y + final_t(2);

% Initialize the output image
regis_image = NaN(m, n);

% Find integer coordinates
x1 = floor(X_regis);
x2 = ceil(X_regis);
y1 = floor(Y_regis);
y2 = ceil(Y_regis);

% Valid mask
validMask = x1 >= 1 & x2 <= n & y1 >= 1 & y2 <= m;

% Interpolation weights
wx = X_regis - x1;
wy = Y_regis - y1;

% Get corner values
Q11 = NaN(m, n); Q21 = NaN(m, n);
Q12 = NaN(m, n); Q22 = NaN(m, n);

Q11(validMask) = double(mask(sub2ind([m, n], y1(validMask), x1(validMask))));
Q21(validMask) = double(mask(sub2ind([m, n], y1(validMask), x2(validMask))));
Q12(validMask) = double(mask(sub2ind([m, n], y2(validMask), x1(validMask))));
Q22(validMask) = double(mask(sub2ind([m, n], y2(validMask), x2(validMask))));

% Bilinear interpolation
regis_image(validMask) = (1 - wx(validMask)) .* (1 - wy(validMask)) .* Q11(validMask) + ...
                         wx(validMask) .* (1 - wy(validMask)) .* Q21(validMask) + ...
                         (1 - wx(validMask)) .* wy(validMask) .* Q12(validMask) + ...
                         wx(validMask) .* wy(validMask) .* Q22(validMask);

% Handle NaN values
regis_image(isnan(regis_image)) = 0;

% 8. Final difference image
diff_after = live - regis_image;

% 9. Display results
fprintf('\nOptimization Results:\n');
fprintf('Best parameters found:\n');
fprintf('TolFun: %e\n', best_params.TolFun);
fprintf('TolX: %e\n', best_params.TolX);
fprintf('Scaling: %d\n', best_params.Scaling);
fprintf('MaxIter: %d\n', best_params.MaxIter);
fprintf('Optimal translation: [%.4f, %.4f]\n', final_t(1), final_t(2));
fprintf('Final cost: %.4f\n', final_cost);

% 10. Display images
figure;
tiledlayout(2,2)

nexttile
imagesc(live)
colormap('gray')
title('Live Image (with contrast)')

nexttile
imagesc(mask)
title('Mask Image (no contrast)')
colormap('gray')

nexttile
imagesc(diff_before)
title('Subtraction Before Registration')
colormap('gray')

nexttile
imagesc(diff_after)
title('Subtraction After Registration')
colormap('gray')

% 11. Save results
save('registration_results.mat', 'results', 'best_params', 'final_t', 'final_cost');
writetable(results_table, 'optimization_results.csv');