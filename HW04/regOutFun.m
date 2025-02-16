%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regOutFun.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stop = regOutFun(x, optimVals, state, fixedImg, movingImg, param)
% p = current parameter vector [Tx, Ty]
% "state" is one of: 'init', 'iter', 'done'
% "optimVals" has info like iteration number, etc.
% We want to capture the difference image at each iteration into a movie.

    persistent frames
    stop = false;  % We do not want to stop the optimizer prematurely

    switch state
        case 'init'
            % Called at the start of fminsearch
            disp('Starting registration...');
            frames = [];  % Initialize an empty array of frames

            % >>> Capture the difference at the *initial* guess <<<
            [diffImg] = getDifferenceImage(x, fixedImg, movingImg, param);
           figure(99);
            % fig= figure; 
           %disp(fig);
            clf;
            imshow(diffImg,[]); 
            %title('Difference Image (Start)');
            drawnow;
            F = getframe(gcf);
           % disp(F);
            frames = [frames F];

        case 'iter'
            disp('Iteration going on');
            % Called at each iteration
            [diffImg] = getDifferenceImage(x, fixedImg, movingImg, param);
            figure(99); clf;
            imshow(diffImg,[]); 
            %title(sprintf('Iteration %d: Tx=%.3f, Ty=%.3f', ...
%                        optimVals.iteration, p(1), p(2)));
            drawnow;
            F = getframe(gcf);
            frames = [frames F];

        case 'done'
            % Called at the end of fminsearch
            disp('Registration complete!');

            % >>> Capture one last frame: final difference <<<
            [diffImg] = getDifferenceImage(x, fixedImg, movingImg, param);
            figure(99); clf;
            imshow(diffImg,[]); 
            %title(sprintf('Final Difference: Tx=%.3f, Ty=%.3f', p(1), p(2)));
            drawnow;
            F = getframe(gcf);
            frames = [frames F];

            % --- Visualize the movie in MATLAB ---
            figure(101); clf;
            movie(frames, F,1, 2);  % play once at 2 frames/sec

            % --- Export to AVI (older approach) ---
            v = VideoWriter('registration_movie.avi');
            open(v);
            writeVideo(v, frames);
            close(v);            
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local helper to get the difference image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function diffImg = getDifferenceImage(p, fixedImg, movingImg, param)
    % If scaling is needed:
    % pScaled = p * 100;
    % Tx = pScaled(1); Ty = pScaled(2);
    Tx = p(1);
    Ty = p(2);

    moved = imtranslate(movingImg, [Tx, Ty],'OutputView','same');
    diffImg = imabsdiff( fixedImg, moved);
end
