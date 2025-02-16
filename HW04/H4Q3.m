%initialize first guess for translation, [tx, ty]
t_i= [0,0];
frames = struct('cdata', {}, 'colormap', {});
%initialize array
param=struct();
param.scaling=1;

%calling the cost function SSE
costF= @(t) registrationcost(t, origin, translated);
%costF= @(t) SSE(origin, translated, t(1),t(2), param );
% cost_test= SSE(origin, translated, -1.8, -2.1);
% display(cost_test)

%run fminsearch on optimal translation parameters
% and custom options to show iterations
options = optimset( 'TolFun',1e-3, 'TolX',1e-3);
%     'OutputFcn', @(x, optimValues, state) ...
%     updateFrames(x, optimValues, state, frames) 
                    %'OutputFcn',@regOutFun);
[t_optimal, fval]= fminsearch(costF, t_i, options);

fprintf(['The optimal translation parameters t_x is %.4f, t_y is %4f, ' ...
    'and minimal cost function is %d\n'], ...
    t_optimal(1), t_optimal(2), fval);

%subtracted image without registration
diffImg_noReg= abs(origin-translated);

figure;
subplot(2,2,1), imshow(origin,[]), title('Contrast 1');
subplot(2,2,2), imshow(translated,[]), title('Contrast 2');
subplot(2,2,3), imshow(uint8(diffImg_noReg)), title('Subtracted Image without Registration');

figure;
axis off;

movie(frames);

function cost = registrationcost(params, fixedImage, movingImage)
    persistent frames;
    if isempty(frames)
        frames = struct('cdata', {}, 'colormap', {});
    end

    % Extract translation parameters
    tx = params(1);
    ty = params(2);

    % Translate the moving image
    translatedImage = imtranslate(movingImage, [tx, ty]);

    % Compute the difference image
    differenceImage = abs(fixedImage - translatedImage);

    % Compute the sum of squared errors
    cost = sum(differenceImage(4:end,3:end).^2,"all");

    % Capture the difference image as a frame
    figure(100);
    imshow(uint8(differenceImage));
    drawnow;
    frame = getframe(gcf);
    
    % Store the frame
    frames(end+1) = frame;

    % Assign the updated frames to the base workspace
    assignin('base', 'frames', frames);
end