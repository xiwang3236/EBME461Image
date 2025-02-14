function storeFrame(translation, fixedIm, movingIm, frames)
    transIm= imtranslate(movingIm, translation);
    diff= imabsdiff(fixedIm, transIm);
    % use a figure handle
    figure(1);
    imshow(diff,[]);
    % get the most updated frame using figure handle
    frame_current= getframe(gcf);
    %cancatenate array for frameS
    frames=[frames, frame_current];

end