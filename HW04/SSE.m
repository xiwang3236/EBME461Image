function error=SSE(fixed, moving, tx, ty)
    translated = imtranslate(moving,[tx,ty], "OutputView","same");
    error= sum((fixed(:)- translated(:)).^2);
end