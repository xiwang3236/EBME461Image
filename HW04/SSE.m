function error=SSE(fixed, moving, tx, ty, param)
    if isfield(param, 'scaling')
        scaleFactor= param.scaling;
    else
        scaleFactor=1;
    end
    Tx= tx*scaleFactor;
    Ty= ty*scaleFactor;
    translated = imtranslate(moving,[Tx,Ty], "OutputView","same");
    error= sum((fixed(:)- translated(:)).^2);
end