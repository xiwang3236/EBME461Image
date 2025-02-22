function error=SSE(fixed, moving, tx, ty, param)
    if isfield(param, 'scaling')
        scaleFactor= param.scaling;
    else
        scaleFactor=1;
    end
    Tx= tx*scaleFactor;
    Ty= ty*scaleFactor;
    translated = imtranslate(moving,[Tx,Ty], "OutputView","same");
    
    differenceImage = abs(fixed-translated);
    error = sum(differenceImage(4:end,3:end).^2,"all");
end