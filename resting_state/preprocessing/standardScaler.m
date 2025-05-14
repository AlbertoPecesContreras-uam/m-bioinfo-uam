
function s_norm = standardScale(s)
    mu = mean(s);
    sigma = std(s);

    s_norm = (s-mu)/sigma;
end