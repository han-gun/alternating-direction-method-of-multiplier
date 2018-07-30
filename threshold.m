
function y = threshold(x, lambda, type)
if nargin < 3
    type = 'soft';
end

switch type
    case 'soft'
        y   = (max(abs(x) - lambda, 0)).*sign(x);
    case 'hard'
        y   = (abs(x) > lambda).*x;
end
end