%% REFERENCE
% https://en.wikipedia.org/wiki/Newton%27s_method

%%
function [x, obj]  = Newton(A0,A1,x,n,COST,bfig)

if (nargin < 5)
    COST.function	= @(x) (0);
    COST.equation	= [];
end

if (nargin < 4)
    n   = 1e2;
end

A1_     = A1(x);
obj     = zeros(n, 1);

for i = 1:n
    %     x   = x - A0(x)./A1(x);
    x   = x - A0(x)./A1_;
    
    
    obj(i)  = COST.function(x);
    
    if bfig
        figure(1); colormap gray;
        subplot(121); imagesc(x);           title([num2str(i) ' / ' num2str(n)]);
        subplot(122); semilogy(obj, '*-');  title(COST.equation);  xlabel('# of iteration');   ylabel('Objective');
                                            xlim([1, n]);   grid on; grid minor;
        drawnow();
    end
end

x = gather(x);

end