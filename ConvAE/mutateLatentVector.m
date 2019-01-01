function [ res ] = mutate(img, latentMean, sigma, probability)
    % Mutate
    for j = 1:64
        if rand < probability  % changed probability to make it not so frequent
            img(j)=img(j)+random('normal',0.0,sigma*0.1,1,1); %changed sigma to make it sane
        end
    end
	res = img;
end
