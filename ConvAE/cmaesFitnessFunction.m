function [ fitness ] = fitPlusSimABtimesMeanHue(images)
    % Takes in a matrix of size samples x mu
    global mu;
    global samples;
    global t;
    images = reshape(images, [samples,mu]);
    
    % Functions 
    global metric;
    global contribs;
    global fitnessFunction;
    % fitnessFunction = @alwaysOne;
    
    % Variables
    images_cell = num2cell(images,1);
    images_cell_decoded = cell(1, mu); 
    fitnessVec = [];
   
    % Transform data types 
    for indx=1:mu
        images_cell_decoded{indx} = decode(t, images_cell{indx});
        fitnessVec = [fitnessVec, fitnessFunction(images_cell_decoded{indx})];
    end

    % Calculate contributions and metric
    [contributions, metricVec] = contribs(images_cell_decoded,metric);
    sorted_contribs = sort(contributions);
    norm_contribs = norm(contributions, 0.5);
    norm_fitness = prod(fitnessVec, 2)^(1/5);

    global distInterval;
    global intervalCounter;
    global metric_plot;
    global contrib_plot;
    global fitness_plot;
    if mod(intervalCounter, distInterval) == 0
        fitnessVec
        contributions
        metricVec
        norm_contribs
        norm_fitness
        metric_plot = [metric_plot; metricVec];               
        % contrib_plot = [contrib_plot; contributions]; 
        fitness_plot = [fitness_plot; fitnessVec];                
    end
    intervalCounter = intervalCounter + 1;

    % fitness = 1.0 - (norm(contributions, 0.5)*prod(fitnessVec)^(1/5)); %  High similarity low diversity
    fitness = 1.0 - norm_contribs*norm_fitness; %  High similarity low diversity
    return; % may  be 1 - um of contribs 
end

