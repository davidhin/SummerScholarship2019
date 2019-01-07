function [ fitness ] = fitPlusSimABtimesMeanHue(images)
    % Takes in a matrix of size samples x mu
    global mu;
    global samples;
    global t;
    global current_images; % Used to hold images of current iteration
    global metric;
    global contribs;
    global fitnessFunction;
    global fitnessVec;
    global metricVec;
    global current_encodings;
    current_encodings = images;
    images = reshape(images, [samples,mu]);
    
    % Variables
    images_cell = num2cell(images,1);
    images_cell_decoded = cell(1, mu); 
    fitnessVec = [];
   
    % Transform data types 
    for indx=1:mu
        images_cell_decoded{indx} = decode(t, images_cell{indx});
        fitnessVec = [fitnessVec, fitnessFunction(images_cell_decoded{indx})];
    end
    current_images = images_cell_decoded;

    % Calculate contributions and metric
    [contributions, metricVec] = contribs(images_cell_decoded,metric);
    sorted_contribs = sort(contributions);
    norm_contribs = norm(contributions, 0.5);
    norm_fitness = prod(fitnessVec);

    % Plotting
    global distInterval;
    global intervalCounter;
    global metric_plot;
    global contrib_plot;
    global fitness_plot;
    if mod(intervalCounter, distInterval) == 0
        %fitnessVec
        %contributions
        %metricVec
        norm_contribs
        norm_fitness
        metric_plot = [metric_plot; metricVec];               
        %contrib_plot = [contrib_plot; contributions]; 
        fitness_plot = [fitness_plot; norm_fitness];  
        showPop(current_images, 1, mu)
        figure(2);
        plot(metric_plot);
        figure(3);
        plot(fitness_plot);
        drawnow
    end
    intervalCounter = intervalCounter + 1;

    fitness = 1.0 - norm_contribs*norm_fitness; %  High similarity low diversity
    return; 
end

%% Find Initial Vecotr
%     simAVec = []; 
%     simBVec = [];
%     global A;
%     global B; 
%     for indx=1:mu
%     simAVec = [simAVec, imDistanceR(A, current_images{indx}, 2)];
%     simBVec = [simBVec, imDistanceR(B, current_images{indx}, 2)];
%     end
%     fitness = 1.0 - (abs(0.5 - metricVec) + abs(simAVec - simBVec));
