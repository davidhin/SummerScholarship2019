function showPop(pop, row, col)
    % Assumes a population of 20 pictures
    % pop is a cell-array of pictures. 
    % assumes that a figure is defined and hold is on.
    global metricVec;
    global fitnessVec;
    [temp1, temp2] = sort(metricVec);
    pop = pop(temp2);
    fitnessVec = fitnessVec(temp2);

    global mu;
    global A;
    global B;
    global current_images;
    simAVec = [];
    simBVec = [];
    for indx=1:mu
        simAVec = [simAVec, imDistanceR(A, current_images{indx}, 2)];
        simBVec = [simBVec, imDistanceR(B, current_images{indx}, 2)];
    end
    simAVec = simAVec(temp2);
    simBVec = simBVec(temp2);

    count=1;
    for rows = 1:row
        for cols = 1:col
            if count<=length(pop)
                figure(1)
                subplot(row,col,count)
                imshow(pop{count})
                %title([temp1(count) simAVec(count) simBVec(count)])
                title([fitnessVec(count) temp1(count) simAVec(count) simBVec(count)])
            end
            count=count+1;
        end
        drawnow 
    end
end


