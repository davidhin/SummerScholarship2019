function showPop(pop, row, col)
    % Assumes a population of 20 pictures
    % pop is a cell-array of pictures. 
    % assumes that a figure is defined and hold is on.
    global metricVec;
    global fitnessVec;
    [temp1, temp2] = sort(metricVec);
    pop = pop(temp2);
    count=1;
    for rows = 1:row
        for cols = 1:col
            if count<=length(pop)
                subplot(row,col,count)
                imshow(pop{count})
                title([fitnessVec(count) temp1(count)])
            end
            count=count+1;
        end
    end
end


