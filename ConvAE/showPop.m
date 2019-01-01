function showPop(pop, row, col)
    % Assumes a population of 20 pictures
    % pop is a cell-array of pictures. 
    % assumes that a figure is defined and hold is on.
    global metricVec;
    global fitnessVec;
    count=1;
    for rows = 1:row
        for cols = 1:col
            if count<=length(pop)
                subplot(row,col,count)
                imshow(pop{count})
                % title([fitnessVec(count) metricVec(count)])
            end
            count=count+1;
        end
    end
end


