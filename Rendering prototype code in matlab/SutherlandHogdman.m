function [output] = SutherlandHogdman(c1, c2, c3, c4)

[output, p1, p2, p3, p4] = sortCCW(c1, c2, c3, c4);

if (output == 0)
    output = 0;
    return;
end


p1 = [p1(1); p1(2)];
p2 = [p2(1); p2(2)];
p3 = [p3(1); p3(2)];
p4 = [p4(1); p4(2)];

% List outputList = subjectPolygon;   
%   for (Edge clipEdge in clipPolygon) do
%      List inputList = outputList;
%      outputList.clear();
%      Point S = inputList.last;
%      for (Point E in inputList) do
%         if (E inside clipEdge) then
%            if (S not inside clipEdge) then
%               outputList.add(ComputeIntersection(S,E,clipEdge));
%            end if
%            outputList.add(E);
%         else if (S inside clipEdge) then
%            outputList.add(ComputeIntersection(S,E,clipEdge));
%         end if
%         S = E;
%      done
%   done

inputList = [p1 p2 p3 p4];
inputListSize = 4;
outputList = [p1 p2 p3 p4];
outputListSize = 4;
%edge along beta (or alpha = 0)
inputList = outputList;
inputListSize = outputListSize;
outputList = [0;0];
outputListSize = 0;
pS = inputList(:,inputListSize);
for i = 1:inputListSize
    pE = inputList(:,i);
    %if (E inside clipEdge)
    if (pE(1) >= 0)
        %if (S not inside clipEdge)
        if (pS(1) < 0)
            %add intersection to the output list
            pT = [0;0];
            pT(1) = 0;
            pT(2) = pS(2) - pS(1)*(pE(2)-pS(2))/(pE(1) - pS(1));
            outputList(:,outputListSize+1) = pT;
            outputListSize = outputListSize+1;
        end
        %add pE to output list
        outputList(:,outputListSize+1) = pE;
        outputListSize = outputListSize+1;
        %else if (S inside clipEdge)
    elseif (pS(1) >= 0)
        %add intersection to the output list
        pT = [0;0];
        pT(1) = 0;
        pT(2) = pS(2) - pS(1)*(pE(2)-pS(2))/(pE(1) - pS(1));
        outputList(:,outputListSize+1) = pT;
        outputListSize = outputListSize+1;
    end
    pS = pE;
end
if (outputListSize == 0)
    output = 0;
    return;
end

%edge along alpha (or beta = 0)
inputList = outputList;
inputListSize = outputListSize;
outputList = [0;0];
outputListSize = 0;
pS = inputList(:,inputListSize);
for (i = 1:inputListSize)
    pE = inputList(:,i);
    %if (E inside clipEdge)
    if (pE(2) >= 0)
        %if (S not inside clipEdge)
        if (pS(2) < 0)
            %add intersection to the output list
            pT = [0;0];
            pT(1) = pS(1) - pS(2)*(pE(1)-pS(1))/(pE(2) - pS(2));
            pT(2) = 0;
            outputList(:,outputListSize+1) = pT;
            outputListSize = outputListSize+1;
        end
        %add pE to output list
        outputList(:,outputListSize+1) = pE;
        outputListSize = outputListSize+1;
    %else if (S inside clipEdge)
    elseif (pS(2) >= 0)
        %add intersection to the output list
        pT = [0;0];
        pT(1) = pS(1) - pS(2)*(pE(1)-pS(1))/(pE(2) - pS(2));
        pT(2) = 0;
        outputList(:,outputListSize+1) = pT;
        outputListSize = outputListSize+1;
    end
    pS = pE;
end
if (outputListSize == 0)
    output = 0;
    return;
end

%oblique edge
inputList = outputList;
inputListSize = outputListSize;
outputList = [0;0];
outputListSize = 0;
pS = inputList(:,inputListSize);
for (i = 1:inputListSize)
    pE = inputList(:,i);
    %if (E inside clipEdge)
    if ((pE(1)+pE(2)) <= 1)
        %if (S not inside clipEdge)
        if ((pS(1)+pS(2)) > 1)
            %add intersection to the output list
            pT = [0;0];
            pT(1) = (pE(1)*(pS(2)-pE(2)) + (1-pE(2))*(pS(1) - pE(1)))/((pS(1) - pE(1))+(pS(2) - pE(2)));
            pT(2) = 1-pT(1);
            outputList(:,outputListSize+1) = pT;
            outputListSize = outputListSize+1;
        end
        %add pE to output list
        outputList(:,outputListSize+1) = pE;
        outputListSize = outputListSize+1;
    %else if (S inside clipEdge)
    elseif ((pS(1)+pS(2)) <= 1)
        %add intersection to the output list
        pT = [0;0];
        pT(1) = (pE(1)*(pS(2)-pE(2)) + (1-pE(2))*(pS(1) - pE(1)))/((pS(1) - pE(1))+(pS(2) - pE(2)));
        pT(2) = 1-pT(1);
        outputList(:,outputListSize+1) = pT;
        outputListSize = outputListSize+1;
    end
    pS = pE;
end
if (outputListSize == 0)
    output = 0;
    return;
end

figure(2)
xdraw = [0 0 1 0];
ydraw = [0 1 0 0];
plot(xdraw, ydraw, 'k-','MarkerSize',9);
hold on

pxdraw = [outputList(1,:) outputList(1,1)];
pydraw = [outputList(2,:) outputList(2,1)];
plot(pxdraw, pydraw, 'r-','MarkerSize',9);
hold on


plot(p1(1), p1(2), 'ob','MarkerSize',9);
plot(p2(1), p2(2), 'ob','MarkerSize',9);
plot(p3(1), p3(2), 'ob','MarkerSize',9);
plot(p4(1), p4(2), 'ob','MarkerSize',9);

if (outputListSize == 0)
    output = 0;
    return;
end

output = outputList(1,outputListSize)*outputList(2,1) - outputList(1,1)*outputList(2,outputListSize);

for i = 1:(outputListSize-1)
    output = output + outputList(1,i)*outputList(2,i+1) - outputList(1,i+1)*outputList(2,i);
end
%polygon area formula
output = output/2;
%divide by the area of triangle
output = output/(1/2);
end

