function [] = rasterizev2(vtx1, vtx2, vtx3, pdir, thetaR, R0)

%sort the vertex ccw wrt the direction

%pick a point
%first guess is the center of three vertices
vtxCenter = (vtx1+vtx2+vtx3)/3;
vtxCenter(3) = sqrt(R0*R0-vtxCenter(1)*vtxCenter(1)-vtxCenter(2)*vtxCenter(2));
vtxs(:,1) = vtxCenter;
vtxs(:,2) = vtx1;
vtxs(:,3) = vtx2;
vtxs(:,4) = vtx3;
outputSearch = 0;

for i = 1:4
    % pGuess = [floor(vtx1(1));floor(vtx1(2))];
    [outputGuess, pGuess] = guestPoint(vtxs(:,i), thetaR, R0);
    if (outputGuess == false)
        continue;
    end
    
    pCur = pGuess;

    %extend until found a point inside
    searchRadius = 0;
    while (outputSearch == 0 && searchRadius < 2)
        for j = -searchRadius:searchRadius
            for k = -searchRadius:searchRadius
                pSearch = [pGuess(1)+j;pGuess(2)+k];
                outputSearch = insideTriangle(pSearch, vtx1, vtx2, vtx3, pdir, thetaR, R0);
                if (outputSearch ~= 0)
                    pCur = pSearch;
                    break;
                end
            end
            if (outputSearch ~= 0)
                break;
            end
        end
        searchRadius = searchRadius + 1 ; 
    end
    if (outputSearch ~= 0)
        break;
    end
end

%if found no point, quit
if (outputSearch == 0)
    return;
end

%extend from that point

nxSeed = pCur(1,1); nySeed = pCur(2,1);
increY = +1;
ny = nySeed;
quit = false;
nxNextLineSeed = pCur(1,1);
foundTurningPointSeed = false;
nxTurningPointSeed = 0;

while (~quit) %while not quit %the y loop
    foundNextLineSeed = false;
    
    nxTurnLR = nxNextLineSeed;
    %move right
    nx = nxTurnLR;
    while (insideTriangle([nx;ny], vtx1, vtx2, vtx3, pdir, thetaR, R0)) %the x loop
        
        %if point is found, do something (save it)
        %plot3(nx,ny,0, 'or','MarkerSize',12);
        graylevel = abs(insideTriangle([nx;ny], vtx1, vtx2, vtx3, pdir, thetaR, R0));
        figure(1)
        hold on
        if (increY == +1)
            rectangle('Position',[nx,ny,1,1],'FaceColor',[1-graylevel 1-graylevel 1]);
        else
            rectangle('Position',[nx,ny,1,1],'FaceColor',[1-graylevel 1-graylevel 0]);
        end
        
        %scan for next seed
        if (foundNextLineSeed == false)
            if (insideTriangle([nx;ny+increY], vtx1, vtx2, vtx3, pdir, thetaR, R0))
                foundNextLineSeed = true;
                nxNextLineSeed = nx;
            end
        end
        
        %if it is the first line, scan for turning point seed
        if (ny == nySeed && foundTurningPointSeed == false && increY == +1)
            if (insideTriangle([nx;ny-1], vtx1, vtx2, vtx3, pdir, thetaR, R0))
                foundTurningPointSeed = true;
                nxTurningPointSeed = nx;
            end
        end
        
        %move to the right
        nx = nx + 1;
    end
    
    %start over from the left
    nx = nxTurnLR - 1;
    while (insideTriangle([nx;ny], vtx1, vtx2, vtx3, pdir, thetaR, R0)) %the x loop
        
        %if point is found, do something (save it)
        %plot3(nx,ny,0, 'or','MarkerSize',12);
        graylevel = abs(insideTriangle([nx;ny], vtx1, vtx2, vtx3, pdir, thetaR, R0));
        figure(1)
        hold on
        if (increY == +1)
            rectangle('Position',[nx,ny,1,1],'FaceColor',[1-graylevel 1 1-graylevel]);
        else
            rectangle('Position',[nx,ny,1,1],'FaceColor',[1-graylevel 0 1-graylevel]);
        end
        
        %scan for next seed
        if (foundNextLineSeed == false)
            if (insideTriangle([nx;ny+increY], vtx1, vtx2, vtx3, pdir, thetaR, R0))
                foundNextLineSeed = true;
                nxNextLineSeed = nx;
            end
        end
        
        %if it is the first line, scan for turning point seed
        if (ny == nySeed && foundTurningPointSeed == false && increY == +1)
            if (insideTriangle([nx;ny-1], vtx1, vtx2, vtx3, pdir, thetaR, R0))
                foundTurningPointSeed = true;
                nxTurningPointSeed = nx;
            end
        end
        
        %move to the left
        nx = nx - 1;
    end
    
    %move to next line
    if (foundNextLineSeed == true)
        ny = ny+increY;
    elseif (increY == +1 && foundTurningPointSeed == true)
        ny = nySeed - 1;
        nxNextLineSeed = nxTurningPointSeed;
        increY = -1;
    else
        quit = true;
    end
end


end

function [output] = insideTriangle(p, vtx1, vtx2, vtx3, pdir, thetaR, R0)
output = 0;
nx = p(1);
ny = p(2);

[output0, p1, p2, p3, p4] = find4points(nx, ny, thetaR, R0);
if (output0 == 0)
    return;
end
[output1, alpha1, beta1] = maptotriangle(vtx1, vtx2, vtx3, p1, pdir);
[output2, alpha2, beta2] = maptotriangle(vtx1, vtx2, vtx3, p2, pdir);
[output3, alpha3, beta3] = maptotriangle(vtx1, vtx2, vtx3, p3, pdir);
[output4, alpha4, beta4] = maptotriangle(vtx1, vtx2, vtx3, p4, pdir);

if (output1 == 0 || output2 == 0 || output3 == 0 || output4 == 0)
    output = 0;
    return;
end
output = SutherlandHogdman([alpha1; beta1],[alpha2; beta2],[alpha3; beta3],[alpha4; beta4]);
% if (alpha >= 0 && beta >= 0 && (alpha+beta) <=1)
%     output = 1;
% end
end

function [output, pGuess] = guestPoint(vtx, thetaR, R0)
%theta R should be in degree

output = true;
x = vtx(1);
y = vtx(2);
z = vtx(3);

%ommit the checking, assume calling function has done it
% if ((x*x+y*y)>(R0*R0))
%     output = false;
%     return;
% else
%     z = sqrt(R0*R0 - x*x - y*y);
% end
    
thetaR = thetaR/180*pi;
ny = asin(y/R0)/thetaR;
Rp = R0*cos(ny*thetaR);
nx = asin(x/Rp)/thetaR;
ny = floor(ny);
nx = floor(nx);
[output0, p1, p2, p3, p4] = find4points(nx, ny, thetaR, R0);

if (output0 == 0)
    output = false;
    return;
end

pGuess = [nx;ny];
end
