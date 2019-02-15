[x,y,z] = sphere;
figure
surf(x,y,z)
axis equal
hold on
clear all
R = 1;
thetaR = 10;
halfStepCount = floor(90/thetaR);
%convert to radian
thetaR = thetaR/180*pi;
scalarR = R*thetaR; 
for ny = -halfStepCount:1:halfStepCount
j = ny+halfStepCount+1;
Rp(j) = cos(ny*thetaR);
thetaRp(j) = scalarR/Rp(j);
for nx = -halfStepCount:1:halfStepCount
i = nx+halfStepCount+1;
angleY = ny*thetaR;
angleX = nx*thetaRp(j);
if ((abs(angleY) > pi/2) || (abs(angleX) > pi/2)) 
z(i,j) = 0;
continue;
end
y(i,j) = sin(angleY);
x(i,j) = Rp(j)*sin(angleX);
z(i,j) = sqrt(1-x(i,j)*x(i,j)-y(i,j)*y(i,j));

end
end

for ny = -halfStepCount:1:halfStepCount
j = ny+halfStepCount+1;
for nx = -halfStepCount:1:halfStepCount
i = nx+halfStepCount+1;

yp(i,j) = y(i,j);
if (ny > 0)
    xp(i,j) = Rp(j)*sin(nx*thetaRp(j-1));
elseif (ny < 0)
    xp(i,j) = Rp(j)*sin(nx*thetaRp(j+1));
elseif (ny == 0)
    xp(i,j) = x(i,j);
end
zp(i,j) = sqrt(1-xp(i,j)*xp(i,j)-yp(i,j)*yp(i,j));

end
end

plot3(x,y,z,'xk','MarkerSize',10);
hold on
plot3(xp,yp,zp,'xr','MarkerSize',10);
xlabel('xxxxxxxxxxxx'); ylabel('yyyyyyyyyyyyy');
hold on

returnvalue = draw4pixaround(0,0,thetaR/pi*180,1)
