function [output] = draw4pixaround(x, y , thetaR, R0)
%theta R should be in degree
if ((x*x+y*y)>(R0*R0))
    output = false;
    return;
else
    z = sqrt(R0*R0 - x*x - y*y);
end
    
thetaR = thetaR/180*pi;
ny = asin(y/R0)/thetaR;
Rp = R0*cos(ny*thetaR);
nx = asin(x/Rp)/thetaR;
ny = floor(ny);
nx = floor(nx);


hold on
plot3(x,y,z, 'ob','MarkerSize',12);
hold on

output = draw4pix(nx,ny,thetaR/pi*180,R0);
end
