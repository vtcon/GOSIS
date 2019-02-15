function [xplot, yplot, zplot, output] = draw4pix(nx, ny, thetaR, R0)

xplot = 0;
yplot = 0;
zplot = 0;

thetaR = thetaR/180*pi;
r = thetaR*R0;
thetaY1 = ny*thetaR;
thetaY2 = (ny+1)*thetaR;

if ((thetaY1 < -pi/2) || (thetaY2 > pi/2))
    output = false;
    return;
end

y1 = R0*sin(thetaY1);
y2 = y1;
y3 = R0*sin(thetaY2);
y4 = y3;
R1p = R0*cos(thetaY1);
R2p = R0*cos(thetaY2);

if (ny >= 0)
    thetaX1 = nx*r/R1p;
    thetaX2 = (nx+1)*r/R1p;
elseif (ny < 0)
    thetaX1 = nx*r/R2p;
    thetaX2 = (nx+1)*r/R2p;
end

if ((thetaX1 < -pi/2) || (thetaX2 > pi/2))
        output = false;
        return;
end

x1 = R1p*sin(thetaX1);
x2 = R1p*sin(thetaX2);
x3 = R2p*sin(thetaX2);
x4 = R2p*sin(thetaX1);

z1 = sqrt(1-x1*x1-y1*y1);
z2 = sqrt(1-x2*x2-y2*y2);
z3 = sqrt(1-x3*x3-y3*y3);
z4 = sqrt(1-x4*x4-y4*y4);

xplot = [x1 x2 x3 x4];
yplot = [y1 y2 y3 y4];
zplot = [z1 z2 z3 z4];


hold on
plot3(xplot,yplot,zplot,'ok','MarkerSize',10);
hold on

output = true;
end

