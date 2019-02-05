clear all
figure(1)
hold on
axis equal
xlabel('xxxxxxxxxxxx'); ylabel('yyyyyyyyyyyyy');
figure(2)
hold on
axis equal
xlabel('xxxxxxxxxxxx'); ylabel('yyyyyyyyyyyyy');
figure(3)
hold on
axis equal
xlabel('xxxxxxxxxxxx'); ylabel('yyyyyyyyyyyyy');

format long

p2 = [-0.5;-0.5;0];
p1 = [1;0;0];
p3 = [-1;1.3;0];
pdir = [0.2;0.2;1];
thetaR = 15;
R0 = 2;

p1(3) = sqrt(R0*R0-p1(1)*p1(1)-p1(2)*p1(2));
p2(3) = sqrt(R0*R0-p2(1)*p2(1)-p2(2)*p2(2));
p3(3) = sqrt(R0*R0-p3(1)*p3(1)-p3(2)*p3(2));


rasterizev2(p1, p2, p3, pdir, thetaR, R0);