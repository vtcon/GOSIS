clear all
p1 = [0;0;1];
p2 = [1;0;0];
p3 = [0;1;1];
px = [-1;-1;0];
pdir = [1;1;1];

figure
hold on

[output, alpha, beta] = maptotriangle(p1, p2, p3, px, pdir)

axis equal
xlabel('xxxxxxxxxxxx'); ylabel('yyyyyyyyyyyyy');
hold on
