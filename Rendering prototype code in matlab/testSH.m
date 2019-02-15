clear all
figure
hold on
axis equal
xlabel('xxxxxxxxxxxx'); ylabel('yyyyyyyyyyyyy');

c3 = [0.5;0.5];
c4 = [0;0];
c1 = [-0.5;0.5];
c2 = [0.5;-0.5];


output = SutherlandHogdman(c1, c2, c3, c4)