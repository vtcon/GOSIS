function [output, alpha, beta] = maptotriangle(p1, p2, p3, px, pdir)

%draw just because I can!
figure(3)
hold on
A=[p1(1) p2(1) p3(1) p1(1)];
B=[p1(2) p2(2) p3(2) p1(2)];
C=[p1(3) p2(3) p3(3) p1(3)];
plot3(A,B,C,'k-');

plot3(px(1),px(2),px(3), 'ok','MarkerSize',9);

A=[0 pdir(1)];
B=[0 pdir(2)];
C=[0 pdir(3)];
plot3(A,B,C,'b-');


%some tests here
output = 0;
alpha = 0;
beta = 0;

%the triangle is colinear
if (norm(cross(p2-p1,p3-p1)) == 0)
    output = 0;
    return;
end

%pdir and triangle coplanar
if (dot(cross(p2-p1,p3-p1),pdir) == 0)
    output = 0;
    return;
end


%px and triangle coplanar
if (dot(cross(p2-p1,p3-p1),px-p1) == 0)
    pdir = [0 0 1];
end

%if any component of pdir is zero
nrZero = 0;
if (pdir(1) == 0)
    nrZero = nrZero + 1;
    A1 = px(1) - p1(1);
    A2 = p2(1) - p1(1);
    A3 = p3(1) - p1(1);
end

if (pdir(2) == 0)
    if (nrZero == 1)
        B1 = A1;
        B2 = A2; 
        B3 = A3;
    end
    nrZero = nrZero + 1;
    A1 = px(2) - p1(2);
    A2 = p2(2) - p1(2);
    A3 = p3(2) - p1(2);
end

if (pdir(3) == 0)
    if (nrZero == 1)
        B1 = A1;
        B2 = A2; 
        B3 = A3;
    elseif (nrZero == 2)
        alpha = 0;
        beta = 0;
        return;
    end
    nrZero = nrZero + 1;
    A1 = px(3) - p1(3);
    A2 = p2(3) - p1(3);
    A3 = p3(3) - p1(3);
end

if (nrZero == 0)
A1 = px(1)*pdir(2) - p1(1)*pdir(2) - px(2)*pdir(1) + p1(2)*pdir(1); 
A2 = p2(1)*pdir(2) - p1(1)*pdir(2) - p2(2)*pdir(1) + p1(2)*pdir(1);
A3 = p3(1)*pdir(2) - p1(1)*pdir(2) - p3(2)*pdir(1) + p1(2)*pdir(1);
end

if (nrZero <= 1)
if (nrZero == 0 || pdir(2) == 0)
    denoLeft = pdir(1);
    numLeft1 = px(1);
    numLeft2 = p2(1);
    numLeft3 = p3(1);
    numLeft4 = p1(1);
    denoRight = pdir(3);
    numRight1 = px(3);
    numRight2 = p2(3);
    numRight3 = p3(3);
    numRight4 = p1(3);
elseif (pdir(1) == 0)
    denoLeft = pdir(2);
    numLeft1 = px(2);
    numLeft2 = p2(2);
    numLeft3 = p3(2);
    numLeft4 = p1(2);
    denoRight = pdir(3);
    numRight1 = px(3);
    numRight2 = p2(3);
    numRight3 = p3(3);
    numRight4 = p1(3);
elseif (pdir(3) == 0)
    denoLeft = pdir(1);
    numLeft1 = px(1);
    numLeft2 = p2(1);
    numLeft3 = p3(1);
    numLeft4 = p1(1);
    denoRight = pdir(2);
    numRight1 = px(2);
    numRight2 = p2(2);
    numRight3 = p3(2);
    numRight4 = p1(2);
end
B1 = (numLeft1 - numLeft4)*denoRight - (numRight1 - numRight4)*denoLeft;
B2 = (numLeft2 - numLeft4)*denoRight - (numRight2 - numRight4)*denoLeft;
B3 = (numLeft3 - numLeft4)*denoRight - (numRight3 - numRight4)*denoLeft;
end

alpha = (A1*B3 - B1*A3)/(A2*B3 - B2*A3);
beta = (A2*B1 - B2*A1)/(A2*B3 - B2*A3);
output = 1;



pdraw = alpha*(p2-p1) + beta*(p3-p1) + p1;

A=[pdraw(1) px(1)];
B=[pdraw(2) px(2)];
C=[pdraw(3) px(3)];
plot3(A,B,C,'r-');

plot3(pdraw(1),pdraw(2),pdraw(3), 'xb','MarkerSize',12);
hold on

end

