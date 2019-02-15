function [output, p1, p2, p3, p4] = sortCCW(c1, c2, c3, c4)

p1 = c1;p2 = c2;p3 = c3;p4 = c4;
output = 1;

c12 = c2 - c1;
c13 = c3 - c1;
c14 = c4 - c1;

crp1 = dot(c13,[c12(2,1);-c12(1,1)]);
crp2 = dot(c14,[c12(2,1);-c12(1,1)]);
if (crp1 == 0)
    if (crp2 == 0)
        %all colinear
        output = 0;
        return;
    else
        dotcolinear = dot(c13,normalize(c12));
        if (dotcolinear > norm(c12))
            p3 = c3;
            if (crp2 > 0)
                p2 = c4; p4 = c2;
                return;
            else
                p2 = c2; p4 = c4;
                return;
            end
        elseif (dotcolinear > 0)
            p3 = c2; 
            if (crp2 > 0)
                p2 = c4; p4 = c3;
                return;
            else
                p2 = c3; p4 = c4;
                return;
            end
        else
            p3 = c4; 
            if (crp2 > 0)
                p2 = c3; p4 = c3;
                return;
            else
                p2 = c2; p4 = c3;
                return;
            end
        end
    end
elseif (crp2 == 0)
    dotcolinear = dot(c14,normalize(c12));
    if (dotcolinear > norm(c12))
        p3 = c4; 
        if (crp1 > 0)
            p2 = c3; p4 = c2;
            return;
        else
            p2 = c2; p4 = c3;
            return;
        end
    elseif (dotcolinear > 0)
         p3 = c2; 
        if (crp1 > 0)
            p2 = c3; p4 = c4;
            return;
        else
            p2 = c4; p4 = c3;
            return;
        end
    else
        p3 = c3; 
        if (crp1 > 0)
            p2 = c4; p4 = c2;
            return;
        else
            p2 = c2; p4 = c4;
            return;
        end
    end
end

dot1 = dot(c12, c13)/(norm(c12)*norm(c13));
dot2 = dot(c12, c14)/(norm(c12)*norm(c14));

if (crp1 > 0)
    if (crp2 > 0)
        p4 = c2;
        if (dot1 > dot2)
            p2 = c4; p3 = c3; 
            return;
        elseif (dot1 < dot2)
            p2 = c3; p3 = c4;
            return;
        else
            %colinear
            if (norm(c13) >= norm(c14))
                p2 = c4; p3 = c3;
                return;
            else
                p2 = c3; p3 = c4;
                return;
            end
        end
    else
        p2 = c3; p3 = c2; p4 = c4;
        return;
    end
else
    if (crp2 > 0)
        p2 = c4; p3 = c2; p4 = c3;
        return;
    else
        p2 = c2;
        if (dot1 > dot2)
            p3 = c3; p4 = c4;
            return;
        elseif (dot1 < dot2)
            p3 = c4; p4 = c3;
            return;
        else
            %colinear
            if (norm(c13) >= norm(c14))
                p3 = c3; p4 = c4;
                return;
            else
                p3 = c4; p4 = c3;
                return;
            end
        end
    end
end

output = 1;
end

