function s = stein(A,B)

s = log(det((A+B)/2)) - 0.5*(log(det(A)) + log(det(B)));

end

