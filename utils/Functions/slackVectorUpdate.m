function slackVectorUpdate(options)
% Optimize the Cost w.r.t all the slack variables
global Vs Vd

vs = struct2vec(Vs); ns = length(vs);
vd = struct2vec(Vd); nd = length(vd);

% Problem Es---------------------------------------------------------------
manifold = euclideanfactory(ns,1);
problemEs.M = manifold;

problemEs.cost = @costEs;
problemEs.egrad = @gradEs;

% Problem Ed---------------------------------------------------------------
manifold = euclideanfactory(nd,1);
problemEd.M = manifold;

problemEd.cost = @costEd;
problemEd.egrad = @gradEd;

% Optimization-------------------------------------------------------------

vs = steepestdescent(problemEs,vs,options);
Vs = vec2struct(vs,Vs);

vd = steepestdescent(problemEd,vd,options);
Vd = vec2struct(vd,Vd); 

end

