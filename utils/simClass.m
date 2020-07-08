function Y = simClass(model,X)
%
%
%
kx = kernel_matrix(model.xtrain(model.selector, 1:model.x_dim), ...
		   model.kernel_type, model.kernel_pars,X);
Y = kx'*(model.alpha(model.selector,1:model.y_dim).*model.ytrain(model.selector,1:model.y_dim))+ones(size(kx,2),1)*model.b(:,1:model.y_dim);
end