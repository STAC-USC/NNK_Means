function [train_img, train_lbl, test_img, test_lbl] = generate_syndata(noise_level)

  n = 800;
  nc = 4;
  
  nsub = round(n/nc)*ones(nc,1);
  nsub(nc) = nsub(nc) + (n - sum(nsub));

  train_img = []; test_img = [];
  train_lbl = [];  test_lbl = [];
  for k = 1:nc
    phi = sort(1.5*linspace(0,1,nsub(k))*pi);
    radi = sqrt(pi+phi) - sqrt(pi);
    rot = (k-1)*(2*pi)/nc;
    R = [cos(rot) -sin(rot); sin(rot) cos(rot)];

    Xsub = [];
    Ysub = [];
    for i = 1:nsub(k)
      Xsub(2,i) = radi(i)*cos(phi(i)) + 0.1;
      Xsub(1,i) = radi(i)*sin(phi(i)) + 0.05;
      Xsub(:,i) = R * Xsub(:,i);
      Ysub(:, i) = zeros(1,nc);
      Ysub(k,i) = 1;
    end
    split_point = round(nsub(k)*0.75);
    seq = randperm(nsub(k));
    train_img = [train_img Xsub(:, seq(1:split_point))];
    train_lbl = [train_lbl Ysub(:, seq(1:split_point))];
    test_img = [test_img Xsub(:, seq(split_point+1:end))];
    test_lbl = [test_lbl Ysub(:, seq(split_point+1:end))];
  end

%   ridx = randperm(n);
%   n1 = round(n/4);
%   if noise_level == 1
%     train_img(:,ridx(1:n1)) = train_img(:,ridx(1:n1)) + 0.01*randn(2,n1); 
%   else
%     train_img(:,ridx(1:n1)) = train_img(:,ridx(1:n1)) + 0.05*randn(2,n1); 
%   end
%   train_img(:,ridx(n1+1:end)) = train_img(:,ridx(n1+1:end)) + 0.01*randn(2,n-n1);
train_img = train_img + 0.02*randn(2,0.75*n);
test_img = test_img + 0.02*randn(2,0.25*n);

  %%
  figure;
  hold on; axis off;
  col = lines(4);% {[1 0 0], [0.6 0 0], [0.3 0 0], [0 1 1]};
  for i = 1:nc
    plot(train_img(1,find(train_lbl(i, :))),train_img(2,find(train_lbl(i, :))),'x','Color', col(i,:),'markersize',7, 'MarkerFaceColor', col(i,:));
    plot(test_img(1,find(test_lbl(i, :))),test_img(2,find(test_lbl(i, :))),'o','Color', col(i,:),'markersize',7, 'MarkerFaceColor', col(i,:));
  end
  xlim([-1, 1]);
  ylim([-1, 1]);
  set(gca,'XTick',[-1, -0.5, 0, 0.5, 1], 'YTick', [-1, -0.5, 0, 0.5, 1]);
%   title('Simulated data with Train (x) and Test split (o)');
  hold off;
 %%
save('databases/simulated_data.mat', 'train_img', 'train_lbl', 'test_img', 'test_lbl');
