classdef AQSC_AI_Agent_v5 < handle
% Toolbox-independent Double-DQN. Kind='cnn' uses a learned temporal 1-D
% convolution; Kind='dense' is the DQN-only ablation. Hard masking is applied
% both during exploration and Bellman target selection when Masked=true.
properties
 Kind='cnn'; Masked=true; StateDim=8; History=8; NumActions=6;
 ConvFilters=10; ConvKernel=3; HiddenDim=32;
 LearningRate=7e-4; DiscountFactor=.96; Epsilon=1; EpsilonMin=.05; EpsilonDecay=.9985;
 BatchSize=64; ReplayCapacity=30000; TargetUpdateInterval=250; GradientClip=5;
 Online; Target; AdamM; AdamV; AdamStep=0; LearnStep=0;
 ReplayHistory; ReplayAction; ReplayReward; ReplayNextHistory; ReplayDone; ReplayNextMask;
 ReplayCount=0; ReplayPosition=0;
end
methods
 function obj=AQSC_AI_Agent_v5(kind,masked,cfg,seed)
  if nargin<1,kind='cnn';end;if nargin<2,masked=true;end;if nargin<3,cfg=struct();end;if nargin<4,seed=1;end
  obj.Kind=kind;obj.Masked=masked;rng(seed,'twister');
  fields={'History','NumActions','ConvFilters','ConvKernel','HiddenDim','LearningRate','DiscountFactor','Epsilon','EpsilonMin','EpsilonDecay','BatchSize','ReplayCapacity','TargetUpdateInterval','GradientClip'};
  for i=1:numel(fields),if isfield(cfg,fields{i}),obj.(fields{i})=cfg.(fields{i});end,end
  if strcmp(obj.Kind,'cnn')
   obj.Online.K=sqrt(2/(obj.ConvKernel*obj.StateDim))*randn(obj.ConvFilters,obj.ConvKernel,obj.StateDim);
   obj.Online.bc=zeros(1,obj.ConvFilters);d=obj.ConvFilters+obj.StateDim;
  else,d=obj.StateDim;end
  obj.Online.W1=sqrt(2/d)*randn(d,obj.HiddenDim);obj.Online.b1=zeros(1,obj.HiddenDim);
  obj.Online.W2=sqrt(2/obj.HiddenDim)*randn(obj.HiddenDim,obj.NumActions);obj.Online.b2=zeros(1,obj.NumActions);
  obj.Target=obj.Online;obj.AdamM=obj.zeroLike(obj.Online);obj.AdamV=obj.zeroLike(obj.Online);
  obj.ReplayHistory=zeros(obj.ReplayCapacity,obj.History,obj.StateDim,'single');
  obj.ReplayNextHistory=zeros(obj.ReplayCapacity,obj.History,obj.StateDim,'single');
  obj.ReplayAction=zeros(obj.ReplayCapacity,1,'uint8');obj.ReplayReward=zeros(obj.ReplayCapacity,1,'single');
  obj.ReplayDone=false(obj.ReplayCapacity,1);obj.ReplayNextMask=false(obj.ReplayCapacity,obj.NumActions);
 end
 function [action,proposal,q]=selectAction(obj,history,mask,training)
  if nargin<4,training=false;end;q=obj.qValues(history,obj.Online);[~,proposal]=max(q);
  explore=training && rand<obj.Epsilon;
  if explore
   if obj.Masked,legal=find(mask);proposal=legal(randi(numel(legal)));else,proposal=randi(obj.NumActions);end
  end
  action=proposal;
  if obj.Masked && ~mask(action),legal=find(mask);[~,j]=max(q(legal));action=legal(j);end
  if training,obj.Epsilon=max(obj.EpsilonMin,obj.Epsilon*obj.EpsilonDecay);end
 end
 function remember(obj,h,a,r,hn,done,nextMask)
  obj.ReplayPosition=mod(obj.ReplayPosition,obj.ReplayCapacity)+1;p=obj.ReplayPosition;
  obj.ReplayHistory(p,:,:)=single(h);obj.ReplayAction(p)=uint8(a);obj.ReplayReward(p)=single(r);
  obj.ReplayNextHistory(p,:,:)=single(hn);obj.ReplayDone(p)=logical(done);obj.ReplayNextMask(p,:)=logical(nextMask);
  obj.ReplayCount=min(obj.ReplayCount+1,obj.ReplayCapacity);
 end
 function loss=learn(obj)
  if obj.ReplayCount<obj.BatchSize,loss=NaN;return;end
  idx=randperm(obj.ReplayCount,obj.BatchSize);h=double(obj.ReplayHistory(idx,:,:));hn=double(obj.ReplayNextHistory(idx,:,:));
  a=double(obj.ReplayAction(idx));r=double(obj.ReplayReward(idx));done=double(obj.ReplayDone(idx));mask=obj.ReplayNextMask(idx,:);
  if strcmp(obj.Kind,'dense'),x=reshape(h(:,end,:),obj.BatchSize,obj.StateDim);xn=reshape(hn(:,end,:),obj.BatchSize,obj.StateDim);else,x=h;xn=hn;end
  qOnline=obj.forward(x,obj.Online,false);if obj.Masked,qOnline(~mask)=-Inf;end
  [~,an]=max(qOnline,[],2);qTarget=obj.forward(xn,obj.Target,false);lin=sub2ind(size(qTarget),(1:obj.BatchSize)',an);
  y=r+obj.DiscountFactor*(1-done).*qTarget(lin);[q,cache]=obj.forward(x,obj.Online,true);
  chosen=sub2ind(size(q),(1:obj.BatchSize)',a);err=q(chosen)-y;loss=mean(err.^2);dQ=zeros(size(q));dQ(chosen)=2*err/obj.BatchSize;
  grad=obj.backward(cache,dQ);grad=obj.clipGradients(grad);obj.AdamStep=obj.AdamStep+1;
  obj.Online=obj.adamUpdate(obj.Online,grad);obj.LearnStep=obj.LearnStep+1;
  if mod(obj.LearnStep,obj.TargetUpdateInterval)==0,obj.Target=obj.Online;end
 end
 function frozen=freeze(obj),frozen=struct('Kind',obj.Kind,'Masked',obj.Masked,'Online',obj.Online);end
 function loadFrozen(obj,frozen),obj.Online=frozen.Online;obj.Target=frozen.Online;obj.Epsilon=0;end
end
methods(Access=private)
 function q=qValues(obj,h,net)
  if strcmp(obj.Kind,'dense'),x=reshape(h(end,:),1,obj.StateDim);else,x=reshape(h,1,obj.History,obj.StateDim);end
  q=obj.forward(x,net,false);
 end
 function [q,cache]=forward(obj,x,net,wantCache)
  if strcmp(obj.Kind,'cnn')
   n=size(x,1);L=obj.History-obj.ConvKernel+1;conv=zeros(n,L,obj.ConvFilters);
   for t=1:L
    seg=x(:,t:t+obj.ConvKernel-1,:);
    for f=1:obj.ConvFilters
     ker=reshape(net.K(f,:,:),1,obj.ConvKernel,obj.StateDim);
     conv(:,t,f)=reshape(sum(seg.*ker,[2 3]),n,1)+net.bc(f);
    end
   end
   relu=max(conv,0);pooled=reshape(mean(relu,2),n,obj.ConvFilters);last=reshape(x(:,end,:),n,obj.StateDim);feat=[pooled last];
  else,conv=[];relu=[];feat=x;end
  z=feat*net.W1+net.b1;hidden=max(z,0);q=hidden*net.W2+net.b2;
  if wantCache,cache=struct('x',x,'conv',conv,'relu',relu,'feat',feat,'z',z,'hidden',hidden);else,cache=[];end
 end
 function g=backward(obj,c,dQ)
  g.W2=c.hidden'*dQ;g.b2=sum(dQ,1);dZ=(dQ*obj.Online.W2').*(c.z>0);g.W1=c.feat'*dZ;g.b1=sum(dZ,1);
  if strcmp(obj.Kind,'cnn')
   n=size(c.x,1);L=size(c.conv,2);dFeat=dZ*obj.Online.W1';dPool=dFeat(:,1:obj.ConvFilters);
   dConv=repmat(reshape(dPool,n,1,obj.ConvFilters)/L,1,L,1).*(c.conv>0);
   g.K=zeros(size(obj.Online.K));g.bc=reshape(sum(dConv,[1 2]),1,obj.ConvFilters);
   for t=1:L
    seg=c.x(:,t:t+obj.ConvKernel-1,:);
    for f=1:obj.ConvFilters
     weighted=seg.*reshape(dConv(:,t,f),n,1,1);g.K(f,:,:)=reshape(sum(weighted,1),1,obj.ConvKernel,obj.StateDim)+g.K(f,:,:);
    end
   end
  end
 end
 function z=zeroLike(~,net),fn=fieldnames(net);for i=1:numel(fn),z.(fn{i})=zeros(size(net.(fn{i})));end,end
 function g=clipGradients(obj,g)
  fn=fieldnames(g);n=0;for i=1:numel(fn),n=n+sum(g.(fn{i})(:).^2);end;n=sqrt(n);
  if n>obj.GradientClip,f=obj.GradientClip/n;for i=1:numel(fn),g.(fn{i})=g.(fn{i})*f;end,end
 end
 function net=adamUpdate(obj,net,g)
  fn=fieldnames(net);b1=.9;b2=.999;e=1e-8;
  for i=1:numel(fn),f=fn{i};obj.AdamM.(f)=b1*obj.AdamM.(f)+(1-b1)*g.(f);obj.AdamV.(f)=b2*obj.AdamV.(f)+(1-b2)*g.(f).^2;
   mh=obj.AdamM.(f)/(1-b1^obj.AdamStep);vh=obj.AdamV.(f)/(1-b2^obj.AdamStep);net.(f)=net.(f)-obj.LearningRate*mh./(sqrt(vh)+e);end
 end
end
end
