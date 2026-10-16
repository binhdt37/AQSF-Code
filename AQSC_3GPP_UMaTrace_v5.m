function trace = AQSC_3GPP_UMaTrace_v5(cfg,seed,nUE,nFrames)
% System-level large-scale channel abstraction using the UMa LOS probability
% and LOS/NLOS path-loss equations from 3GPP TR 38.901. CQI efficiencies
% follow TS 38.214. This is not a waveform/conformance test.
rng(seed+17*nUE,'twister');
trace.State=zeros(nFrames,1); trace.State(1)=sample(cfg.initialStateProb);
for k=2:nFrames, trace.State(k)=sample(cfg.transition(trace.State(k-1),:)); end
trace.Required=reshape(cfg.requiredCategory(trace.State),[],1);
trace.Threat=zeros(nFrames,1); trace.Threat(1)=cfg.threatBase(trace.State(1))+.07*randn;
for k=2:nFrames
 trace.Threat(k)=.84*trace.Threat(k-1)+.16*cfg.threatBase(trace.State(k))+.045*randn;
end
trace.Threat=min(max(trace.Threat,0),1);
trace.Anomaly=min(max(.08+.78*(trace.State==1)+.10*randn(nFrames,1),0),1);

m=min(nUE,cfg.probeUEs); radius=35+(500-35)*sqrt(rand(m,1)); angle=2*pi*rand(m,1);
x=radius.*cos(angle); y=radius.*sin(angle); isLOS=rand(m,1)<losProbability(radius);
sigma=4*isLOS+6*(~isLOS); shadow=sigma.*randn(m,1);
a=(0:5)'*pi/3; nb=[1000*cos(a),1000*sin(a)];
trace.Capacity_Mbps=zeros(nFrames,1);trace.CQI=zeros(nFrames,1);trace.CellEdgeSINR_dB=zeros(nFrames,1);
trace.BackgroundCPU=zeros(nFrames,1);trace.BackgroundCPU(1)=clip(.20+.15*nUE/10000+.08*randn,.05,.75);
for k=1:nFrames
 if k>1
  angle=angle+.0015*randn(m,1);x=radius.*cos(angle);y=radius.*sin(angle);
  shadow=.96*shadow+sqrt(1-.96^2).*sigma.*randn(m,1);
  trace.BackgroundCPU(k)=clip(.92*trace.BackgroundCPU(k-1)+.08*(.20+.25*nUE/10000)+.035*randn,.05,.90);
 end
 desired_dBm=46+8-umaPL(radius,isLOS,cfg.carrierGHz)-shadow;
 interf_mW=zeros(m,1);
 for b=1:6
  d=sqrt((x-nb(b,1)).^2+(y-nb(b,2)).^2);
  interf_mW=interf_mW+10.^((46+8-umaPL(max(d,35),false(size(d)),cfg.carrierGHz)-7)/10);
 end
 noise_mW=10.^((-174+10*log10(cfg.bandwidthMHz*1e6)+7)/10);
 sinr=10*log10(10.^(desired_dBm/10)./(interf_mW+noise_mW));
 trace.CellEdgeSINR_dB(k)=prctileLocal(sinr,100*(1-cfg.targetCoverage));
 [trace.Capacity_Mbps(k),trace.CQI(k)]=cqiCapacity(trace.CellEdgeSINR_dB(k),cfg);
end
end
function p=losProbability(d)
d=max(d,1);p=min(18./d,1).*(1-exp(-d/63))+exp(-d/63);
end
function pl=umaPL(d2d,isLOS,fc)
d3=sqrt(d2d.^2+(25-1.5)^2);los=28+22*log10(d3)+20*log10(fc);
nlos=13.54+39.08*log10(d3)+20*log10(fc);pl=los;pl(~isLOS)=max(los(~isLOS),nlos(~isLOS));
end
function [cap,cqi]=cqiCapacity(sinr,cfg)
eff=[0 .1523 .2344 .3770 .6016 .8770 1.1758 1.4766 1.9141 2.4063 2.7305 3.3223 3.9023 4.5234 5.1152 5.5547 6.2266 6.9141 7.4063];
thr=10*log10(max(2.^(eff/.72)-1,1e-12))+cfg.cqiMargin_dB;cqi=find(sinr>=thr,1,'last');
if isempty(cqi),cqi=1;end;cap=cfg.bandwidthMHz*cfg.mimoLayers*eff(cqi)*.82;cqi=cqi-1;
end
function x=clip(x,a,b),x=min(max(x,a),b);end
function i=sample(p),u=rand;c=cumsum(p/sum(p));i=find(u<=c,1);end
function q=prctileLocal(x,p)
x=sort(x(:));pos=1+(numel(x)-1)*p/100;lo=floor(pos);hi=ceil(pos);
if lo==hi,q=x(lo);else,q=x(lo)+(pos-lo)*(x(hi)-x(lo));end
end
