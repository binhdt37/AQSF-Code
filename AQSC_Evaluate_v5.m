function results = AQSC_Evaluate_v5(frozen,cfg,profiles)
if ~exist(cfg.outputDir,'dir'),mkdir(cfg.outputDir);end
cnn=AQSC_AI_Agent_v5('cnn',true,cfg.agent,cfg.masterSeed+101);cnn.loadFrozen(frozen.cnn_mask);
dense=AQSC_AI_Agent_v5('dense',true,cfg.agent,cfg.masterSeed+102);dense.loadFrozen(frozen.dense_mask);
unmasked=AQSC_AI_Agent_v5('cnn',false,cfg.agent,cfg.masterSeed+103);unmasked.loadFrozen(frozen.cnn_unmasked);
methods={'CNN-DQN+Mask','DQN-only+Mask','CNN-DQN penalty-only','Heuristic+Mask','Random-feasible','Static-C5-long'};
S=repmat(emptySummary(),0,1);F=repmat(emptyFrame(),0,1);ns=0;nf=0;
for ir=1:cfg.monteCarloRuns
 seed=cfg.testSeeds(ir);
 for iu=1:numel(cfg.ueGrid)
  nUE=cfg.ueGrid(iu);trace=AQSC_3GPP_UMaTrace_v5(cfg,seed,nUE,cfg.evalFrames);
  for im=1:numel(methods)
   method=methods{im};prev=initial(cfg);hist=zeros(cfg.agent.History,8);last=4;
    lat=zeros(cfg.evalFrames,1);thr=lat;loss=lat;comp=lat;util=lat;expo=lat;period=lat;sw=lat;invalid=0;intervene=0;
   rng(seed+100*im+nUE,'twister');
   for k=1:cfg.evalFrames
    hist=[hist(2:end,:);AQSC_MakeObservation_v5(prev,trace,k,cfg)];mask=AQSC_ActionMask_v5(trace.Required(k),profiles);
    switch method
     case 'CNN-DQN+Mask',[action,proposal]=cnn.selectAction(hist,mask,false);
     case 'DQN-only+Mask',[action,proposal]=dense.selectAction(hist,mask,false);
     case 'CNN-DQN penalty-only',[action,proposal]=unmasked.selectAction(hist,true(1,6),false);
     case 'Heuristic+Mask',action=heuristic(trace,k,mask,profiles,cfg);proposal=action;
     case 'Random-feasible',legal=find(mask);action=legal(randi(numel(legal)));proposal=action;
     otherwise,action=2;proposal=2;
    end
    invalid=invalid+~mask(proposal);intervene=intervene+(action~=proposal);
    met=AQSC_ProfileCost_v5(profiles(action),nUE,trace.Capacity_Mbps(k),trace.BackgroundCPU(k),trace.Threat(k),cfg);
    compliant=profiles(action).Category>=trace.Required(k);lat(k)=met.Latency_ms;thr(k)=met.Throughput_Mbps;loss(k)=met.Loss;comp(k)=compliant;util(k)=met.CryptoUtil;expo(k)=met.Exposure;period(k)=profiles(action).Refresh_s;sw(k)=action~=last;
    nf=nf+1;F(nf,1)=frameRow(ir,seed,nUE,method,k,trace,action,profiles(action),met,compliant);prev=met;last=action;
   end
   ns=ns+1;S(ns,1)=summaryRow(ir,seed,nUE,method,lat,thr,loss,comp,util,expo,period,sw,invalid,intervene,cfg);
  end
 end
end
summary=struct2table(S);frames=struct2table(F);statistics=pairedStats(summary,cfg);
ablation=groupsummary(summary,'Method','mean',{'MeanLatency_ms','P95Latency_ms','MeanThroughput_Mbps','SecurityCompliance_pct','MeanCryptoUtil_pct','MeanExposure','MeanCryptoperiod_s','DeadlineMiss_pct','InvalidProposal_pct','MaskIntervention_pct'});
writetable(summary,fullfile(cfg.outputDir,'run_level_summary_v5.csv'));writetable(frames,fullfile(cfg.outputDir,'frame_level_trace_v5.csv'));
writetable(statistics,fullfile(cfg.outputDir,'paired_statistics_v5.csv'));writetable(ablation,fullfile(cfg.outputDir,'ablation_summary_v5.csv'));
AQSC_PlotFigure6_v5(summary,cfg);results=struct('summary',summary,'frames',frames,'statistics',statistics,'ablation',ablation,'profiles',profiles);
save(fullfile(cfg.outputDir,'AQSC_results_v5.mat'),'results','-v7.3');
end
function a=heuristic(trace,k,mask,profiles,cfg)
legal=find(mask);short=trace.Threat(k)>.70 && trace.BackgroundCPU(k)<.78 && trace.Capacity_Mbps(k)>1.05*cfg.offeredLoad_Mbps;a=legal(end);
for j=legal,if contains(profiles(j).Name,'short')==short,a=j;return;end,end
end
function r=summaryRow(run,seed,ue,method,lat,thr,loss,comp,util,expo,period,sw,invalid,intervene,cfg)
r=emptySummary();r.Run=run;r.Seed=seed;r.UE=ue;r.Method={method};r.MeanLatency_ms=mean(lat);r.P95Latency_ms=pct(lat,95);r.P99Latency_ms=pct(lat,99);
r.MeanThroughput_Mbps=mean(thr);r.MeanLoss_pct=100*mean(loss);r.DeadlineMiss_pct=100*mean(lat>cfg.deadline_ms);r.SecurityCompliance_pct=100*mean(comp);
r.MeanCryptoUtil_pct=100*mean(util);r.MeanExposure=mean(expo);r.MeanCryptoperiod_s=mean(period);r.SwitchCount=sum(sw);r.InvalidProposal_pct=100*invalid/cfg.evalFrames;r.MaskIntervention_pct=100*intervene/cfg.evalFrames;
end
function r=frameRow(run,seed,ue,method,k,trace,action,p,met,comp)
r=emptyFrame();r.Run=run;r.Seed=seed;r.UE=ue;r.Method={method};r.Frame=k;r.RequiredCategory=trace.Required(k);r.Action=action;r.ActionName={p.Name};
r.Latency_ms=met.Latency_ms;r.Throughput_Mbps=met.Throughput_Mbps;r.Loss_pct=100*met.Loss;r.Compliance=comp;r.CryptoUtil_pct=100*met.CryptoUtil;
r.CQI=trace.CQI(k);r.CellEdgeSINR_dB=trace.CellEdgeSINR_dB(k);r.Capacity_Mbps=trace.Capacity_Mbps(k);
end
function r=emptySummary(),r=struct('Run',0,'Seed',0,'UE',0,'Method',{{''}},'MeanLatency_ms',0,'P95Latency_ms',0,'P99Latency_ms',0,'MeanThroughput_Mbps',0,'MeanLoss_pct',0,'DeadlineMiss_pct',0,'SecurityCompliance_pct',0,'MeanCryptoUtil_pct',0,'MeanExposure',0,'MeanCryptoperiod_s',0,'SwitchCount',0,'InvalidProposal_pct',0,'MaskIntervention_pct',0);end
function r=emptyFrame(),r=struct('Run',0,'Seed',0,'UE',0,'Method',{{''}},'Frame',0,'RequiredCategory',0,'Action',0,'ActionName',{{''}},'Latency_ms',0,'Throughput_Mbps',0,'Loss_pct',0,'Compliance',false,'CryptoUtil_pct',0,'CQI',0,'CellEdgeSINR_dB',0,'Capacity_Mbps',0);end
function p=pct(x,q),x=sort(x(:));z=1+(numel(x)-1)*q/100;l=floor(z);h=ceil(z);if l==h,p=x(l);else,p=x(l)+(z-l)*(x(h)-x(l));end,end
function T=pairedStats(S,cfg)
comparators={'DQN-only+Mask','Heuristic+Mask','Static-C5-long','CNN-DQN penalty-only'};metrics={'MeanLatency_ms',-1;'P95Latency_ms',-1;'MeanThroughput_Mbps',1;'MeanLoss_pct',-1;'DeadlineMiss_pct',-1;'SecurityCompliance_pct',1;'MeanCryptoUtil_pct',-1;'MeanExposure',-1};
R=repmat(struct('UE',0,'Outcome',{{''}},'Comparator',{{''}},'CNNMean',0,'ComparatorMean',0,'Benefit',0,'CI95Low',0,'CI95High',0,'EffectDz',0,'PairedT_P',0),0,1);n=0;
for ue=cfg.ueGrid
 a=S(S.UE==ue & strcmp(S.Method,'CNN-DQN+Mask'),:);[~,o]=sort(a.Seed);a=a(o,:);
 for c=1:numel(comparators),b=S(S.UE==ue & strcmp(S.Method,comparators{c}),:);[~,o]=sort(b.Seed);b=b(o,:);
  for m=1:size(metrics,1),name=metrics{m,1};sgn=metrics{m,2};d=sgn*(a.(name)-b.(name));boots=zeros(cfg.bootstrapReplicates,1);for z=1:numel(boots),boots(z)=mean(d(randi(numel(d),numel(d),1)));end
   n=n+1;R(n,1).UE=ue;R(n).Outcome={name};R(n).Comparator={comparators{c}};R(n).CNNMean=mean(a.(name));R(n).ComparatorMean=mean(b.(name));R(n).Benefit=mean(d);R(n).CI95Low=pct(boots,2.5);R(n).CI95High=pct(boots,97.5);R(n).EffectDz=mean(d)/max(std(d),eps);[~,R(n).PairedT_P]=ttest(d);
  end
 end
end
T=struct2table(R);
end
function p=initial(cfg),p=struct('Latency_ms',2,'Loss',.001,'Throughput_Mbps',cfg.offeredLoad_Mbps,'CryptoUtil',.25);end
