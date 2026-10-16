function [frozen,trainingLog] = AQSC_TrainAgents_v5(cfg,profiles)
defs={'cnn_mask','cnn',true,1;'dense_mask','dense',true,2;'cnn_unmasked','cnn',false,3};
logs=cell(size(defs,1),1);frozen=struct();
for j=1:size(defs,1)
 name=defs{j,1};fprintf('Training %s\n',name);agent=AQSC_AI_Agent_v5(defs{j,2},defs{j,3},cfg.agent,cfg.masterSeed+defs{j,4});
 ret=zeros(cfg.trainEpisodes,1);comp=ret;
 for ep=1:cfg.trainEpisodes
  nUE=cfg.ueGrid(randi(numel(cfg.ueGrid)));trace=AQSC_3GPP_UMaTrace_v5(cfg,cfg.masterSeed+10000+ep,nUE,cfg.trainFrames);
  prev=initial(cfg);hist=zeros(cfg.agent.History,8);last=4;ok=0;
  for k=1:cfg.trainFrames
   hist=[hist(2:end,:);AQSC_MakeObservation_v5(prev,trace,k,cfg)];mask=AQSC_ActionMask_v5(trace.Required(k),profiles);
   action=agent.selectAction(hist,mask,true);met=AQSC_ProfileCost_v5(profiles(action),nUE,trace.Capacity_Mbps(k),trace.BackgroundCPU(k),trace.Threat(k),cfg);
   compliant=profiles(action).Category>=trace.Required(k);reward=AQSC_Reward_v5(met,action~=last,compliant,cfg);
   nextk=min(k+1,cfg.trainFrames);hn=[hist(2:end,:);AQSC_MakeObservation_v5(met,trace,nextk,cfg)];nextMask=AQSC_ActionMask_v5(trace.Required(nextk),profiles);
   agent.remember(hist,action,reward,hn,k==cfg.trainFrames,nextMask);agent.learn();ret(ep)=ret(ep)+reward;ok=ok+compliant;prev=met;last=action;
  end
  comp(ep)=100*ok/cfg.trainFrames;if mod(ep,50)==0,fprintf('  %d/%d return %.2f compliance %.2f%%\n',ep,cfg.trainEpisodes,mean(ret(ep-49:ep)),mean(comp(ep-49:ep)));end
 end
 frozen.(name)=agent.freeze();Agent=repmat(string([defs{j,2} '-' ternary(defs{j,3},'masked','unmasked')]),cfg.trainEpisodes,1);
 logs{j}=table((1:cfg.trainEpisodes)',ret,comp,Agent,'VariableNames',{'Episode','Return','Compliance_pct','Agent'});
end
trainingLog=vertcat(logs{:});writetable(trainingLog,fullfile(cfg.outputDir,'training_log_v5.csv'));save(fullfile(cfg.outputDir,'frozen_agents_v5.mat'),'frozen','cfg','profiles');
end
function p=initial(cfg),p=struct('Latency_ms',2,'Loss',.001,'Throughput_Mbps',cfg.offeredLoad_Mbps,'CryptoUtil',.25);end
function x=ternary(c,a,b),if c,x=a;else,x=b;end,end
