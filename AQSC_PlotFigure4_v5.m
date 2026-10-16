function f = AQSC_PlotFigure4_v5(trainingLog,cfg)
% AQSC_PlotFigure4_v5
% Reproduce manuscript Figure 4 from the MATLAB training log.
% The figure is intentionally left OPEN/VISIBLE after export so that the
% result is shown on screen when the script finishes.
%
% Manuscript caption:
% "Training-return ablation for temporal CNN-DQN, DQN-only, and
%  penalty-only CNN-DQN."

arguments
    trainingLog table
    cfg struct
end

required = {'Episode','Return','Compliance_pct','Agent'};
assert(all(ismember(required,trainingLog.Properties.VariableNames)), ...
    'trainingLog must contain: %s',strjoin(required,', '));

% Convert labels robustly whether readtable returned strings/cells/categorical.
agent = string(trainingLog.Agent);
window = 20;

idxCNN   = agent == "cnn-masked";
idxDense = agent == "dense-masked";
idxUnmask= agent == "cnn-unmasked";
assert(any(idxCNN) && any(idxDense) && any(idxUnmask), ...
    'Expected cnn-masked, dense-masked and cnn-unmasked rows in trainingLog.');

% Sort each series by episode before smoothing.
[eCNN,rCNN,cCNN]       = localSeries(trainingLog,idxCNN);
[eDense,rDense,cDense] = localSeries(trainingLog,idxDense);
[eUnmask,rUnmask,cUnmask] = localSeries(trainingLog,idxUnmask);

rCNN20   = movmean(rCNN,[window-1 0],'Endpoints','shrink');
rDense20 = movmean(rDense,[window-1 0],'Endpoints','shrink');
cCNN20   = movmean(cCNN,[window-1 0],'Endpoints','shrink');
cDense20 = movmean(cDense,[window-1 0],'Endpoints','shrink');
cUnmask20= movmean(cUnmask,[window-1 0],'Endpoints','shrink');

f = figure('Name','Figure 4 - Training ablation', ...
    'NumberTitle','off','Color','w','Visible','on', ...
    'Position',[80 120 1250 500]);
t = tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');

% Panel (a): fair representation ablation between the two masked agents.
% The unmasked controller has a different safety mechanism and large early
% penalty excursions, so its learning behavior is represented by compliance
% in panel (b), matching the supplied v5 audit figure.
ax1 = nexttile(t,1); hold(ax1,'on');
plot(ax1,eCNN,rCNN20,'LineWidth',1.8,'DisplayName','CNN-DQN + Mask');
plot(ax1,eDense,rDense20,'LineWidth',1.8,'DisplayName','DQN-only + Mask');
grid(ax1,'on'); box(ax1,'on');
xlabel(ax1,'Training episode');
ylabel(ax1,sprintf('%d-episode mean return',window));
title(ax1,'(a) Training-return ablation');
legend(ax1,'Location','best');
xlim(ax1,[1 max([eCNN;eDense])]);

% Panel (b): constraint behavior including the penalty-only ablation.
ax2 = nexttile(t,2); hold(ax2,'on');
plot(ax2,eCNN,cCNN20,'LineWidth',1.8,'DisplayName','CNN-DQN + Mask');
plot(ax2,eDense,cDense20,'LineWidth',1.8,'DisplayName','DQN-only + Mask');
plot(ax2,eUnmask,cUnmask20,'LineWidth',1.8,'DisplayName','CNN-DQN penalty-only');
yline(ax2,100,'--','100% target','LabelHorizontalAlignment','left', ...
    'LabelVerticalAlignment', 'bottom', ...
    'HandleVisibility','off');
grid(ax2,'on'); box(ax2,'on');
xlabel(ax2,'Training episode');
ylabel(ax2,sprintf('%d-episode mean compliance (%%)',window));
title(ax2,'(b) Category-compliance during training');
legend(ax2,'Location','best');
xlim(ax2,[1 max([eCNN;eDense;eUnmask])]);
ylim(ax2,[max(0,min(cUnmask20)-2) 100.5]);

title(t,'A-QSF v5 training ablation','FontWeight','bold');

drawnow;
if ~exist(cfg.outputDir,'dir'), mkdir(cfg.outputDir); end
pngFile = fullfile(cfg.outputDir,'Figure_training_ablation_v5.png');
pdfFile = fullfile(cfg.outputDir,'Figure_training_ablation_v5.pdf');
exportgraphics(f,pngFile,'Resolution',600);
figFile = fullfile(cfg.outputDir,'Figure_training_ablation_v5.fig');
exportgraphics(f,pdfFile,'ContentType','vector');
savefig(f,figFile);
fprintf('Figure 4 shown and exported:\n  %s\n  %s\n  %s\n',pngFile,pdfFile,figFile);
end

function [e,r,c] = localSeries(T,idx)
e = double(T.Episode(idx));
r = double(T.Return(idx));
c = double(T.Compliance_pct(idx));
[e,ord] = sort(e);
r = r(ord); c = c(ord);
end
