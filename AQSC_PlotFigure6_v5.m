function f = AQSC_PlotFigure6_v5(summary,cfg)
% AQSC_PlotFigure6_v5
% Reproduce manuscript Figure 6 from the paired system-level summary.
% Four panels show mean latency, p95 latency, security compliance, and
% MEC cryptographic utilization across 500-10,000 broadcast receivers.
% Points are the across-seed means; error bars are 95% normal intervals.
% The figure remains OPEN/VISIBLE after export.
%
% Manuscript caption:
% "Paired 3GPP-aligned system-level results across 500-10,000 receivers
%  (mean and 95% normal intervals)."

arguments
    summary table
    cfg struct
end

required = {'UE','Method','MeanLatency_ms','P95Latency_ms', ...
            'SecurityCompliance_pct','MeanCryptoUtil_pct'};
assert(all(ismember(required,summary.Properties.VariableNames)), ...
    'summary must contain: %s',strjoin(required,', '));

method = string(summary.Method);
methods = ["CNN-DQN+Mask","DQN-only+Mask","Heuristic+Mask","Static-C5-long"];
for i=1:numel(methods)
    assert(any(method==methods(i)),'Missing method in summary: %s',methods(i));
end

% Prefer the UE grid declared by the reproducible configuration. If a
% summary imported from CSV contains a subset, use the common available grid.
ueAvailable = unique(double(summary.UE(:)))';
ueGrid = double(cfg.ueGrid(:))';
ueGrid = ueGrid(ismember(ueGrid,ueAvailable));
assert(~isempty(ueGrid),'No cfg.ueGrid values are present in summary.');

metrics = {'MeanLatency_ms','P95Latency_ms', ...
           'SecurityCompliance_pct','MeanCryptoUtil_pct'};
ylabels = {'Mean latency (ms)','P95 latency (ms)', ...
           'Security compliance (%)','MEC crypto utilization (%)'};
panelTitles = {'(a) Mean latency','(b) P95 latency', ...
               '(c) Security compliance','(d) MEC crypto utilization'};

f = figure('Name','Figure 6 - 3GPP-aligned system-level validation', ...
    'NumberTitle','off','Color','w','Visible','on', ...
    'Position',[70 60 1250 820]);
t = tiledlayout(f,2,2,'TileSpacing','compact','Padding','compact');

for j=1:4
    ax = nexttile(t,j); hold(ax,'on');
    for m=1:numel(methods)
        mu = nan(size(ueGrid)); ci = nan(size(ueGrid));
        for u=1:numel(ueGrid)
            idx = double(summary.UE)==ueGrid(u) & method==methods(m);
            x = double(summary.(metrics{j})(idx));
            x = x(isfinite(x));
            assert(~isempty(x),'No observations for %s at UE=%g',methods(m),ueGrid(u));
            mu(u) = mean(x);
            % Manuscript Figure 6 states 95% normal intervals.
            if numel(x)>1
                ci(u) = 1.96*std(x,0)/sqrt(numel(x));
            else
                ci(u) = 0;
            end
        end
        errorbar(ax,ueGrid,mu,ci,'-o','LineWidth',1.35, ...
            'MarkerSize',5,'DisplayName',methods(m));
    end
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'Broadcast receivers');
    ylabel(ax,ylabels{j});
    title(ax,panelTitles{j});
    xlim(ax,[max(0,min(ueGrid)-0.04*rangeSafe(ueGrid)) ...
             max(ueGrid)+0.04*rangeSafe(ueGrid)]);

    % Keep the compliance panel visually informative around the 100% line.
    if j==3
        ylim(ax,[99 101]);
        yline(ax,100,'--','100%','HandleVisibility','off');
    end

    if j==1 || j==4
        legend(ax,'Location','best');
    end
end

title(t,'A-QSF v5: paired 3GPP-aligned system-level evaluation', ...
    'FontWeight','bold');
drawnow;

if ~exist(cfg.outputDir,'dir'), mkdir(cfg.outputDir); end
pngFile = fullfile(cfg.outputDir,'Figure_system_level_v5.png');
pdfFile = fullfile(cfg.outputDir,'Figure_system_level_v5.pdf');
figFile = fullfile(cfg.outputDir,'Figure_system_level_v5.fig');
exportgraphics(f,pngFile,'Resolution',600);
exportgraphics(f,pdfFile,'ContentType','vector');
savefig(f,figFile);
fprintf('Figure 6 shown and exported:\n  %s\n  %s\n  %s\n',pngFile,pdfFile,figFile);
end

function r = rangeSafe(x)
r = max(x)-min(x);
if r<=0, r=max(1,abs(x(1))); end
end
