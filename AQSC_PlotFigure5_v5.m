function f = AQSC_PlotFigure5_v5(csvPath,cfg)
% AQSC_PlotFigure5_v5
% Reproduce manuscript Figure 5 directly from the measured hardware CSV.
% Bars = median single-thread latency; black diamond markers = p95.
% Y axis is logarithmic, as stated in the manuscript caption.
% The figure remains OPEN/VISIBLE after export.

arguments
    csvPath {mustBeFile}
    cfg struct
end

T = readtable(csvPath,'TextType','string');
required = {'algorithm','operation','median_ms','p95_ms'};
assert(all(ismember(required,T.Properties.VariableNames)), ...
    'Benchmark CSV must contain: %s',strjoin(required,', '));

alg = string(T.algorithm); op = string(T.operation);
wantAlg = ["ML-KEM-512","ML-KEM-768","ML-KEM-1024", ...
           "SLH-DSA-SHA2-128f","SLH-DSA-SHA2-192f","SLH-DSA-SHA2-256f"];
wantOp  = ["encapsulate","encapsulate","encapsulate","sign","sign","sign"];
labels  = {'ML-KEM-512','ML-KEM-768','ML-KEM-1024', ...
           'SLH-128f','SLH-192f','SLH-256f'};

med = zeros(1,numel(wantAlg)); p95 = med;
for i=1:numel(wantAlg)
    idx = alg==wantAlg(i) & op==wantOp(i);
    assert(nnz(idx)==1,'Missing/duplicate benchmark row for %s / %s',wantAlg(i),wantOp(i));
    med(i) = localNumeric(T.median_ms(idx));
    p95(i) = localNumeric(T.p95_ms(idx));
end

f = figure('Name','Figure 5 - PQC hardware benchmark', ...
    'NumberTitle','off','Color','w','Visible','on', ...
    'Position',[100 100 1050 560]);
ax = axes(f); hold(ax,'on');

b = bar(ax,1:6,med,0.72,'FaceColor','flat','DisplayName','Median');
% Visually separate KEM and signature groups while staying MATLAB-default
% compatible. Explicit colors are used only to encode the two operation
% families in this publication figure.
b.CData(1:3,:) = repmat([0.0000 0.4470 0.7410],3,1);
b.CData(4:6,:) = repmat([0.8500 0.3250 0.0980],3,1);

plot(ax,1:6,p95,'kd','MarkerFaceColor','k','MarkerSize',6, ...
    'LineStyle','none','DisplayName','p95');
set(ax,'YScale','log','XTick',1:6,'XTickLabel',labels,'FontSize',10);
xtickangle(ax,18);
xlabel(ax,'PQC primitive / parameter set');
ylabel(ax,'Single-thread latency (ms, log scale)');
title(ax,'Measured PQC operation latency on AMD EPYC 9V74');
grid(ax,'on'); box(ax,'on');
legend(ax,'Location','northwest');

% Annotate medians to make the exact measured values visible in MATLAB.
for i=1:6
    text(ax,i,med(i),sprintf('  %.3f',med(i)), ...
        'VerticalAlignment','bottom','HorizontalAlignment','left', ...
        'FontSize',8,'Rotation',90);
end

% Expand bounds so the p95 markers and annotations are never clipped.
ymin = min(med)*0.65; ymax = max(p95)*1.55;
ylim(ax,[ymin ymax]); xlim(ax,[0.3 6.7]);
drawnow;

if ~exist(cfg.outputDir,'dir'), mkdir(cfg.outputDir); end
pngFile = fullfile(cfg.outputDir,'Figure_hardware_benchmark_v5.png');
pdfFile = fullfile(cfg.outputDir,'Figure_hardware_benchmark_v5.pdf');
exportgraphics(f,pngFile,'Resolution',600);
figFile = fullfile(cfg.outputDir,'Figure_hardware_benchmark_v5.fig');
exportgraphics(f,pdfFile,'ContentType','vector');
savefig(f,figFile);
fprintf('Figure 5 shown and exported:\n  %s\n  %s\n  %s\n',pngFile,pdfFile,figFile);
end

function x = localNumeric(v)
if iscell(v), v=v{1}; end
if isstring(v) || ischar(v), x=str2double(v); else, x=double(v); end
assert(isfinite(x) && x>0,'Benchmark value must be positive and finite.');
end
