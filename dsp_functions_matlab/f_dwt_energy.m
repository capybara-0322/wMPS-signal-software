function result = f_dwt_energy_new(dwt_result, options)
% F_DWT_ENERGY  对 DWT 分解结构体的各层系数计算能量
%
% 用法:
%   result = f_dwt_energy(dwt_result)
%   result = f_dwt_energy(dwt_result, 'Mode', 'peak_window', 'HalfWin', 500)
%   result = f_dwt_energy(dwt_result, 'Mode', 'threshold',   'ThreshN', 3, 'Plot', true)
%   result = f_dwt_energy(dwt_result, 'Mode', 'time_window', 'WinTime', 200e-6)
%
% ── 输入 ─────────────────────────────────────────────────────────────────
%   dwt_result     DWT 分解结构体，须包含以下字段:
%                    .wavelet      小波名
%                    .level        分解层次 L
%                    .fs           原始采样率 (Hz)
%                    .cA{k}        第 k 层近似系数
%                    .cD{k}        第 k 层细节系数
%                    .fs_k(k)      第 k 层等效采样率 = fs/2^k (Hz)
%                    .t_k{k}       第 k 层时间轴 (s)
%                    .freq_band{k} 第 k 层细节频带 [f_lo, f_hi] (Hz)
%
% ── 可选具名参数 ──────────────────────────────────────────────────────────
%   'Mode'        能量采集方式 (string)                    默认 'full'
%                   'full'          全段系数，所有点平方和
%                   'peak_window'   以各层绝对值最大点为中心，± HalfWin 采样点窗口
%                   'threshold'     N 倍标准差阈值，仅对超过阈值的点直接求平方和
%                                   （不扩展窗口，HalfWin 对此模式无效）
%                   'time_window'   以各层绝对值最大点为中心，WinTime 总长时间窗口
%
%   'HalfWin'     半窗口采样点数（正整数），仅用于 'peak_window'   默认 500
%   'ThreshN'     阈值倍数 N，仅用于 'threshold'           默认 3
%   'WinTime'     窗口总时长 (s)，仅用于 'time_window'     默认 100e-6
%   'RefFs'       基准采样率 (Hz)                          默认 50e6
%   'Normalize'   是否除以窗口持续时间归一化为功率 (logical)  默认 false
%   'Plot'        是否绘图 (logical)                        默认 false

    %% ── 参数声明 ─────────────────────────────────────────────────────────
    arguments
        dwt_result        struct
        options.Mode      {mustBeMember(options.Mode, ...
                            {'full','peak_window','threshold','time_window'})} ...
                          = 'full'
        options.HalfWin   (1,1) double {mustBePositive, mustBeInteger} = 500
        options.ThreshN   (1,1) double {mustBePositive} = 3
        options.WinTime   (1,1) double {mustBePositive} = 100e-6
        options.RefFs     (1,1) double {mustBePositive} = 50e6
        options.Normalize (1,1) logical = false
        options.Plot      (1,1) logical = false
    end

    %% ── 校验输入结构体 ───────────────────────────────────────────────────
    required = {'level','fs','cA','cD','fs_k','t_k','freq_band'};
    for fi = 1:numel(required)
        if ~isfield(dwt_result, required{fi})
            error('f_dwt_energy:missingField', ...
                  '输入结构体缺少必需字段: %s', required{fi});
        end
    end

    L      = dwt_result.level;
    fs_ori = dwt_result.fs;
    mode   = options.Mode;

    %% ── 初始化输出 ───────────────────────────────────────────────────────
    result.mode                  = mode;
    result.normalize             = options.Normalize;
    result.ref_fs                = options.RefFs;
    result.fc                    = zeros(1, L);
    result.energy_cD             = zeros(1, L);
    result.win_idx_cD            = cell(1, L);
    result.win_time_cD           = cell(1, L);
    result.thresh_fallback_cD    = false(1, L);

    %% ── 各层细节系数处理 ─────────────────────────────────────────────────
    for k = 1:L
        coef  = dwt_result.cD{k}(:)';
        t_k   = dwt_result.t_k{k}(:)';
        fs_k  = dwt_result.fs_k(k);
        fband = dwt_result.freq_band{k};

        if fband(1) <= 0
            result.fc(k) = fband(2) / 2;
        else
            result.fc(k) = sqrt(fband(1) * fband(2));
        end

        [win_idx, win_time, E, fallback] = ...
            process_layer_(coef, t_k, fs_k, mode, options);

        result.win_idx_cD{k}         = win_idx;
        result.win_time_cD{k}        = win_time;
        result.energy_cD(k)          = E;
        result.thresh_fallback_cD(k) = fallback;
    end

    %% ── 近似层 cA{L} 处理 ───────────────────────────────────────────────
    coef_a = dwt_result.cA{L}(:)';
    t_a    = dwt_result.t_k{L}(:)';
    fs_a   = dwt_result.fs_k(L);

    result.fc_approx = fs_ori / 2^(L+2);

    [win_idx_a, win_time_a, E_a, fallback_a] = ...
        process_layer_(coef_a, t_a, fs_a, mode, options);

    result.win_idx_cA         = win_idx_a;
    result.win_time_cA        = win_time_a;
    result.energy_cA          = E_a;
    result.thresh_fallback_cA = fallback_a;

    %% ── 可选绘图 ─────────────────────────────────────────────────────────
    if options.Plot
        plot_energy_(dwt_result, result, L, fs_ori, options);
    end

end


%% ════════════════════════════════════════════════════════════════════════
%  内部函数：处理单层
% ════════════════════════════════════════════════════════════════════════
function [win_idx, win_time, E, fallback] = ...
        process_layer_(coef, t_k, fs_k, mode, opt)

    N        = numel(coef);
    fallback = false;

    switch mode
        case 'full'
            win_idx  = [1, N];
            win_time = [t_k(1), t_k(N)];
            E = compute_energy_(coef, fs_k, opt.Normalize);

        case 'peak_window'
            [~, ip]  = max(abs(coef));
            hpts     = max(1, round(opt.HalfWin * fs_k / opt.RefFs));
            i1 = max(1, ip - hpts);
            i2 = min(N, ip + hpts);
            win_idx  = [i1, i2];
            win_time = [t_k(i1), t_k(i2)];
            E = compute_energy_(coef(i1:i2), fs_k, opt.Normalize);

        case 'threshold'
            thresh   = opt.ThreshN * std(coef);
            over_idx = find(abs(coef) >= thresh);

            if isempty(over_idx)
                fallback = true;
                over_idx = (1:N);
                warning('f_dwt_energy:threshFallback', ...
                    '该层无超过 %.2g×std 阈值的点，退化为全段。', opt.ThreshN);
            end

            win_idx  = over_idx(:)';
            win_time = t_k(over_idx(:))';
            E = compute_energy_(coef(over_idx), fs_k, opt.Normalize);

        case 'time_window'
            [~, ip]  = max(abs(coef));
            hpts     = max(1, round((opt.WinTime / 2) * fs_k));
            i1 = max(1, ip - hpts);
            i2 = min(N, ip + hpts);
            win_idx  = [i1, i2];
            win_time = [t_k(i1), t_k(i2)];
            E = compute_energy_(coef(i1:i2), fs_k, opt.Normalize);

        otherwise
            win_idx  = [1, N];
            win_time = [t_k(1), t_k(N)];
            E = compute_energy_(coef, fs_k, opt.Normalize);
    end
end


%% ════════════════════════════════════════════════════════════════════════
%  内部函数：计算能量（或功率）
% ════════════════════════════════════════════════════════════════════════
function E = compute_energy_(seg, fs_k, normalize)
    E = sum(seg .^ 2);
    if normalize && E ~= 0
        dur = numel(seg) / fs_k;
        E   = E / dur;
    end
end


%% ════════════════════════════════════════════════════════════════════════
%  内部函数：绘图（紧凑布局版）
% ════════════════════════════════════════════════════════════════════════
function plot_energy_(dw, res, L, fs_ori, opt)

    n_rows = L + 1;   % L 层细节 + 1 层近似

    % ── 颜色定义 ─────────────────────────────────────────────────────────
    c_coef = [0.20, 0.45, 0.75];   % 蓝  - 系数波形
    c_win  = [0.95, 0.35, 0.20];   % 红  - 连续窗口高亮
    c_dot  = [0.85, 0.15, 0.55];   % 品红 - threshold 散点
    c_bar  = [0.25, 0.65, 0.40];   % 绿  - 能量柱

    % ── 字体设置 ─────────────────────────────────────────────────────────
    fn_zh  = 'SimSun';             % 宋体（汉字）
    fn_en  = 'Times New Roman';    % 英文
    fs_txt = 11;                   % 统一字号

    % ── 画布尺寸 ─────────────────────────────────────────────────────────
    fig_h = 95 * n_rows + 55;      % 每行约 95px + 顶部标题留白
    fig = figure('Name', 'DWT 各层能量分析', 'NumberTitle', 'off', ...
                 'Position', [80, 60, 960, fig_h]);

    % ── 布局参数（归一化坐标） ────────────────────────────────────────────
    % 左图占 65%，右图占 21%，间隔 4%，左右外边距各 4%/2%
    mar_l  = 0.09;   % 左外边距（为 ylabel 留空间）
    mar_r  = 0.02;   % 右外边距
    gap_lr = 0.035;  % 左右列间距
    w_left = 0.62;   % 左图宽度
    w_right= 0.22;   % 右图宽度（较窄，仅显示能量大小）

    mar_top   = 0.06;   % 顶部留给 sgtitle
    mar_bot   = 0.055;  % 底部留给 xlabel
    gap_row   = 0.008;  % 行间间距（紧凑）

    row_h = (1 - mar_top - mar_bot - gap_row*(n_rows-1)) / n_rows;

    % x 起始
    x_left  = mar_l;
    x_right = mar_l + w_left + gap_lr;

    % 预先计算各行 y 起始（从上往下）
    y_start = zeros(1, n_rows);
    for i = 1:n_rows
        y_start(i) = 1 - mar_top - i*row_h - (i-1)*gap_row;
    end

    % ── 能量范围（统一 y 轴上限） ─────────────────────────────────────────
    all_E = [res.energy_cD, res.energy_cA];
    E_max = max(all_E);
    if E_max == 0, E_max = 1; end

    % ── 绘制一行 ─────────────────────────────────────────────────────────
    function draw_row_(row_i, coef, t_us, win_idx, E_val, layer_label)

        pos_l = [x_left,  y_start(row_i), w_left,  row_h];
        pos_r = [x_right, y_start(row_i), w_right, row_h];

        % ── 左图：系数波形 ────────────────────────────────────────────────
        ax1 = axes('Position', pos_l); %#ok<LAXES>
        plot(t_us, coef, 'Color', c_coef, 'LineWidth', 0.7); hold on;
        ylo = min(coef); yhi = max(coef);
        if ylo == yhi; ylo = ylo - 1e-12; yhi = yhi + 1e-12; end
        ylim([ylo, yhi]);

        if numel(win_idx) == 2
            i1 = win_idx(1); i2 = win_idx(2);
            fill([t_us(i1) t_us(i2) t_us(i2) t_us(i1)], ...
                 [ylo       ylo       yhi       yhi     ], ...
                 c_win, 'FaceAlpha', 0.18, 'EdgeColor', c_win, 'LineWidth', 1.0);
        else
            stem(t_us(win_idx), coef(win_idx), 'Color', c_dot, ...
                 'MarkerSize', 2.5, 'LineWidth', 0.6, ...
                 'BaseValue', 0, 'ShowBaseLine', 'off');
        end

        % 纵轴标注：仅层号
        ylabel(layer_label, 'FontName', fn_en, 'FontSize', fs_txt-1);

        % 最后一行才显示 x 轴标签
        if row_i == n_rows
            xlabel(sprintf('{\\fontname{%s}时间 (μs)}', fn_zh), ...
                   'Interpreter', 'tex', 'FontSize', fs_txt, 'FontName', fn_zh);
        else
            set(ax1, 'XTickLabel', []);
        end

        set(ax1, 'FontName', fn_en, 'FontSize', fs_txt-1, ...
                 'TickDir', 'in', 'TickLength', [0.005 0.005], ...
                 'YTickLabel', [], 'Box', 'on');
        grid on; hold off;

        % ── 右图：能量柱状图 ──────────────────────────────────────────────
        ax2 = axes('Position', pos_r); %#ok<LAXES>
        bc = c_bar * (0.45 + 0.55 * E_val / E_max);
        bar(1, E_val, 0.55, 'FaceColor', bc, 'EdgeColor', 'none'); hold on;

        % 数值标注（柱顶）
        text(1, E_val + E_max*0.03, sprintf('%.3g', E_val), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', fs_txt-1, 'FontWeight', 'bold', 'FontName', fn_en);

        ylim([0, E_max * 1.25]);
        xlim([0.3, 1.7]); xticks([]);

        if opt.Normalize
            ylab_r = sprintf('{\\fontname{%s}功率 (V²/s)}', fn_zh);
        else
            ylab_r = sprintf('{\\fontname{%s}能量 (V²·pts)}', fn_zh);
        end
        % 仅第一行显示右图 ylabel，避免重复
        if row_i == 1
            ylabel(ylab_r, 'Interpreter', 'tex', ...
                   'FontSize', fs_txt-1, 'FontName', fn_zh);
        end

        % 最后一行显示 xlabel 占位（保持对齐）
        if row_i == n_rows
            xlabel(' ', 'FontSize', fs_txt);
        else
            set(ax2, 'XTickLabel', []);
        end

        set(ax2, 'FontName', fn_en, 'FontSize', fs_txt-1, ...
                 'TickDir', 'in', 'TickLength', [0.01 0.01], ...
                 'YTickLabel', [], 'YAxisLocation', 'right', 'Box', 'on');
        grid on; hold off;
    end

    % ── 各细节层 ─────────────────────────────────────────────────────────
    for k = 1:L
        coef  = dw.cD{k}(:)';
        t_us  = dw.t_k{k}(:)' * 1e6;
        layer_lbl = sprintf('cD{%d}', k);
        draw_row_(k, coef, t_us, res.win_idx_cD{k}, res.energy_cD(k), layer_lbl);
    end

    % ── 近似层 ───────────────────────────────────────────────────────────
    coef_a = dw.cA{L}(:)';
    t_a_us = dw.t_k{L}(:)' * 1e6;
    layer_lbl_a = sprintf('cA{%d}', L);
    draw_row_(L+1, coef_a, t_a_us, res.win_idx_cA, res.energy_cA, layer_lbl_a);

    % ── 总标题（顶部，单行紧凑） ─────────────────────────────────────────
    switch opt.Mode
        case 'full'
            ms = 'full';
        case 'peak_window'
            ms = sprintf('peak\\_window  HalfWin=%d (RefFs=%.3gMHz)', ...
                         round(opt.HalfWin), opt.RefFs/1e6);
        case 'threshold'
            ms = sprintf('threshold  N=%.2g\\sigma', opt.ThreshN);
        case 'time_window'
            ms = sprintf('time\\_window  %.1f\\mus', opt.WinTime*1e6);
        otherwise
            ms = opt.Mode;
    end
    ns = '';
    if opt.Normalize, ns = '  归一化'; end

    % sgtitle 混排：汉字宋体，其余 Times New Roman
    title_str = sprintf('{\\fontname{%s}DWT各层能量分析}{\\fontname{%s}  |  Mode: %s%s}', ...
        fn_zh, fn_en, ms, ns);
    sgt = sgtitle(title_str, 'Interpreter', 'tex', ...
                  'FontSize', fs_txt, 'FontWeight', 'bold');

    % * 号说明（threshold 退化提示）
    if any(res.thresh_fallback_cD) || res.thresh_fallback_cA
        annotation(fig, 'textbox', [0.01, 0.0, 0.5, 0.04], ...
            'String', sprintf('{\\fontname{%s}* 该层无超阈值点，已退化为全段}', fn_zh), ...
            'Interpreter', 'tex', 'FontSize', fs_txt-2, ...
            'EdgeColor', 'none', 'Color', [0.5 0.3 0.1]);
    end

end