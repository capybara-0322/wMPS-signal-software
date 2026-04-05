function segments = f_detect_pulses(sig, options)
% F_DETECT_PULSES  基于阈值的峰值检测，返回包含脉冲片段的 Signal 对象数组
%
% 输入（必须）:
%   sig             — Signal 对象（用于裁剪输出片段）
%
% 输入（可选具名参数）:
%   'RefSig'        参考 Signal 对象，默认为空（即使用 sig 自身做检测）
%                   若传入，则阈值检测在 RefSig 上执行，裁剪位置对应到 sig 输出
%                   要求：RefSig 与 sig 采样率相同，长度相同
%   'Method'        阈值计算方法，字符串，默认 'absolute'
%                     'absolute'   — 直接使用 Threshold 指定的绝对值
%                     'mean_std'   — 均值 + N × 标准差（N 由 Threshold 指定，默认 3）
%                     'peak_ratio' — 峰值 × Threshold 百分比（0~1，默认 0.5）
%                     'median_mad' — 中位数 + N × MAD（N 由 Threshold 指定，默认 3）
%   'Threshold'     阈值参数（含义随 Method 变化，见上），默认 0.5
%   'MinWidth'      脉冲最小宽度 (μs)，默认 1.0
%                   宽度不足的脉冲被丢弃；相邻脉冲间距小于此值时合并为一段
%   'Margin'        左右延申量 (μs)，默认 5.0（超出信号边界时截断）
%   'Plot'          是否绘图 (logical)，默认 false
%   'Clilog'        是否在控制台输出日志
%
% 输出:
%   segments        — Signal 对象数组（1×K），每个元素为一段含脉冲的片段信号
%                   （片段数据来自 sig，位置由 RefSig 或 sig 的检测结果决定）
%
% 示例:
%   segs = f_detect_pulses(sig)
%   segs = f_detect_pulses(sig, 'Method', 'mean_std', 'Threshold', 5, 'Plot', true)
%   segs = f_detect_pulses(sig, 'Method', 'peak_ratio', 'Threshold', 0.3, 'MinWidth', 2)
%   segs = f_detect_pulses(sig, 'RefSig', ref_sig, 'Threshold', 0.5, 'Plot', true)

    %% ── 参数声明 ─────────────────────────────────────────────────────────
    arguments
        sig                 Signal
        options.RefSig                    = []   % Signal 或空，用于检测的参考信号
        options.Method      {mustBeText, mustBeMember(options.Method, ...
                             {'absolute','mean_std','peak_ratio','median_mad'})} ...
                            = 'absolute'
        options.Threshold   (1,1) double = 0.5
        options.MinWidth    (1,1) double {mustBePositive} = 1.0     % μs
        options.Margin      (1,1) double {mustBeNonnegative} = 5.0  % μs
        options.Plot        (1,1) logical = false
        options.Clilog      (1,1) logical = false

    end

    %% ── 确定检测信号 & 输出信号 ─────────────────────────────────────────
    % det_sig：用于阈值检测的信号；sig：用于输出片段的信号
    if isempty(options.RefSig)
        det_sig = sig;
    else
        if ~isa(options.RefSig, 'Signal')
            error('f_detect_pulses:badRefSig', 'RefSig 必须是 Signal 对象。');
        end
        if options.RefSig.fs ~= sig.fs
            error('f_detect_pulses:fsMismatch', ...
                'RefSig 采样率 (%.6g Hz) 与 sig 采样率 (%.6g Hz) 不一致。', ...
                options.RefSig.fs, sig.fs);
        end
        if options.RefSig.N ~= sig.N
            error('f_detect_pulses:lengthMismatch', ...
                'RefSig 长度 (%d) 与 sig 长度 (%d) 不一致。', ...
                options.RefSig.N, sig.N);
        end
        det_sig = options.RefSig;
    end
    use_ref = ~isempty(options.RefSig);

    %% ── 计算阈值 ─────────────────────────────────────────────────────────
    x   = det_sig.data;   % 检测用数据
    x_out = sig.data;     % 输出用数据
    fs  = sig.fs;
    N   = sig.N;

    switch options.Method
        case 'absolute'
            thr = options.Threshold;

        case 'mean_std'
            thr = mean(x) + options.Threshold * std(x);

        case 'peak_ratio'
            if options.Threshold <= 0 || options.Threshold >= 1
                error('f_detect_pulses:badThreshold', ...
                    'peak_ratio 方法要求 Threshold ∈ (0, 1)，当前值 = %g', ...
                    options.Threshold);
            end
            thr = max(abs(x)) * options.Threshold;

        case 'median_mad'
            med = median(x);
            mad_val = median(abs(x - med));
            thr = med + options.Threshold * mad_val;
    end

    %% ── 采样点数换算 ─────────────────────────────────────────────────────
    min_samples  = max(1, round(options.MinWidth  * 1e-6 * fs));
    margin_samps = round(options.Margin * 1e-6 * fs);

    %% ── 阈值过零检测 ─────────────────────────────────────────────────────
    above = x >= thr;                           % 超过阈值的掩码

    % 找到上升沿和下降沿
    d_above = diff([0, above, 0]);
    rise_idx = find(d_above ==  1);             % 超阈的起始点
    fall_idx = find(d_above == -1) - 1;         % 超阈的结束点

    if isempty(rise_idx)
        warning('f_detect_pulses:noPulse', ...
            '未检测到超过阈值（%.4g）的脉冲。', thr);
        segments = Signal.empty(1, 0);
        return;
    end

    %% ── 丢弃宽度不足的脉冲 ───────────────────────────────────────────────
    widths  = fall_idx - rise_idx + 1;
    valid   = widths >= min_samples;
    rise_idx = rise_idx(valid);
    fall_idx = fall_idx(valid);

    if isempty(rise_idx)
        warning('f_detect_pulses:noPulse', ...
            '所有脉冲宽度均小于 MinWidth = %.2f μs，无有效脉冲。', options.MinWidth);
        segments = Signal.empty(1, 0);
        return;
    end

    %% ── 合并相邻脉冲（间距 < min_samples）────────────────────────────────
    k = 1;
    while k < numel(rise_idx)
        gap = rise_idx(k+1) - fall_idx(k) - 1;
        if gap < min_samples
            fall_idx(k) = fall_idx(k+1);
            rise_idx(k+1) = [];
            fall_idx(k+1) = [];
        else
            k = k + 1;
        end
    end

    %% ── 加左右延申，截取片段 ──────────────────────────────────────────────
    n_pulses = numel(rise_idx);
    segments = Signal.empty(1, 0);

    for k = 1 : n_pulses
        i1 = max(1, rise_idx(k) - margin_samps);
        i2 = min(N, fall_idx(k) + margin_samps);
        seg_data = x_out(i1 : i2);

        % 继承 meta，附加脉冲信息
        seg_meta = sig.meta;
        seg_meta.pulse_index   = k;
        seg_meta.pulse_count   = n_pulses;
        seg_meta.threshold     = thr;
        seg_meta.method        = options.Method;
        seg_meta.t_start_s     = (i1 - 1) / fs;
        seg_meta.t_end_s       = (i2 - 1) / fs;
        seg_meta.source        = sprintf('pulse_%d_of_%d', k, n_pulses);
        if use_ref
            if isfield(det_sig.meta, 'source')
                seg_meta.ref_source = det_sig.meta.source;
            else
                seg_meta.ref_source = 'RefSig';
            end
        end

        segments(end+1) = Signal(seg_data, fs, seg_meta); %#ok<AGROW>
    end

    %% ── 可选：绘图 ───────────────────────────────────────────────────────
    if options.Plot
        t_ms = sig.t * 1e3;

        figure('Name', '脉冲检测结果', 'NumberTitle', 'off', ...
               'Position', [100 100 1000 500]);

        % 绘制输出信号（主信号）
        plot(t_ms, x_out, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8, ...
             'DisplayName', '输出信号 (sig)');
        hold on;

        % 若使用参考信号，额外叠画
        if use_ref
            plot(t_ms, x, 'Color', [0.2 0.5 0.9], 'LineWidth', 0.8, ...
                 'DisplayName', '检测信号 (RefSig)');
        end

        yline(thr, 'r--', sprintf('阈值 = %.4g', thr), ...
              'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');

        cmap = lines(n_pulses);
        for k = 1 : n_pulses
            i1 = max(1, rise_idx(k) - margin_samps);
            i2 = min(N, fall_idx(k) + margin_samps);
            t1 = (i1 - 1) / fs * 1e3;
            t2 = (i2 - 1) / fs * 1e3;
            yl = ylim;
            patch([t1 t2 t2 t1], [yl(1) yl(1) yl(2) yl(2)], ...
                  cmap(k,:), 'FaceAlpha', 0.18, 'EdgeColor', 'none');
            text((t1+t2)/2, yl(2)*0.95, sprintf('#%d', k), ...
                 'HorizontalAlignment', 'center', 'FontSize', 8, ...
                 'Color', cmap(k,:));
        end

        xlabel('时间 (ms)');
        ylabel('幅值 (V)');
        if use_ref
            title(sprintf('脉冲检测  |  方法: %s  |  阈值: %.4g  |  检测到 %d 个脉冲  [参考信号检测]', ...
                  options.Method, thr, n_pulses), 'Interpreter', 'none');
        else
            title(sprintf('脉冲检测  |  方法: %s  |  阈值: %.4g  |  检测到 %d 个脉冲', ...
                  options.Method, thr, n_pulses), 'Interpreter', 'none');
        end
        legend('Location', 'northeast');
        grid on; box on;

        % 各脉冲片段独立子图（最多显示 8 个）
        n_show = min(n_pulses, 8);
        if n_show > 1
            figure('Name', '脉冲片段', 'NumberTitle', 'off', ...
                   'Position', [120 80 1100 180*ceil(n_show/2)]);
            for k = 1 : n_show
                subplot(ceil(n_show/2), 2, k);
                t_seg = segments(k).t * 1e6;   % 相对时间 μs
                plot(t_seg, segments(k).data, 'Color', cmap(k,:), 'LineWidth', 1);
                xlabel('相对时间 (μs)');  ylabel('V');
                title(sprintf('脉冲 #%d  (%.3f ms)', k, ...
                      segments(k).meta.t_start_s * 1e3), 'Interpreter', 'none');
                grid on;
            end
            if n_pulses > 8
                sgtitle(sprintf('前 8 / 共 %d 个脉冲片段', n_pulses));
            else
                sgtitle('各脉冲片段');
            end
        end
    end

    %% ── 控制台摘要 ───────────────────────────────────────────────────────
    if options.Clilog
        if use_ref
            fprintf('[f_detect_pulses] 方法: %-12s  阈值: %.4g  检测到: %d 个脉冲  （参考信号检测，输出来自 sig）\n', ...
                    options.Method, thr, n_pulses);
        else
            fprintf('[f_detect_pulses] 方法: %-12s  阈值: %.4g  检测到: %d 个脉冲\n', ...
                    options.Method, thr, n_pulses);
        end
        for k = 1 : n_pulses
            fprintf('  脉冲 #%d: t = [%.4f, %.4f] ms  宽度 = %.2f μs\n', ...
                    k, ...
                    segments(k).meta.t_start_s * 1e3, ...
                    segments(k).meta.t_end_s   * 1e3, ...
                    (fall_idx(k) - rise_idx(k) + 1) / fs * 1e6);
        end
    end
end