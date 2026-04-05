function out = f_moving_energy(sig, win_sec, options)
% F_MOVING_ENERGY  对 Signal 对象计算逐点移动窗口能量
%
% 返回值:
%   out  — Signal 对象，封装移动窗口能量序列，采样率与输入相同
%
% 用法:
%   out = f_moving_energy(sig, 0.001)
%   out = f_moving_energy(sig, 0.5e-3, 'Method', 'rms')
%   out = f_moving_energy(sig, 1e-3,   'Method', 'sum', 'Plot', true)
%
% ── 参数 ──────────────────────────────────────────────────────────────────
%   sig       Signal 对象（输入信号）
%   win_sec   窗口长度，单位秒（正标量）
%
% ── 具名参数 ──────────────────────────────────────────────────────────────
%   'Method'  能量计算方式（不区分大小写）：
%               'mean'  — mean(x²)，归一化功率（默认）
%               'rms'   — sqrt(mean(x²))，均方根，量纲与原信号相同
%               'sum'   — sum(x²)，窗内总能量，随窗长变化
%   'Plot'    是否绘图 (logical)，默认 false
%               绘图时在同一窗口中以双 y 轴展示原始信号与能量曲线
%
% ── 实现说明 ──────────────────────────────────────────────────────────────
%   逐点滑动，输出长度与输入完全相同。
%   边界处理：对称零填充（两端各填充 floor(win_pts/2) 个零），
%   保证时间轴对齐，窗口中心对应当前样本。
%   利用累积和（cumsum）实现 O(N) 快速滑动均值，避免逐点循环。

    %% ── 参数声明 ──────────────────────────────────────────────────────────
    arguments
        sig     (1,1) Signal
        win_sec (1,1) double {mustBePositive}
        options.Method  {mustBeMember(options.Method, ...
                         {'mean','rms','sum','Mean','Rms','Sum', ...
                          'MEAN','RMS','SUM'})} = 'mean'
        options.Plot    (1,1) logical = false
    end

    method = lower(options.Method);

    %% ── 窗口长度换算为样本点数 ────────────────────────────────────────────
    win_pts = round(win_sec * sig.fs);

    if win_pts < 1
        error('f_moving_energy:winTooShort', ...
            '窗口时长 %.4g s 对应 %.2f 个采样点（< 1），请增大 win_sec 或提高 fs。', ...
            win_sec, win_sec * sig.fs);
    end

    if win_pts > sig.N
        warning('f_moving_energy:winLargerThanSignal', ...
            '窗口（%d 点）大于信号长度（%d 点），已截断为信号长度。', ...
            win_pts, sig.N);
        win_pts = sig.N;
    end

    %% ── 计算移动窗口能量 ──────────────────────────────────────────────────
    % 先对 x² 序列做对称零填充，再用 cumsum 滑动求和，最后换算
    x2 = sig.data .^ 2;                          % 逐点平方

    half = floor(win_pts / 2);
    x2_pad = [zeros(1, half), x2, zeros(1, half)];  % 左右各补 half 个零

    % 累积和快速滑动求和（O(N)）
    cs      = [0, cumsum(x2_pad)];
    win_sum = cs(win_pts+1 : win_pts+sig.N) - cs(1 : sig.N);  % 长度 = N

    switch method
        case 'mean'
            energy = win_sum / win_pts;
        case 'rms'
            energy = sqrt(win_sum / win_pts);
        case 'sum'
            energy = win_sum;
    end

    %% ── 确定输出信号单位标注（写入 meta） ────────────────────────────────
    switch method
        case 'mean',  method_label = 'mean(x²)';   unit_label = 'V²';
        case 'rms',   method_label = 'RMS';         unit_label = 'V';
        case 'sum',   method_label = 'sum(x²)';     unit_label = 'V²·pts';
    end

    %% ── 封装为 Signal 对象 ────────────────────────────────────────────────
    meta_out = struct( ...
        'source',        'f_moving_energy', ...
        'parent_source', '', ...
        'method',        method_label, ...
        'unit',          unit_label, ...
        'win_sec',       win_sec, ...
        'win_pts',       win_pts ...
    );
    if isfield(sig.meta, 'source')
        meta_out.parent_source = sig.meta.source;
    end

    out = Signal(energy, sig.fs, meta_out);

    %% ── 可选：绘图 ────────────────────────────────────────────────────────
    if options.Plot
        t_ms = sig.t * 1e3;   % 统一用毫秒

        fig = figure('Name', '移动窗口能量', 'NumberTitle', 'off', ...
                     'Position', [100, 100, 1100, 480]);

        % ── 左轴：原始信号 ────────────────────────────────────────────────
        yyaxis left;
        plot(t_ms, sig.data, 'Color', [0.25, 0.55, 0.85], ...
             'LineWidth', 0.8, 'DisplayName', '原始信号');
        ylabel('幅值 (V)');

        % ── 右轴：移动窗口能量 ────────────────────────────────────────────
        yyaxis right;
        plot(t_ms, energy, 'Color', [0.90, 0.35, 0.20], ...
             'LineWidth', 1.4, 'DisplayName', ['移动窗口 ' method_label]);
        ylabel([method_label ' (' unit_label ')']);

        xlabel('时间 (ms)');
        grid on; box on;

        % ── 标题 ──────────────────────────────────────────────────────────
        src = '';
        if isfield(sig.meta, 'source')
            src = ['  |  ' sig.meta.source];
        end
        title(sprintf('移动窗口能量  |  方法: %s  |  窗口: %.4g ms  (%d pts)%s', ...
            method_label, win_sec*1e3, win_pts, src), ...
            'Interpreter', 'none');

        % ── 图例 ──────────────────────────────────────────────────────────
        % 两个 yyaxis 各有独立句柄，需手动合并到同一图例
        ax = gca;
        lines = findobj(ax, 'Type', 'Line');
        legend(flip(lines), {'原始信号', ['移动窗口 ' method_label]}, ...
               'Location', 'best');
    end

end
