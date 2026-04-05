function out = f_remove_dc(sig, options)
% F_REMOVE_DC  去除 Signal 对象中的直流分量（减去全段均值）
%
% 返回值:
%   out  — Signal 对象，直流已去除，采样率与输入相同
%
% 用法:
%   out = f_remove_dc(sig)
%   out = f_remove_dc(sig, 'Plot', true)
%
% ── 参数 ──────────────────────────────────────────────────────────────────
%   sig       Signal 对象（输入信号，必填）
%
% ── 具名参数 ──────────────────────────────────────────────────────────────
%   'Plot'    是否绘图 (logical)，默认 false
%               绘图时在同一窗口中以双 y 轴展示原始信号与去直流后的信号

    %% ── 参数声明 ──────────────────────────────────────────────────────────
    arguments
        sig          (1,1) Signal
        options.Plot (1,1) logical = false
    end

    %% ── 去除直流 ──────────────────────────────────────────────────────────
    dc    = mean(sig.data);
    data_out = sig.data - dc;

    %% ── 封装为 Signal 对象 ────────────────────────────────────────────────
    meta_out = sig.meta;
    meta_out.dc_removed    = dc;
    meta_out.parent_source = '';
    if isfield(sig.meta, 'source')
        meta_out.parent_source = sig.meta.source;
    end
    meta_out.source = 'f_remove_dc';

    out = Signal(data_out, sig.fs, meta_out);

    fprintf('  [f_remove_dc] 直流分量: %.6g V  已去除\n', dc);

    %% ── 可选：绘图 ────────────────────────────────────────────────────────
    if options.Plot
        t_ms = sig.t * 1e3;

        src = '';
        if isfield(sig.meta, 'source')
            src = ['  |  ' sig.meta.source];
        end

        figure('Name', '去除直流', 'NumberTitle', 'off', ...
               'Position', [100, 100, 1100, 480]);

        yyaxis left;
        plot(t_ms, sig.data, 'Color', [0.25, 0.55, 0.85], ...
             'LineWidth', 0.8, 'DisplayName', '原始信号');
        yline(dc, '--', 'Color', [0.15, 0.35, 0.75], 'LineWidth', 1.0, ...
              'Label', sprintf('DC = %.4g V', dc), ...
              'LabelHorizontalAlignment', 'left');
        ylabel('幅值 (V)');

        yyaxis right;
        plot(t_ms, data_out, 'Color', [0.90, 0.35, 0.20], ...
             'LineWidth', 0.8, 'DisplayName', '去直流后');
        yline(0, '--', 'Color', [0.70, 0.20, 0.10], 'LineWidth', 0.8);
        ylabel('幅值 (V)');

        xlabel('时间 (ms)');
        grid on; box on;
        title(sprintf('去除直流  |  DC = %.6g V%s', dc, src), ...
              'Interpreter', 'none');

        ax = gca;
        lines = findobj(ax, 'Type', 'Line');
        legend(flip(lines(end-1:end)), {'原始信号', '去直流后'}, ...
               'Location', 'best');
    end

end
