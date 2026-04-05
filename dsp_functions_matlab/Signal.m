classdef Signal
% SIGNAL  封装一维采样信号及其采样率的值类
%
% 构造:
%   sig = Signal(data, fs)          % 直接构造
%   sig = Signal.fromFile(filepath) % 从 .mat 文件加载（工厂方法）
%   sig = Signal.fromFile(filepath, 'channel', 'ch2')
%
% 常用属性:
%   sig.data   — 信号序列（double 行向量，单位 V）
%   sig.fs     — 采样率（Hz）
%   sig.N      — 采样点数
%   sig.dt     — 采样间隔（s）
%   sig.duration — 总时长（s）
%
% 常用方法:
%   sig.plot()               — 绘制时域波形
%   sig.plot('MaxPoints', N) — 限制绘图点数
%   sig.psd()                — 绘制功率谱密度
%   sig.rms()                — 计算 RMS
%   sig.crop(t1, t2)         — 按时间截取，返回新 Signal
%   sig.resample(fs_new)     — 重采样，返回新 Signal

    %% ── 属性 ─────────────────────────────────────────────────────────────
    properties (SetAccess = public)
        data    (1,:) double    % 信号序列（行向量）
        fs      (1,1) double    % 采样率 Hz
    end

    properties (Dependent)
        N           % 采样点数
        dt          % 采样间隔 s
        duration    % 总时长 s
        t           % 时间轴向量 s（按需计算，不存储）
    end

    properties (SetAccess = private)
        meta        struct  = struct()  % 可选元信息（来自文件等）
    end

    %% ── 构造函数 ─────────────────────────────────────────────────────────
    methods
        function obj = Signal(data, fs, meta)
            arguments
                data (1,:) double
                fs   (1,1) double {mustBePositive}
                meta       struct = struct()
            end
            obj.data = data;
            obj.fs   = fs;
            obj.meta = meta;
        end
    end

    %% ── Dependent 属性 ───────────────────────────────────────────────────
    methods
        function n = get.N(obj)
            n = numel(obj.data);
        end
        function d = get.dt(obj)
            d = 1 / obj.fs;
        end
        function d = get.duration(obj)
            d = obj.N / obj.fs;
        end
        function t = get.t(obj)
            t = (0 : obj.N - 1) / obj.fs;
        end
    end

    %% ── 工厂方法 ─────────────────────────────────────────────────────────
    methods (Static)
        function obj = fromFile(filepath, options)
        % 从量化 .mat 文件加载，返回 Signal 对象
        %
        % 用法:
        %   sig = Signal.fromFile('0000.mat')
        %   sig = Signal.fromFile('0000.mat', 'channel', 'ch2')
            arguments
                filepath            {mustBeText}
                options.channel     {mustBeText} = 'ch1'
            end

            filepath  = char(filepath);
            ch_suffix = char(options.channel);

            if ~isfile(filepath)
                error('Signal:fileNotFound', '找不到文件: %s', filepath);
            end

            raw = load(filepath);

            % 采样率
            if ~isfield(raw, 'fs')
                error('Signal:missingFs', '文件中未找到 fs 字段');
            end
            fs = double(raw.fs(1));

            % 量化数据
            q_field = ['q_' ch_suffix];
            if ~isfield(raw, q_field)
                error('Signal:missingChannel', ...
                      '未找到通道字段 "%s"，可用: %s', ...
                      q_field, strjoin(fieldnames(raw), ', '));
            end
            q_data = double(raw.(q_field)(:)');

            % 反量化
            if ~isfield(raw, 'scale')
                error('Signal:missingScale', '文件中未找到 scale 字段');
            end
            scale = double(raw.scale(1));

            quant_mode = '';
            if isfield(raw, 'quant_mode')
                quant_mode = strtrim(char(raw.quant_mode));
            end

            switch quant_mode
                case 'int16_symmetric_v1'
                    data = q_data * scale;
                otherwise
                    if ~isempty(quant_mode)
                        warning('Signal:unknownQuantMode', ...
                                '未知量化模式 "%s"，尝试 data = q * scale', quant_mode);
                    end
                    data = q_data * scale;
            end

            % 元信息
            meta = struct('source', filepath, 'channel', ch_suffix);
            if isfield(raw, 'meta_json')
                try
                    meta.device = jsondecode(char(raw.meta_json));
                catch
                end
            end

            obj = Signal(data, fs, meta);
        end
    end

    %% ── 实例方法 ─────────────────────────────────────────────────────────
    methods

        % ── 绘制时域波形 ──────────────────────────────────────────────────
        function plot(obj, options)
            arguments
                obj
                options.MaxPoints   (1,1) double = 10000000
                options.Title       {mustBeText}  = ''
            end

            [t_plot, sig_plot, ds_note] = obj.downsample_for_plot_(options.MaxPoints);
            t_ms = t_plot * 1e3;

            v_unit = 'V';
            if isfield(obj.meta, 'device') && isfield(obj.meta.device, 'Vertical_Units')
                v_unit = obj.meta.device.Vertical_Units;
            end

            fig_name = '';
            if isfield(obj.meta, 'source')
                fig_name = obj.meta.source;
            end

            figure('Name', ['Signal: ' fig_name], 'NumberTitle', 'off');
            plot(t_ms, sig_plot, 'LineWidth', 0.8);
            xlabel('时间 (ms)');
            ylabel(['幅值 (' v_unit ')']);
            grid on; box on;

            if ~isempty(options.Title)
                ttl = options.Title;
            else
                ch = '';
                if isfield(obj.meta, 'channel')
                    ch = ['  通道 ' upper(obj.meta.channel)];
                end
                ttl = sprintf('fs = %.3g MHz  N = %d%s%s', ...
                              obj.fs/1e6, obj.N, ch, ds_note);
            end
            title(ttl, 'Interpreter', 'none');

            dcm = datacursormode(gcf);
            dcm.UpdateFcn = @(~, evt) Signal.datatip_(evt, obj.fs);
        end

        % ── 绘制功率谱密度 ────────────────────────────────────────────────
        function psd(obj)
            figure('NumberTitle', 'off', 'Name', 'PSD');
            pwelch(obj.data, [], [], [], obj.fs);
            title(sprintf('功率谱密度  fs = %.3g MHz', obj.fs/1e6));
        end

        % ── RMS ───────────────────────────────────────────────────────────
        function val = rms(obj)
            val = sqrt(mean(obj.data .^ 2));
        end

        % ── 按时间截取，返回新 Signal ──────────────────────────────────────
        function out = crop(obj, t1, t2)
        % crop(t1, t2)  截取 [t1, t2] 秒范围内的信号
            arguments
                obj
                t1 (1,1) double {mustBeNonnegative}
                t2 (1,1) double
            end
            if t2 <= t1
                error('Signal:crop', 't2 必须大于 t1');
            end
            i1 = max(1,   round(t1 * obj.fs) + 1);
            i2 = min(obj.N, round(t2 * obj.fs) + 1);
            out = Signal(obj.data(i1:i2), obj.fs, obj.meta);
        end

        % ── 重采样，返回新 Signal ─────────────────────────────────────────
        function out = resample(obj, fs_new)
            arguments
                obj
                fs_new (1,1) double {mustBePositive}
            end
            [p, q] = rat(fs_new / obj.fs, 1e-6);
            new_data = resample(obj.data, p, q);
            out = Signal(new_data, fs_new, obj.meta);
        end

        % ── disp：在命令行显示摘要 ────────────────────────────────────────
        function disp(obj)
            fprintf('  Signal\n');
            fprintf('    fs       : %.6g Hz  (%.3g MHz)\n', obj.fs, obj.fs/1e6);
            fprintf('    N        : %d 点\n',  obj.N);
            fprintf('    duration : %.4g s\n', obj.duration);
            fprintf('    RMS      : %.4g V\n', obj.rms());
            if isfield(obj.meta, 'source')
                fprintf('    source   : %s\n', obj.meta.source);
            end
        end

    end

    %% ── 私有辅助方法 ─────────────────────────────────────────────────────
    methods (Access = private)
        function [t_out, sig_out, note] = downsample_for_plot_(obj, max_pts)
            t = obj.t;
            if obj.N > max_pts
                step    = floor(obj.N / max_pts);
                idx     = 1 : step : obj.N;
                t_out   = t(idx);
                sig_out = obj.data(idx);
                note    = sprintf('（显示 %d/%d 点）', numel(idx), obj.N);
            else
                t_out   = t;
                sig_out = obj.data;
                note    = '';
            end
        end
    end

    methods (Static, Access = private)
        function txt = datatip_(evt, fs)
            pos  = evt.Position;
            t_ms = pos(1);  t_s = t_ms / 1e3;  volt = pos(2);

            if abs(t_s) >= 1
                t_str = sprintf('%.6g s',  t_s);
            elseif abs(t_s) >= 1e-3
                t_str = sprintf('%.8g ms', t_ms);
            else
                t_str = sprintf('%.6g us', t_ms * 1e3);
            end

            if abs(volt) >= 1
                v_str = sprintf('%.4g V',  volt);
            elseif abs(volt) >= 1e-3
                v_str = sprintf('%.4g mV', volt * 1e3);
            else
                v_str = sprintf('%.4g uV', volt * 1e6);
            end

            txt = {['t = ' t_str], ['V = ' v_str], ...
                   ['Sample #' num2str(round(t_s * fs) + 1)]};
        end
    end

end
