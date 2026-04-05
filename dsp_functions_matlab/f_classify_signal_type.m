function label = f_classify_signal_type(sig_frag_dwt_eng, options)
% F_CLASSIFY_SIGNAL_TYPE  基于 DWT 能量特征对信号片段进行线性分类
%
% 功能：
%   将 sig_frag_dwt_eng 结构体中的各层能量向量（energy_cD × 5层 + energy_cA × 1层）
%   拼合为 6 维特征向量，与权重向量 w 做内积后加偏置 b，判断结果正负以区分
%   扫描光信号（"scan"）与同步光信号（"sync"）。
%
% 输入：
%   sig_frag_dwt_eng  — DWT 能量分析结果结构体，需包含字段：
%                         .energy_cD  (1×5 double)  各层细节系数能量
%                         .energy_cA  (1×1 double)  近似系数能量
%   options.w         — 权重向量 (1×6 double)，默认 [-5.06, 0.24, 1.72, 7.08, 6.96, -1.33]
%   options.b         — 偏置标量 (1×1 double)，默认 -0.488
%
% 输出：
%   label  — 分类结果字符串："scan" 或 "sync"
%
% 判别规则：
%   x     = [energy_cD, energy_cA]（行向量，共 6 维）
%   x_norm = x / sum(x)            （L1 归一化，使各分量之和为 1）
%   score  = w · x_norm + b
%   score > 0  →  "sync"
%   score <= 0 →  "scan"
%
% 用法示例：
%   label = f_classify_signal_type(sig_frag_dwt_eng)
%   label = f_classify_signal_type(sig_frag_dwt_eng, 'w', my_w, 'b', my_b)

    arguments
        sig_frag_dwt_eng   (1,1) struct
        options.w          (1,6) double = [-5.06, 0.24, 1.72, 7.08, 6.96, -1.33]
        options.b          (1,1) double = -0.488
    end

    %% ── 输入校验 ────────────────────────────────────────────────────────

    if ~isfield(sig_frag_dwt_eng, 'energy_cD') || ~isfield(sig_frag_dwt_eng, 'energy_cA')
        error('f_classify_signal_type:missingField', ...
              '输入结构体缺少必要字段，需包含 energy_cD 与 energy_cA。');
    end

    energy_cD = double(sig_frag_dwt_eng.energy_cD(:)');   % 确保为行向量
    energy_cA = double(sig_frag_dwt_eng.energy_cA(1));

    if numel(energy_cD) ~= 5
        error('f_classify_signal_type:wrongDim', ...
              'energy_cD 应为 5 维向量，当前为 %d 维。', numel(energy_cD));
    end

    %% ── 特征拼合与归一化 ────────────────────────────────────────────────

    x      = [energy_cD, energy_cA];   % 1×6 特征向量（拼合顺序：cD1~cD5, cA）
    x_sum  = sum(x);

    if x_sum <= 0
        error('f_classify_signal_type:zeroEnergy', ...
              '各层能量之和为零或负值，无法进行归一化，请检查输入结构体。');
    end

    x_norm = x / x_sum;               % L1 归一化，使各分量之和为 1

    %% ── 线性判别 ────────────────────────────────────────────────────────

    score = dot(options.w, x_norm) + options.b;

    %% ── 输出分类标签 ────────────────────────────────────────────────────

    if score > 0
        label = "sync";
    else
        label = "scan";
    end

end