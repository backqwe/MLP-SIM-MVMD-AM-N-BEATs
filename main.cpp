#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <chrono>

struct StandardizationParams {
    torch::Tensor mean;
    torch::Tensor std;
};

std::tuple<torch::Tensor, torch::Tensor, StandardizationParams, StandardizationParams> standardize(
    torch::Tensor data, torch::Tensor target) {
    auto data_mean = data.mean(/*dim=*/0, /*keepdim=*/true);
    auto data_std = data.std(/*dim=*/0, /*keepdim=*/true);
    data_std = torch::clamp(data_std, 1e-4, std::numeric_limits<float>::max());
    data = (data - data_mean) / data_std;

    auto target_mean = target.mean(/*dim=*/0, /*keepdim=*/true);
    auto target_std = target.std(/*dim=*/0, /*keepdim=*/true);
    target_std = torch::clamp(target_std, 1e-4, std::numeric_limits<float>::max());
    target = (target - target_mean) / target_std;

    StandardizationParams data_params{ data_mean, data_std };
    StandardizationParams target_params{ target_mean, target_std };
    return { data, target, data_params, target_params };
}

torch::Tensor destandardize(torch::Tensor standardized, const StandardizationParams& params) {
    return standardized * params.std + params.mean;
}

torch::Device get_device() {
    return torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
}

// MVMD分解函数
std::tuple<torch::Tensor, torch::Tensor> mvmd(
    const std::vector<std::vector<float>>& features,
    const std::vector<float>& targets,
    int num_modes,
    int window_size,
    int max_iter,
    float alpha,
    float tol,
    torch::Device device) {

    int num_samples = features.size();
    int feature_dim = features[0].size();

    torch::Tensor features_tensor = torch::zeros({ feature_dim, num_samples }, torch::kFloat32);
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j)
            features_tensor[j][i] = features[i][j];
    }
    features_tensor = features_tensor.to(device);

    std::vector<torch::Tensor> modes(num_modes);
    std::vector<torch::Tensor> omegas(num_modes);
    for (int k = 0; k < num_modes; ++k) {
        modes[k] = torch::zeros({ feature_dim, num_samples }, torch::dtype(torch::kFloat32).device(device));
        omegas[k] = torch::tensor((k + 1) * 0.1 / (num_modes + 1), torch::dtype(torch::kFloat32).device(device));
    }

    torch::Tensor freqs, signal_fft, lambda;
    try {
        freqs = torch::fft::fftfreq(num_samples, 1.0).to(device);
        signal_fft = torch::fft::fft(features_tensor, /*dim=*/1);
        lambda = torch::zeros({ feature_dim, num_samples }, torch::dtype(torch::kComplexFloat).device(device));
    }
    catch (const std::exception& e) {
        freqs = torch::linspace(-0.5, 5, num_samples).to(device);
        signal_fft = torch::complex(features_tensor, torch::zeros_like(features_tensor));
        lambda = torch::zeros({ feature_dim, num_samples }, torch::dtype(torch::kComplexFloat).device(device));
    }

    std::cout << "Starting MVMD decomposition with " << num_modes << " modes, "
        << max_iter << " max iterations..." << std::endl;

    for (int iter = 0; iter < max_iter; ++iter) {
        float total_update = 0.0f;

        for (int k = 0; k < num_modes; ++k) {
            torch::Tensor old_mode = modes[k].clone();
            torch::Tensor mode_fft;

            try {
                mode_fft = torch::fft::fft(modes[k], /*dim=*/1);
            }
            catch (const std::exception& e) {
                mode_fft = torch::complex(modes[k], torch::zeros_like(modes[k]));
            }

            torch::Tensor sum_other_modes_fft = torch::zeros_like(mode_fft);
            for (int j = 0; j < num_modes; ++j) {
                if (j != k) {
                    try {
                        sum_other_modes_fft += torch::fft::fft(modes[j], /*dim=*/1);
                    }
                    catch (const std::exception& e) {
                        sum_other_modes_fft += torch::complex(modes[j], torch::zeros_like(modes[j]));
                    }
                }
            }

            auto numerator = signal_fft - sum_other_modes_fft + lambda / 2.0;
            auto denominator = 1.0 + alpha * torch::pow(freqs - omegas[k], 2).unsqueeze(0);
            mode_fft = numerator / denominator.to(torch::kComplexFloat);

            torch::Tensor mode_new;
            try {
                mode_new = torch::real(torch::fft::ifft(mode_fft, /*n=*/num_samples, /*dim=*/1));
            }
            catch (const std::exception& e) {
                mode_new = torch::real(mode_fft);
            }

            total_update += torch::norm(mode_new - old_mode).item<float>();
            modes[k] = mode_new;

            try {
                auto power_spectrum = torch::abs(torch::fft::fft(modes[k], /*dim=*/1)).pow(2).sum(0);
                auto freq_weighted = (power_spectrum * freqs).sum();
                auto power_sum = power_spectrum.sum();
                if (power_sum.item<float>() > 1e-6) {
                    auto new_omega = freq_weighted / power_sum;
                    omegas[k] = 0.9 * omegas[k] + 0.1 * new_omega;
                }
            }
            catch (...) {}
        }

        try {
            torch::Tensor sum_modes_fft = torch::zeros_like(signal_fft);
            for (int k = 0; k < num_modes; ++k) {
                try {
                    sum_modes_fft += torch::fft::fft(modes[k], /*dim=*/1);
                }
                catch (...) {
                    sum_modes_fft += torch::complex(modes[k], torch::zeros_like(modes[k]));
                }
            }
            lambda += 0.3 * (signal_fft - sum_modes_fft);
        }
        catch (...) {}

        if (total_update / num_modes < tol) {
            std::cout << "MVMD converged at iteration " << iter + 1 << std::endl;
            break;
        }

        if (iter % 100 == 0) {
            std::cout << "MVMD Iteration " << iter + 1 << ", Update: " << total_update / num_modes << std::endl;
        }
    }

    torch::Tensor sum_modes = torch::zeros_like(features_tensor);
    for (int k = 0; k < num_modes; ++k) sum_modes += modes[k];
    torch::Tensor residual = features_tensor - sum_modes;

    std::vector<torch::Tensor> all_components;
    for (int k = 0; k < num_modes; ++k) all_components.push_back(modes[k]);
    all_components.push_back(residual);
    torch::Tensor components = torch::stack(all_components).to(device);

    int valid_samples = std::max(1, num_samples - window_size + 1);
    torch::Tensor time_indices = torch::arange(valid_samples, torch::kLong).unsqueeze(1) +
        torch::arange(window_size, torch::kLong).unsqueeze(0);
    time_indices = time_indices.clamp(0, num_samples - 1).to(device);

    torch::Tensor output_features = components.index_select(2, time_indices.view(-1))
        .permute({ 2, 0, 1 })
        .reshape({ valid_samples, (num_modes + 1), window_size * feature_dim })
        .contiguous();

    torch::Tensor targets_tensor_2d = torch::from_blob(const_cast<float*>(targets.data()), { num_samples }, torch::kFloat32)
        .clone().unsqueeze(1).slice(0, window_size - 1, num_samples);

    // 添加回MVMD完成信息
    std::cout << "MVMD decomposition completed! Generated " << valid_samples
        << " windowed samples with " << num_modes + 1 << " components." << std::endl;

    return { output_features.to(device), targets_tensor_2d.to(device) };
}

// 数据读取函数
std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<float>> read_csv_no_outlier_removal(
    const std::string& filename, int input_size, bool has_target = true) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::vector<float>> features;
    std::vector<float> targets;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        try {
            int cols_to_read = has_target ? input_size + 1 : input_size;
            for (int i = 0; i < cols_to_read; ++i) {
                if (!std::getline(ss, cell, ',')) {
                    throw std::invalid_argument("Invalid CSV format");
                }
                float value = std::stof(cell);
                row.push_back(value);
            }
            if (has_target) {
                features.push_back(std::vector<float>(row.begin(), row.begin() + input_size));
                targets.push_back(row[input_size]);
            }
            else {
                features.push_back(row);
                targets.push_back(0.0f);
            }
        }
        catch (const std::exception& e) {
            continue;
        }
    }

    if (features.empty()) {
        throw std::runtime_error("No valid data found in file: " + filename);
    }

    std::vector<float> original_targets = targets;
    std::vector<std::vector<float>> clean_features = features;
    std::vector<float> clean_targets = targets;

    // 特征工程
    std::vector<std::vector<float>> enhanced_features;
    std::vector<float> enhanced_targets;

    for (size_t i = 1; i < clean_features.size(); ++i) {
        std::vector<float> row = clean_features[i];

        // 时间特征
        float t = (i % 6) / 6.0 * 2 * M_PI;
        row.push_back(std::sin(t));
        row.push_back(std::cos(t));

        // 滞后特征
        row.push_back(clean_targets[i - 1]);

        enhanced_features.push_back(row);
        enhanced_targets.push_back(clean_targets[i]);
    }

    return { enhanced_features, enhanced_targets, original_targets };
}

std::tuple<torch::Tensor, torch::Tensor> prepare_data_mvmd(
    const std::vector<std::vector<float>>& features,
    const std::vector<float>& targets,
    int window_size, int num_modes, int max_iter, torch::Device device, float alpha, float tol) {
    auto [components, target_tensor] = mvmd(features, targets, num_modes, window_size, max_iter, alpha, tol, device);
    return { components, target_tensor };
}

// 核心修改：真正的MVMD模态加权注意力机制**
class MVMDModalWeightingImpl : public torch::nn::Module {
public:
    MVMDModalWeightingImpl(int num_modes, int feature_dim, int window_size)
        : num_modes_(num_modes), feature_dim_(feature_dim), window_size_(window_size) {

        // 环境感知编码器 - 分析当前窗口的整体特征
        context_encoder = register_module("context_encoder",
            torch::nn::Sequential(
                torch::nn::Linear(window_size * feature_dim, 64),
                torch::nn::ReLU(),
                torch::nn::Linear(64, 32),
                torch::nn::ReLU()
            ));

        // 模态重要性评估器 - 为每个模态计算重要性分数
        modal_evaluators = torch::nn::ModuleList();
        for (int i = 0; i < num_modes + 1; ++i) {  // +1 for residual
            auto evaluator = torch::nn::Sequential(
                torch::nn::Linear(window_size * feature_dim + 32, 16),  // 模态特征 + 上下文
                torch::nn::ReLU(),
                torch::nn::Linear(16, 1)  // 输出单个重要性分数
            );
            modal_evaluators->push_back(evaluator);
            register_module("modal_eval_" + std::to_string(i), evaluator);
        }

        // 自适应权重计算
        weight_generator = register_module("weight_generator",
            torch::nn::Sequential(
                torch::nn::Linear(num_modes + 1, (num_modes + 1) * 2),
                torch::nn::ReLU(),
                torch::nn::Linear((num_modes + 1) * 2, num_modes + 1)
            ));
    }

    // MVMD模态加权函数
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor mvmd_modes) {
        int batch_size = mvmd_modes.size(0);

        // 1. 计算当前窗口的环境上下文
        auto combined_context = mvmd_modes.mean(1);  // [batch_size, window_size * feature_dim]
        auto context_features = context_encoder->forward(combined_context);  // [batch_size, 32]

        // 2. 为每个模态计算重要性分数
        std::vector<torch::Tensor> modal_scores;
        for (int i = 0; i < num_modes_ + 1; ++i) {
            auto modal_features = mvmd_modes.select(1, i);  // [batch_size, window_size * feature_dim]
            auto combined_input = torch::cat({ modal_features, context_features }, 1);

            auto evaluator = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(modal_evaluators[i]);
            auto score = evaluator->forward(combined_input);  // [batch_size, 1]
            modal_scores.push_back(score);
        }

        // 3. 计算自适应权重
        auto raw_scores = torch::cat(modal_scores, 1);  // [batch_size, num_modes+1]
        auto adaptive_weights = weight_generator->forward(raw_scores);
        auto final_weights = torch::softmax(adaptive_weights, 1);  // [batch_size, num_modes+1]

        // 4. 根据权重加权融合各模态
        torch::Tensor weighted_output = torch::zeros({ batch_size, window_size_ * feature_dim_ }, mvmd_modes.options());
        for (int i = 0; i < num_modes_ + 1; ++i) {
            auto modal_data = mvmd_modes.select(1, i);  // [batch_size, window_size * feature_dim]
            auto weight = final_weights.select(1, i).unsqueeze(1);  // [batch_size, 1]
            weighted_output += weight * modal_data;
        }

        return { weighted_output, final_weights };
    }

private:
    int num_modes_, feature_dim_, window_size_;
    torch::nn::Sequential context_encoder{ nullptr };
    torch::nn::ModuleList modal_evaluators;
    torch::nn::Sequential weight_generator{ nullptr };
};
TORCH_MODULE(MVMDModalWeighting);

// AdaBelief优化器
class AdaBelief_CALR {
public:
    AdaBelief_CALR(std::vector<torch::Tensor> parameters, float lr_max = 1e-3, float lr_min = 1e-6,
        float betas1 = 0.9, float betas2 = 0.999, float eps = 1e-8, int T_max = 50,
        float gamma = 0.98, bool exponential = false, float weight_decay = 0.0)
        : parameters_(parameters), lr_max_(lr_max), lr_min_(lr_min), betas1_(betas1),
        betas2_(betas2), eps_(eps), t_(0), T_max_(T_max), gamma_(gamma),
        exponential_(exponential), weight_decay_(weight_decay) {
        lr_ = lr_max;
        initial_lr_max_ = lr_max;
        initial_lr_min_ = lr_min;
        for (auto& param : parameters_) {
            m_.push_back(torch::zeros_like(param));
            v_.push_back(torch::zeros_like(param));
        }
    }

    void step() {
        t_++;
        update_lr();
        float beta1_t = std::pow(betas1_, std::min(t_, 1000LL));
        float beta2_t = std::pow(betas2_, std::min(t_, 1000LL));

        for (size_t i = 0; i < parameters_.size(); ++i) {
            if (!parameters_[i].grad().defined()) continue;
            auto grad = parameters_[i].grad();

            if (weight_decay_ > 0) {
                grad = grad + weight_decay_ * parameters_[i].data();
            }

            m_[i] = betas1_ * m_[i] + (1 - betas1_) * grad;
            auto s_t = grad - m_[i];
            v_[i] = betas2_ * v_[i] + (1 - betas2_) * s_t.pow(2);
            auto m_hat = m_[i] / (1 - beta1_t);
            auto v_hat = v_[i] / (1 - beta2_t);
            auto v_hat_sqrt = torch::sqrt(torch::maximum(v_hat, torch::zeros_like(v_hat))) + eps_;
            parameters_[i].data() -= lr_ * m_hat / v_hat_sqrt;
        }
    }

    void zero_grad() {
        for (auto& param : parameters_)
            if (param.grad().defined()) param.grad().zero_();
    }

    float get_lr() const { return lr_; }

    void reduce_lr(float factor) {
        lr_max_ *= factor;
        lr_min_ *= factor;
        lr_max_ = std::max(lr_max_, initial_lr_max_ * 1e-3f);
        lr_min_ = std::max(lr_min_, initial_lr_min_ * 1e-1f);
        update_lr();
    }

private:
    void update_lr() {
        if (T_max_ > 0) {
            float cycle_progress = static_cast<float>(t_ % T_max_) / T_max_;
            float cosine_factor = (1 + std::cos(M_PI * cycle_progress)) / 2;
            lr_ = lr_min_ + (lr_max_ - lr_min_) * cosine_factor;

            if (exponential_) {
                float decay = std::exp(-gamma_ * t_ / 5000.0f);
                lr_ *= std::max(decay, 0.05f);
            }
            else {
                float decay = std::pow(gamma_, t_ / 500.0f);
                lr_ *= std::max(decay, 0.05f);
            }

            lr_ = std::clamp(lr_, lr_min_, lr_max_);
        }
    }

    std::vector<torch::Tensor> parameters_;
    std::vector<torch::Tensor> m_, v_;
    float lr_, lr_max_, lr_min_, betas1_, betas2_, eps_;
    float initial_lr_max_, initial_lr_min_;
    int64_t t_;
    int T_max_;
    float gamma_;
    bool exponential_;
    float weight_decay_;
};

// FC预测块
class FCBlockImpl : public torch::nn::Module {
public:
    FCBlockImpl(int input_size, int hidden_size, int output_size, float dropout_rate = 0.1) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
        backcast_fc = register_module("backcast_fc", torch::nn::Linear(hidden_size, input_size));
        forecast_fc = register_module("forecast_fc", torch::nn::Linear(hidden_size, output_size));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto original_sizes = x.sizes();
        int batch_size = original_sizes[0];
        x = x.view({ batch_size, -1 });

        auto h = torch::relu(fc1(x));
        h = dropout(h);
        h = torch::relu(fc2(h));
        h = dropout(h);

        auto backcast = backcast_fc(h).view(original_sizes);
        auto forecast = forecast_fc(h);
        return { backcast, forecast };
    }

private:
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
    torch::nn::Dropout dropout{ nullptr };
    torch::nn::Linear backcast_fc{ nullptr }, forecast_fc{ nullptr };
};
TORCH_MODULE(FCBlock);

// 修改后的Stack类 - 实现分解-加权-预测架构**
class StackImpl : public torch::nn::Module {
public:
    StackImpl(int num_modes, int feature_dim, int window_size, int hidden_size, int output_size,
        int num_blocks, float dropout_rate = 0.1)
        : num_modes_(num_modes), feature_dim_(feature_dim), window_size_(window_size) {

        // 阶段2：MVMD模态加权机制
        modal_weighting = register_module("modal_weighting",
            MVMDModalWeighting(num_modes, feature_dim, window_size));

        // 阶段3：预测块（N-BEATS架构）
        blocks = torch::nn::ModuleList();
        int input_size = window_size * feature_dim;

        for (int i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<FCBlockImpl>(input_size, hidden_size, output_size, dropout_rate);
            register_module("fc_block_" + std::to_string(i), block);
            blocks->push_back(block);
        }

        output_stabilizer = register_module("output_stabilizer",
            torch::nn::Sequential(torch::nn::Linear(output_size, output_size)));
    }

    // 分解-加权-预测的完整流程
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor mvmd_components) {
        // 输入：mvmd_components [batch_size, num_modes+1, window_size * feature_dim]

        // 阶段2：模态加权 - 根据当前窗口环境对MVMD模态进行自适应加权
        auto [weighted_features, modal_weights] = modal_weighting->forward(mvmd_components);

        // 阶段3：预测 - 使用加权后的特征进行层次化预测
        torch::Tensor forecast = torch::zeros({ mvmd_components.size(0), 1 }, mvmd_components.options());
        auto current_input = weighted_features;

        for (size_t i = 0; i < blocks->size(); ++i) {
            auto fc_block = std::dynamic_pointer_cast<FCBlockImpl>(blocks[i]);
            auto [backcast, block_forecast] = fc_block->forward(current_input);
            current_input = current_input - backcast;  // 残差学习
            forecast = forecast + block_forecast;       // 预测累积
        }

        forecast = output_stabilizer->forward(forecast);
        return { forecast, modal_weights };
    }

private:
    MVMDModalWeighting modal_weighting{ nullptr };
    torch::nn::ModuleList blocks;
    torch::nn::Sequential output_stabilizer{ nullptr };
    int num_modes_, feature_dim_, window_size_;
};
TORCH_MODULE(Stack);

// MLP模拟器
class MLPImpl : public torch::nn::Cloneable<MLPImpl> {
public:
    MLPImpl(int input_size, int hidden_size, int output_size, float dropout_rate = 0.2) {
        input_layer = register_module("input_layer", torch::nn::Linear(input_size, hidden_size));
        hidden_layer1 = register_module("hidden_layer1", torch::nn::Linear(hidden_size, hidden_size));
        hidden_layer2 = register_module("hidden_layer2", torch::nn::Linear(hidden_size, hidden_size));
        output_layer = register_module("output_layer", torch::nn::Linear(hidden_size, output_size));

        dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(hidden_size));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(hidden_size));
        bn3 = register_module("bn3", torch::nn::BatchNorm1d(hidden_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        if (x.dim() > 2) x = x.view({ x.size(0), -1 });

        x = torch::relu(bn1(input_layer(x)));
        x = dropout(x);
        x = torch::relu(bn2(hidden_layer1(x)));
        x = dropout(x);
        x = torch::relu(bn3(hidden_layer2(x)));
        x = dropout(x);
        x = output_layer(x);

        return x;
    }

    void reset() override {
        input_layer = register_module("input_layer", torch::nn::Linear(input_layer->options.in_features(), input_layer->options.out_features()));
        hidden_layer1 = register_module("hidden_layer1", torch::nn::Linear(hidden_layer1->options.in_features(), hidden_layer1->options.out_features()));
        hidden_layer2 = register_module("hidden_layer2", torch::nn::Linear(hidden_layer2->options.in_features(), hidden_layer2->options.out_features()));
        output_layer = register_module("output_layer", torch::nn::Linear(output_layer->options.in_features(), output_layer->options.out_features()));
        dropout = register_module("dropout", torch::nn::Dropout(dropout->options.p()));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(bn1->options.num_features()));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(bn2->options.num_features()));
        bn3 = register_module("bn3", torch::nn::BatchNorm1d(bn3->options.num_features()));
    }

private:
    torch::nn::Linear input_layer{ nullptr }, hidden_layer1{ nullptr }, hidden_layer2{ nullptr };
    torch::nn::Linear output_layer{ nullptr };
    torch::nn::Dropout dropout{ nullptr };
    torch::nn::BatchNorm1d bn1{ nullptr }, bn2{ nullptr }, bn3{ nullptr };
};
TORCH_MODULE(MLP);

// MLP训练函数
std::tuple<StandardizationParams, StandardizationParams> train_mlp_model(
    const std::vector<std::vector<float>>& original_features,
    const torch::Tensor& mvmd_components,
    const std::vector<std::vector<float>>& ver_original_features,
    const torch::Tensor& ver_mvmd_components,
    int window_size, int num_modes, int epochs, int batch_size, torch::Device device,
    MLP& mlp_model, AdaBelief_CALR& optimizer,
    int patience, int lr_patience, float lr_factor) {

    int num_samples = original_features.size();
    int feature_dim = original_features[0].size();
    int ver_num_samples = ver_original_features.size();

    torch::Tensor original_tensor = torch::zeros({ num_samples, feature_dim }, torch::kFloat32);
    torch::Tensor ver_original_tensor = torch::zeros({ ver_num_samples, feature_dim }, torch::kFloat32);

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            original_tensor[i][j] = original_features[i][j];
        }
    }
    for (int i = 0; i < ver_num_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            ver_original_tensor[i][j] = ver_original_features[i][j];
        }
    }

    original_tensor = original_tensor.to(device);
    ver_original_tensor = ver_original_tensor.to(device);

    // 创建窗口化数据
    int valid_samples = std::max(1, num_samples - window_size + 1);
    int ver_valid_samples = std::max(1, ver_num_samples - window_size + 1);

    torch::Tensor window_indices = torch::arange(valid_samples, torch::kLong).unsqueeze(1) +
        torch::arange(window_size, torch::kLong).unsqueeze(0);
    window_indices = torch::clamp(window_indices, 0, num_samples - 1).to(device);

    torch::Tensor ver_window_indices = torch::arange(ver_valid_samples, torch::kLong).unsqueeze(1) +
        torch::arange(window_size, torch::kLong).unsqueeze(0);
    ver_window_indices = torch::clamp(ver_window_indices, 0, ver_num_samples - 1).to(device);

    torch::Tensor windowed_original = original_tensor.index_select(0, window_indices.view(-1))
        .view({ valid_samples, window_size, feature_dim });
    torch::Tensor ver_windowed_original = ver_original_tensor.index_select(0, ver_window_indices.view(-1))
        .view({ ver_valid_samples, window_size, feature_dim });

    // 标准化
    auto original_mean = windowed_original.mean({ 0 }, true);
    auto original_std = windowed_original.std({ 0 }, true);
    original_std = torch::clamp(original_std, 1e-4, std::numeric_limits<float>::max());
    windowed_original = (windowed_original - original_mean) / original_std;
    ver_windowed_original = (ver_windowed_original - original_mean) / original_std;

    auto mvmd_mean = mvmd_components.mean({ 0 }, true);
    auto mvmd_std = mvmd_components.std({ 0 }, true);
    mvmd_std = torch::clamp(mvmd_std, 1e-4, std::numeric_limits<float>::max());
    auto mvmd_std_components = (mvmd_components - mvmd_mean) / mvmd_std;
    auto ver_mvmd_std_components = (ver_mvmd_components - mvmd_mean) / mvmd_std;

    StandardizationParams original_params{ original_mean, original_std };
    StandardizationParams mvmd_params{ mvmd_mean, mvmd_std };

    // 训练循环
    float best_ver_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    int lr_counter = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        mlp_model->train();

        torch::Tensor shuffle_indices = torch::randperm(valid_samples, torch::kLong).to(device);
        auto shuffled_original = windowed_original.index_select(0, shuffle_indices);
        auto shuffled_mvmd = mvmd_std_components.index_select(0, shuffle_indices);

        float total_train_loss = 0.0;
        int num_train_batches = 0;

        for (int i = 0; i < valid_samples; i += batch_size) {
            int end_idx = std::min(i + batch_size, valid_samples);
            auto batch_original = shuffled_original.slice(0, i, end_idx);
            auto batch_mvmd = shuffled_mvmd.slice(0, i, end_idx);

            optimizer.zero_grad();
            auto predictions = mlp_model->forward(batch_original);
            predictions = predictions.view(batch_mvmd.sizes());

            auto train_loss = torch::mse_loss(predictions, batch_mvmd);

            train_loss.backward();
            torch::nn::utils::clip_grad_norm_(mlp_model->parameters(), 0.3);
            optimizer.step();

            total_train_loss += train_loss.item<float>();
            num_train_batches++;
        }

        // 验证
        mlp_model->eval();
        torch::NoGradGuard no_grad;
        auto ver_predictions = mlp_model->forward(ver_windowed_original);
        ver_predictions = ver_predictions.view(ver_mvmd_std_components.sizes());
        float avg_ver_loss = torch::mse_loss(ver_predictions, ver_mvmd_std_components).item<float>();

        if (epoch % 50 == 0) {
            std::cout << "MLP Epoch " << epoch + 1 << "/" << epochs
                << ", Train Loss: " << total_train_loss / num_train_batches
                << ", Val Loss: " << avg_ver_loss << std::endl;
        }

        // 学习率调度和早停
        if (avg_ver_loss >= best_ver_loss) {
            lr_counter++;
            if (lr_counter >= lr_patience) {
                optimizer.reduce_lr(lr_factor);
                lr_counter = 0;
            }
        }
        else {
            lr_counter = 0;
        }

        if (avg_ver_loss < best_ver_loss) {
            best_ver_loss = avg_ver_loss;
            patience_counter = 0;
            torch::save(mlp_model, "best_mlp_model.pt");
        }
        else {
            patience_counter++;
        }

        if (patience_counter >= patience) {
            std::cout << "MLP early stopping at epoch " << epoch + 1 << std::endl;
            break;
        }
    }

    torch::load(mlp_model, "best_mlp_model.pt");
    return { original_params, mvmd_params };
}

// MVMD模拟函数
torch::Tensor simulate_mvmd_with_mlp(
    const std::vector<std::vector<float>>& features,
    int window_size, int num_modes, MLP& mlp_model,
    const StandardizationParams& original_params,
    const StandardizationParams& mvmd_params,
    torch::Device device) {

    mlp_model->eval();
    torch::NoGradGuard no_grad;

    int num_samples = features.size();
    int feature_dim = features[0].size();
    int valid_samples = std::max(1, num_samples - window_size + 1);

    torch::Tensor features_tensor = torch::zeros({ num_samples, feature_dim }, torch::kFloat32);
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            features_tensor[i][j] = features[i][j];
        }
    }
    features_tensor = features_tensor.to(device);

    torch::Tensor window_indices = torch::arange(valid_samples, torch::kLong).unsqueeze(1) +
        torch::arange(window_size, torch::kLong).unsqueeze(0);
    window_indices = torch::clamp(window_indices, 0, num_samples - 1).to(device);

    torch::Tensor windowed_input = features_tensor.index_select(0, window_indices.view(-1))
        .view({ valid_samples, window_size, feature_dim });

    windowed_input = (windowed_input - original_params.mean) / original_params.std;

    auto predictions = mlp_model->forward(windowed_input);
    predictions = predictions.view({ valid_samples, num_modes + 1, window_size * feature_dim });
    predictions = predictions * mvmd_params.std + mvmd_params.mean;

    return predictions;
}

// 目标值对齐函数
torch::Tensor create_aligned_targets(
    const std::vector<float>& original_targets,
    int data_samples,
    int window_size) {

    int valid_samples = std::max(1, data_samples - window_size + 1);
    torch::Tensor targets = torch::zeros({ valid_samples, 1 }, torch::kFloat32);

    for (int i = 0; i < valid_samples; ++i) {
        int original_index = i + window_size;
        if (original_index < original_targets.size()) {
            targets[i][0] = original_targets[original_index];
        }
        else {
            targets[i][0] = original_targets.back();
        }
    }

    return targets;
}

// 预测模型训练函数
std::tuple<StandardizationParams> train_prediction_model(
    int batch_size, int epochs, int patience, torch::Device device,
    torch::Tensor data, torch::Tensor target,
    torch::Tensor ver_data, torch::Tensor ver_target,
    AdaBelief_CALR& optimizer, Stack& model) {

    float best_ver_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    StandardizationParams target_params;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model->train();
        float total_loss = 0;

        for (int i = 0; i < data.size(0); i += batch_size) {
            optimizer.zero_grad();
            auto batch_data = data.slice(0, i, std::min(i + batch_size, (int)data.size(0)));
            auto batch_target = target.slice(0, i, std::min(i + batch_size, (int)data.size(0)));

            auto [output, modal_weights] = model->forward(batch_data);
            auto loss = torch::mse_loss(output, batch_target);

            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 0.3);
            optimizer.step();

            total_loss += loss.item<float>() * batch_data.size(0);
        }

        model->eval();
        torch::NoGradGuard no_grad;
        auto [ver_output, ver_weights] = model->forward(ver_data);
        float ver_loss = torch::mse_loss(ver_output, ver_target).item<float>();

        if (epoch % 50 == 0) {
            std::cout << "Pred Epoch " << epoch + 1 << " | Train Loss: " << total_loss / data.size(0)
                << " | Val Loss: " << ver_loss << std::endl;
        }

        if (ver_loss < best_ver_loss) {
            best_ver_loss = ver_loss;
            torch::save(model, "best_model.pt");
            target_params = { target.mean(0, true), target.std(0, true) };
            patience_counter = 0;
        }
        else {
            patience_counter++;
            if (patience_counter >= patience) break;
        }
    }

    torch::load(model, "best_model.pt");
    return { target_params };
}

// 测试函数 - 添加预测时间测量
std::tuple<torch::Tensor, torch::Tensor, double> test_model(
    int batch_size, torch::Device device, torch::Tensor test_data,
    Stack& model, const StandardizationParams& target_params) {
    model->eval();
    torch::NoGradGuard no_grad;
    std::vector<float> predictions;
    std::vector<std::vector<float>> all_modal_weights;

    // 使用chrono测量预测时间
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < test_data.size(0); i += batch_size) {
        int end_idx = std::min(i + batch_size, (int)test_data.size(0));
        auto batch_data = test_data.slice(0, i, end_idx);
        auto [pred, modal_weights] = model->forward(batch_data);

        pred = destandardize(pred, target_params);
        auto cpu_pred = pred.to(torch::kCPU).contiguous();
        auto cpu_weights = modal_weights.to(torch::kCPU).contiguous();

        auto pred_accessor = cpu_pred.accessor<float, 2>();
        auto weight_accessor = cpu_weights.accessor<float, 2>();

        for (int j = 0; j < cpu_pred.size(0); ++j) {
            predictions.push_back(std::max(0.001f, pred_accessor[j][0]));

            std::vector<float> sample_weights;
            for (int k = 0; k < cpu_weights.size(1); ++k) {
                sample_weights.push_back(weight_accessor[j][k]);
            }
            all_modal_weights.push_back(sample_weights);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double prediction_time_ms = static_cast<double>(duration);

    auto pred_tensor = torch::tensor(predictions).reshape({ -1, 1 }).to(device);

    // 返回权重信息用于分析
    torch::Tensor weight_tensor = torch::zeros({ (int)all_modal_weights.size(), (int)all_modal_weights[0].size() });
    for (size_t i = 0; i < all_modal_weights.size(); ++i) {
        for (size_t j = 0; j < all_modal_weights[i].size(); ++j) {
            weight_tensor[i][j] = all_modal_weights[i][j];
        }
    }

    return { pred_tensor, weight_tensor, prediction_time_ms };
}

// 计算局部平滑度(σ_i) - 用于VCPE指标
std::vector<float> calculate_local_smoothness(const torch::Tensor& true_values, int window_radius) {
    auto cpu_values = true_values.to(torch::kCPU).contiguous();
    auto accessor = cpu_values.accessor<float, 2>();
    int n = cpu_values.size(0);
    std::vector<float> smoothness(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        // 确定窗口范围，处理边界情况
        int start = std::max(0, i - window_radius);
        int end = std::min(n - 1, i + window_radius);
        int window_size = end - start + 1;

        // 计算窗口内真实值的均值
        float window_mean = 0.0f;
        for (int j = start; j <= end; ++j) {
            window_mean += accessor[j][0];
        }
        window_mean /= window_size;

        // 计算局部平滑度
        float variance = 0.0f;
        for (int j = start; j <= end; ++j) {
            float diff = accessor[j][0] - window_mean;
            variance += diff * diff;
        }

        // 防止除零和负数平方根
        if (window_size == 1) {
            // 当窗口大小为1时，将平滑度设为1.0
            smoothness[i] = 1.0f;
        }
        else {
            smoothness[i] = std::sqrt(std::max(variance / window_size, 1e-6f));
        }
    }

    return smoothness;
}

// 评估模型性能，包括VCPE指标和预测时间
void evaluate_results(torch::Tensor test_targets, torch::Tensor predictions,
    const StandardizationParams& target_params, double prediction_time_ms, int smoothness_window_radius = 6) {
    auto original_targets = destandardize(test_targets, target_params).to(torch::kCPU);
    auto cpu_preds = predictions.to(torch::kCPU);

    auto target_accessor = original_targets.accessor<float, 2>();
    auto pred_accessor = cpu_preds.accessor<float, 2>();

    float mae = 0.0f, mse = 0.0f, mape = 0.0f, smape = 0.0f;
    float ss_res = 0.0f, ss_tot = 0.0f, mean_target = 0.0f;
    int n = original_targets.size(0);
    int negative_count = 0;

    // 计算局部平滑度
    std::vector<float> local_smoothness = calculate_local_smoothness(original_targets, smoothness_window_radius);

    // 计算VCPE
    float vcpe = 0.0f;

    // 计算均值
    for (int i = 0; i < n; ++i) {
        mean_target += target_accessor[i][0];
    }
    mean_target /= n;

    // 计算所有指标
    for (int i = 0; i < n; ++i) {
        float true_val = target_accessor[i][0];
        float pred_val = pred_accessor[i][0];

        if (pred_val < 0) negative_count++;

        float error = pred_val - true_val;
        float abs_error = std::abs(error);

        mae += abs_error;
        mse += error * error;
        mape += abs_error / (std::abs(true_val) + 1e-6);

        float smape_i = 2.0f * abs_error / (std::abs(true_val) + std::abs(pred_val) + 1e-6);
        smape += smape_i;

        // 计算VCPE - 波动补偿百分比误差
        float vcpe_i = smape_i / (local_smoothness[i] + 1e-6); // 防止除零
        vcpe += vcpe_i;

        ss_res += error * error;
        ss_tot += (true_val - mean_target) * (true_val - mean_target);
    }

    mae /= n;
    mse /= n;
    mape = (mape / n) * 100.0f;
    smape = (smape / n) * 100.0f;
    vcpe = vcpe / n; // 计算VCPE平均值
    float r_square = (ss_tot > 0) ? 1.0f - (ss_res / ss_tot) : 0.0f;

    // 计算每个样本的平均预测时间
    double avg_prediction_time_per_sample_ms = prediction_time_ms / n;

    std::cout << "\n=== Stabilized Evaluation Results (" << n << " samples) ===\n"
        << "===========================================================\n"
        << "MAE: " << mae << "\nMSE: " << mse
        << "\nRMSE: " << std::sqrt(mse) << "\nR²: " << r_square
        << "\nMAPE: " << mape << "%"
        << "\nsMAPE: " << smape << "%"
        << "\nVCPE (k=" << smoothness_window_radius << "): " << vcpe
        << "\nNegative predictions: " << negative_count << " (" << (100.0f * negative_count / n) << "%)"
        << "\nTotal prediction time: " << prediction_time_ms << " ms"
        << "\nAverage prediction time per sample: " << avg_prediction_time_per_sample_ms << " ms\n" << std::endl;

    // 保存结果
    std::ofstream results_file("stabilized_results.txt");
    results_file << "Stabilized Water Quality Prediction Results\n";
    results_file << "===========================================\n";
    results_file << "MAE: " << mae << "\nMSE: " << mse << "\nRMSE: " << std::sqrt(mse)
        << "\nR²: " << r_square << "\nMAPE: " << mape << "%"
        << "\nsMAPE: " << smape << "%"
        << "\nVCPE (k=" << smoothness_window_radius << "): " << vcpe << "%"
        << "\nNegative predictions: " << negative_count
        << "\nTotal prediction time: " << prediction_time_ms << " ms"
        << "\nAverage prediction time per sample: " << avg_prediction_time_per_sample_ms << " ms\n";
    results_file.close();

    // 保存详细预测结果和指标值
    std::ofstream pred_file("stabilized_predictions.csv");
    pred_file << "index,actual,predicted,error,smape,local_smoothness,vcpe\n";
    for (int i = 0; i < n; ++i) {
        float true_val = target_accessor[i][0];
        float pred_val = pred_accessor[i][0];
        float error = pred_val - true_val;
        float smape_i = 2.0f * std::abs(error) / (std::abs(true_val) + std::abs(pred_val) + 1e-6);
        float vcpe_i = smape_i / (local_smoothness[i] + 1e-6);

        pred_file << i << "," << true_val << "," << pred_val << ","
            << error << "," << smape_i << "," << local_smoothness[i] << "," << vcpe_i << "\n";
    }
    pred_file.close();

    std::cout << "Results saved to stabilized_results.txt and stabilized_predictions.csv" << std::endl;
}

// 主函数
int main() {
    std::cout << "=== DEWEPRE: Decomposition-Weighting-Prediction Architecture ===\n";
    std::cout << "Three-Stage Framework: MVMD Decomposition → Modal Weighting → Prediction\n\n";

    torch::Device device = get_device();
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

    {
        int seed = 42;

        std::srand(seed);
        torch::manual_seed(seed);
        if (device.is_cuda()) {
            torch::cuda::manual_seed_all(seed);
        }
    }

    // 超参数设置
    int window_size = 4;
    int input_size = 10;
    int hidden_size = 220;
    int output_size = 1;
    int num_blocks = 10;
    int batch_size = 24;
    int num_modes = 4;
    int max_iter = 800;
    float alpha = 400.0;
    float tol = 1e-7;

    int patience = 200;
    int mlp_patience = 200;
    int mlp_lr_patience = 25;
    float mlp_lr_factor = 0.85;

    float MLP_CALR_lr_max = 4e-4;
    float MLP_CALR_lr_min = 1e-5;
    float PRED_CALR_lr_max = 6e-4;
    float PRED_CALR_lr_min = 1e-5;

    float dropout_rate = 0.06;

    int epochs, mlp_epochs;
    std::cout << "Training epochs: ";
    std::cin >> epochs;
    std::cout << "MLP training epochs: ";
    std::cin >> mlp_epochs;

    try {
        // 1. 读取数据
        auto [features, targets, original_targets] = read_csv_no_outlier_removal("resources/data.csv", 7);
        auto [ver_features, ver_targets, ver_original_targets] = read_csv_no_outlier_removal("resources/verify.csv", 7);
        auto [test_features, test_targets, test_original_targets] = read_csv_no_outlier_removal("resources/test.csv", 7);

        std::cout << "Data loaded successfully!" << std::endl;

        // 2. 阶段1：分解 - MVMD分解训练数据
        std::cout << "\n=== Stage 1: MVMD Decomposition ===" << std::endl;
        auto [data_components, target_low] = prepare_data_mvmd(
            features, targets, window_size, num_modes, max_iter, device, alpha, tol);

        auto [ver_data_components, ver_target_low] = prepare_data_mvmd(
            ver_features, ver_targets, window_size, num_modes, max_iter, device, alpha, tol);

        // 3. 训练MLP模拟器（特征蒸馏）
        std::cout << "\n=== Feature Distillation: Training MLP Simulator ===" << std::endl;
        int mlp_input_size = window_size * input_size;
        int mlp_output_size = (num_modes + 1) * window_size * input_size;
        MLP mlp_model(mlp_input_size, 128, mlp_output_size, dropout_rate);
        mlp_model->to(device);

        AdaBelief_CALR mlp_optimizer(
            mlp_model->parameters(),
            MLP_CALR_lr_max, MLP_CALR_lr_min,
            0.9, 0.999, 1e-8, 200, 0.9998, false, 3e-6
        );

        auto [original_params, mvmd_params] = train_mlp_model(
            features, data_components, ver_features, ver_data_components,
            window_size, num_modes, mlp_epochs, batch_size, device,
            mlp_model, mlp_optimizer, mlp_patience, mlp_lr_patience, mlp_lr_factor);

        // 4. 使用MLP模拟器处理验证和测试数据
        auto ver_data_simulated = simulate_mvmd_with_mlp(
            ver_features, window_size, num_modes, mlp_model, original_params, mvmd_params, device);
        auto test_data_simulated = simulate_mvmd_with_mlp(
            test_features, window_size, num_modes, mlp_model, original_params, mvmd_params, device);

        // 5. 准备目标数据
        torch::Tensor ver_target_aligned = create_aligned_targets(ver_original_targets, ver_features.size(), window_size).to(device);
        torch::Tensor test_target_aligned = create_aligned_targets(test_original_targets, test_features.size(), window_size).to(device);

        // 6. 数据标准化
        auto [data_std, target_std, data_params, target_params] = standardize(data_components, target_low);

        // 验证集：手动应用训练集参数
        auto ver_data_std = (ver_data_simulated - data_params.mean) / data_params.std;
        auto ver_target_std = (ver_target_aligned - target_params.mean) / target_params.std;

        // 测试集：同上
        auto test_data_std = (test_data_simulated - data_params.mean) / data_params.std;
        auto test_target_std = (test_target_aligned - target_params.mean) / target_params.std;

        // 7. 阶段2+3：模态加权 + 预测
        std::cout << "\n=== Stage 2&3: Modal Weighting + Prediction ===" << std::endl;
        Stack model(num_modes, input_size, window_size, hidden_size, output_size, num_blocks, dropout_rate);
        model->to(device);

        AdaBelief_CALR optimizer(
            model->parameters(),
            PRED_CALR_lr_max, PRED_CALR_lr_min,
            0.9, 0.999, 1e-8, 150, 0.9998, false, 3e-6
        );

        // 8. 训练预测模型
        auto [final_target_params] = train_prediction_model(
            batch_size, epochs, patience, device, data_std, target_std,
            ver_data_std, ver_target_std, optimizer, model);

        // 9. 测试 - 现在包含了预测时间测量
        std::cout << "\n=== Testing DEWEPRE Model ===" << std::endl;
        auto [predictions, modal_weights, prediction_time_ms] = test_model(batch_size, device, test_data_std, model, target_params);
        evaluate_results(test_target_std, predictions, target_params, prediction_time_ms);

        // 10. 保存模型
        torch::save(model, "dewepre_model.pt");
        torch::save(mlp_model, "dewepre_mlp_simulator.pt");
        std::cout << "\nDEWEPRE models saved successfully!" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n=== DEWEPRE Training Completed Successfully ===" << std::endl;
    return 0;
}