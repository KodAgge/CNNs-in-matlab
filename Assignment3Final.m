% === Load data ===
[trainX, trainy, valX, valy]= load_data();
trainY = oneHotEncoder(trainy, max(trainy));
valY = oneHotEncoder(valy, max(trainy));
load('assignment3_dimensions.mat') % d | K | n_len
N = size(trainX, 2);

% === Hyper parameters ===

% Architectural parameters
n_1 = 40;
k_1 = 5;

n_2 = 40;
k_2 = 3;

% Training parameters
eta = 0.001;
rho = 0.9;

% Create the network
[ConvNet, n_lens, n_layers] = CreateNetwork(eta, rho, n_len, d, K, n_1, k_1, n_2, k_2, 100);



% batch size | n_epochs | n_update
batch_size = 39;
n_updates = 2e4;

% Balanced
n_epohcs = floor(n_updates / K) + 1;
GDparams = [batch_size, n_epohcs, 500];
[ConvNetTrained, ConfusionMatrices] = MiniBatchGD(trainX, trainY, trainy, valX, valY, valy, GDparams, n_lens, ConvNet, "True");

% Unbalanced
n_epohcs = floor(n_updates / (N / batch_size)) + 1;
GDparams = [batch_size, n_epohcs, 500];
[ConvNetTrained, ConfusionMatrices] = MiniBatchGDUnbalanced(trainX, trainY, trainy, valX, valY, valy, GDparams, n_lens, ConvNet, "True");

% Plot confusion charts
figure
confusionchart(ConfusionMatrices(:, :, end))

% Prediction on friends' surnames
surnames = {'regnell', 'haraldsson', 'logren', 'zhou', 'kerakos', 'ivinskiy'};
P = GuessFriends(surnames, n_lens)

% === Check if gradients are accurate ===

MFs = cell(1,n_layers);
for i = 1:n_layers
    MFs{i} = MakeMFMatrix(ConvNet.F{i}, n_lens{i});
end

% Reduce the size of the dataset
batch_size = length(trainy);
X_batch = trainX(:,1:batch_size);
Y_batch = trainY(:,1:batch_size);

% Evaluate classifier
[P, X_1, X_2] = EvaluateClassifier(X_batch, MFs, ConvNet.W);

% Compute gradients
[grad_W, grad_F1, grad_F2] = ComputeGradients(X_batch, X_1, X_2, Y_batch, P, ConvNet.W, ConvNet.F, n_len);
Gs = NumericalGradient(X_batch, Y_batch, ConvNet, n_lens, 1e-6);

% Check if accurate
h = 1e-6;
GW = DifferenceGradients(grad_W, Gs{end}, h)
GF1 = DifferenceGradients(grad_F1, Gs{1}, h)
GF2 = DifferenceGradients(grad_F2, Gs{2}, h) 

% === Check if MX and MF gives the same answer ===
x_input = trainX(:, 1);

MF = MakeMFMatrix(ConvNet.F{1}, n_len);

[d, k, n_f] = size(ConvNet.F{1});

MX = MakeMXMatrix(x_input, n_len, d, k, n_f);


% Check if computations are correct
s1 = MX * ConvNet.F{1}(:);
s2 = MF * x_input;

diff = s1 - vecS
accuracy = sum(abs(diff) < 1e-6) /length(diff) * 100

% === Check if same as in DebugInfo ===
load('DebugInfo.mat')

MF = MakeMFMatrix(F, n_len);

[d, k, n_f] = size(F);

MX = MakeMXMatrix(x_input, n_len, d, k, n_f);

% Check if computations are correct
s1 = MX * F(:);
s2 = MF * x_input;

diff = s2 - vecS;
accuracy = sum(abs(diff) < 1e-6) /length(diff) * 100


function [ConvNetStar, ConfusionMatrices] = MiniBatchGD(X, Y, y, valX, valY, valy, GDparams, n_lens, ConvNet, plotting)
    % Extract parameters
    n_batch = GDparams(1);
    n_epochs = GDparams(2);
    n_update = GDparams(3);
    n_len = n_lens{1};
    n = length(y);
    n_layers = 2;
    
    % Preallocate memory
    v_W = zeros(size(ConvNet.W));
    v_F1 = zeros(size(ConvNet.F{1}));
    v_F2 = zeros(size(ConvNet.F{2}));
    
    % Reference without training
    MFs = cell(1,n_layers);
    for k = 1:n_layers
        MFs{k} = MakeMFMatrix(ConvNet.F{k}, n_lens{k});
    end
    
    L_val = [ComputeLoss(valX, valY, MFs, ConvNet.W)];
    L_train = [ComputeLoss(X, Y, MFs, ConvNet.W)];

    Acc_train = [ComputeAccuracy(X, y, MFs, ConvNet.W)];
    Acc_val = [ComputeAccuracy(valX, valy, MFs, ConvNet.W)];
    ConfusionMatrices = zeros(ConvNet.K, ConvNet.K, floor(n_epochs * 39 * ConvNet.K / (n_update * n_batch)));
    
    step = [0];
    
    % Precompute the MXs
    MXs = PrecomputeMX(X, ConvNet.F{1});
    
    % Counting the steps
    t = 1;
%     tic
    % Epoch-loop
    for i=1:floor(n_epochs)
        [trainX_sampled, trainy_sampled, trainY_sampled, MXs_sampled] = SampleEqually(X, y, MXs, i);
        % Batch-loop
        
        n_sampled = length(trainy_sampled);
        
        % Deterministic order
        n_batches = n_sampled/n_batch;
        batch_order = 1:n_batches;
        for j=batch_order
            % Extract a batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            indices = j_start:j_end;
            X_batch = trainX_sampled(:, indices);
            Y_batch = trainY_sampled(:, indices);
            
            % Compute MF matrices
            for k = 1:n_layers
                MFs{k} = MakeMFMatrix(ConvNet.F{k}, n_lens{k});
            end

            % Compute P
            [P_batch, X_1_batch, X_2_batch] = EvaluateClassifier(X_batch, MFs, ConvNet.W);

            % Compute gradients
            [grad_W, grad_F1, grad_F2] = ComputeGradients(X_1_batch, X_2_batch, Y_batch, P_batch, ConvNet.W, ConvNet.F, MXs_sampled, indices, n_len);
            
            % Update parameters
            
            % Momentum
            v_W = ConvNet.rho * v_W + ConvNet.eta * grad_W;
            v_F1 = ConvNet.rho * v_F1 + ConvNet.eta * grad_F1;
            v_F2 = ConvNet.rho * v_F2 + ConvNet.eta * grad_F2;
            
            ConvNet.W = ConvNet.W - v_W;
            ConvNet.F{1} = ConvNet.F{1} - v_F1;
            ConvNet.F{2} = ConvNet.F{2} - v_F2;
            
%             if t == 100
%                 toc
%             end                    
            
            if mod(t, n_update) == 0 && plotting == "True"
                "Update step #" + num2str(t) + " completed"
                step = [step, t];
                
                % Update MFs
                for k = 1:n_layers
                    MFs{k} = MakeMFMatrix(ConvNet.F{k}, n_lens{k});
                end
                
                % Calculate losses
                L_val = [L_val, ComputeLoss(valX, valY, MFs, ConvNet.W)];
                L_train = [L_train, ComputeLoss(X, Y, MFs, ConvNet.W)];

                % Calculate accuracies
                Acc_train = [Acc_train, ComputeAccuracy(X, y, MFs, ConvNet.W)];
                Acc_val = [Acc_val, ComputeAccuracy(valX, valy, MFs, ConvNet.W)];
                
                % Calculate confusion matrix
                valy_hat = Prediction(valX, MFs, ConvNet.W);
                C = confusionmat(valy, valy_hat);
                ConfusionMatrices(:, :, t/n_update) = C;
            end
            t = t + 1;
        end
        
    end
    if plotting == "True"
        % Plot the total loss
        subplot(1,2,1)
        plot(step, L_train, 'color', [0, 0.5, 0])
        hold all
        plot(step, L_val, 'r')
        legend('training', 'validation')
        xlabel('update step')
        ylabel('total loss')
        title('Total loss plot')
        yl = ylim;
        ylim([0, yl(2)]);

        % Plot the accuracies
        subplot(1,2,2)
        plot(step, Acc_train, 'color', [0, 0.5, 0])
        hold all
        plot(step, Acc_val, 'r')
        legend('training', 'validation')
        xlabel('update step')
        ylabel('accuracy')
        title('Accuracy plot')
        yl = ylim;
        ylim([0, yl(2)]);

        "MiniBatchGD complete"
    end

    % Save optimal values
    ConvNetStar = ConvNet;  
end

function MXs = PrecomputeMX(X, F1)
    "Pre-computing MX matrices..."
%     tic
    % Put matrices in a cell
    n = size(X, 2);
    MXs = cell(n, 1);
    
    % Dimensions of first filter
    [d, k, n_f] = size(F1);
    
    % Fill the cellarray
    for i = 1:n
       MXs{i} = MakeMXMatrix(X(:, i), d, k, n_f);
    end
    "Done!"
%     toc
end

function X = LoadFriends(surnames)
    N = length(surnames);
    load('assignment3_CharToInd.mat', 'char_to_ind', 'd', 'n_len');
    X = zeros(d * n_len, N);
    j = 1;
    for name_cell = surnames
        name = cell2mat(name_cell);
        x = zeros(d, n_len);
        for i = 1:length(name)
            char_to_ind(name(i));
            x(char_to_ind(name(i)), i) = 1;
            X(:, j) = x(:);
        end
        j = j + 1;
    end
end

function P = GuessFriends(surnames, n_lens)
    load('best_network.mat')
    X = LoadFriends(surnames);
    
    MFs = cell(1,2);
    for i = 1:2
        MFs{i} = MakeMFMatrix(ConvNetTrained.F{i}, n_lens{i});
    end
    
    P = EvaluateClassifier(X, MFs, ConvNetTrained.W);
end

function [ConvNetStar, ConfusionMatrices] = MiniBatchGDUnbalanced(X, Y, y, valX, valY, valy, GDparams, n_lens, ConvNet, plotting)
    % Extract parameters
    n_batch = GDparams(1);
    n_epochs = GDparams(2);
    n_update = GDparams(3);
    n_len = n_lens{1};
    n = length(y);
    n_layers = 2;
    
    % Back-up
    X_original = X;
    
    % Preallocate memory
    v_W = zeros(size(ConvNet.W));
    v_F1 = zeros(size(ConvNet.F{1}));
    v_F2 = zeros(size(ConvNet.F{2}));
    
    % Reference without training
    MFs = cell(1,n_layers);
    for k = 1:n_layers
        MFs{k} = MakeMFMatrix(ConvNet.F{k}, n_lens{k});
    end
    
    L_val = [ComputeLoss(valX, valY, MFs, ConvNet.W)];
    L_train = [ComputeLoss(X, Y, MFs, ConvNet.W)];

    Acc_train = [ComputeAccuracy(X, y, MFs, ConvNet.W)];
    Acc_val = [ComputeAccuracy(valX, valy, MFs, ConvNet.W)];
    ConfusionMatrices = zeros(ConvNet.K, ConvNet.K, floor(n_epochs * n / (n_update * n_batch)));
    
    step = [0];
    
    % Precompute the MXs
    MXs = PrecomputeMX(X, ConvNet.F{1});
    
    % Counting the steps
    t = 1;
    
    % Epoch-loop
    for i=1:floor(n_epochs)
        % Shuffle order
        rng(i)
        shuffled_indices = randperm(n);
        X = X_original(:, shuffled_indices);
        
        % Batch-loop
        
        % Deterministic order
        n_batches = n/n_batch;
        batch_order = 1:n_batches;
        for j=batch_order
            % Extract a batch
            j_start = (j-1)*n_batch + 1;
            j_end = j*n_batch;
            indices = j_start:j_end;
            X_batch = X(:, indices);
            Y_batch = Y(:, indices);
            
            % Compute MF matrices
            for k = 1:n_layers
                MFs{k} = MakeMFMatrix(ConvNet.F{k}, n_lens{k});
            end

            % Compute P
            [P_batch, X_1_batch, X_2_batch] = EvaluateClassifier(X_batch, MFs, ConvNet.W);

            % Compute gradients
            [grad_W, grad_F1, grad_F2] = ComputeGradients(X_1_batch, X_2_batch, Y_batch, P_batch, ConvNet.W, ConvNet.F, MXs, indices, n_len);
            
            % Update parameters
            
            % Momentum
            v_W = ConvNet.rho * v_W + ConvNet.eta * grad_W;
            v_F1 = ConvNet.rho * v_F1 + ConvNet.eta * grad_F1;
            v_F2 = ConvNet.rho * v_F2 + ConvNet.eta * grad_F2;
            
            ConvNet.W = ConvNet.W - v_W;
            ConvNet.F{1} = ConvNet.F{1} - v_F1;
            ConvNet.F{2} = ConvNet.F{2} - v_F2;
                        
            if mod(t, n_update) == 0 && plotting == "True"
                "Update step #" + num2str(t) + " completed"
                step = [step, t];
                
                % Update MFs
                for k = 1:n_layers
                    MFs{k} = MakeMFMatrix(ConvNet.F{k}, n_lens{k});
                end
                
                % Calculate losses
                L_val = [L_val, ComputeLoss(valX, valY, MFs, ConvNet.W)];
                L_train = [L_train, ComputeLoss(X, Y, MFs, ConvNet.W)];

                % Calculate accuracies
                Acc_train = [Acc_train, ComputeAccuracy(X, y, MFs, ConvNet.W)];
                Acc_val = [Acc_val, ComputeAccuracy(valX, valy, MFs, ConvNet.W)];
                
                % Calculate confusion matrix
                valy_hat = Prediction(valX, MFs, ConvNet.W);
                C = confusionmat(valy, valy_hat);
                ConfusionMatrices(:, :, t/n_update) = C;
            end
            t = t + 1;
        end
        
    end
    if plotting == "True"
        % Plot the total loss
        subplot(1,2,1)
        plot(step, L_train, 'color', [0, 0.5, 0])
        hold all
        plot(step, L_val, 'r')
        legend('training', 'validation')
        xlabel('update step')
        ylabel('total loss')
        title('Total loss plot')
        yl = ylim;
        ylim([0, yl(2)]);

        % Plot the accuracies
        subplot(1,2,2)
        plot(step, Acc_train, 'color', [0, 0.5, 0])
        hold all
        plot(step, Acc_val, 'r')
        legend('training', 'validation')
        xlabel('update step')
        ylabel('accuracy')
        title('Accuracy plot')
        yl = ylim;
        ylim([0, yl(2)]);

        "MiniBatchGD complete"
    end

    % Save optimal values
    ConvNetStar = ConvNet;  
end

function [ConvNet, n_lens, n_layers] = CreateNetwork(eta, rho, n_len, d, K, n_1, k_1, n_2, k_2, seed)
    % Parameters
    n_layers = 2;
    n_len_1 = n_len - k_1 + 1;
    n_len_2 = n_len_1 - k_2 + 1;
    n_lens = cell(1,3);
    n_lens{1} = n_len;
    n_lens{2} = n_len_1;
    n_lens{3} = n_len_2;
    fsize = n_2 * n_len_2;
    
    % Set hyperparameters
    ConvNet = ConvNetObject;
    ConvNet.eta = eta;
    ConvNet.rho = rho;
    ConvNet.K = K;

    % Initialization parameters (He)
    sig_1 = sqrt(2 / d);
    sig_2 = sqrt(2 / n_2);
    sig_3 = sqrt(2 / K);

    % Initialization 
    rng(seed);
    ConvNet.F = cell(1,n_layers);
    ConvNet.F{1} = randn(d, k_1, n_1)*sig_1;
    ConvNet.F{2} = randn(n_1, k_2, n_2)*sig_2;
    ConvNet.W = randn(K, fsize)*sig_3;
end

function [trainX_sampled, trainy_sampled, trainY_sampled, MXs_sampled] = SampleEqually(trainX, trainy, MXs, seed)
    % Find class with fewest datapoints
    [counts, values] = groupcounts(trainy);
    K = max(values);
    count_min = min(counts);
    
    % Total number of points in a sampled set
    n = count_min * K;
    
    % Preallocate memoy
    trainX_sampled = zeros(size(trainX, 1), n);
    trainy_sampled = zeros(n, 1);
    MXs_sampled = cell(n, 1);
    
    % Create the matrices
    rng(seed)
    start_index = 1;
    for i = 1:K
        % Find all points belonging to class i
        indices = find(trainy == i);
        
        % Sample indices
        sampled_indices = randsample(indices, count_min);
        sampled_class_X = trainX(:, sampled_indices);
        sampled_class_y = trainy(sampled_indices);
        
        % Put in the right place
        end_index = start_index + count_min - 1;
        
        trainX_sampled(:, start_index:end_index) = sampled_class_X;
        trainy_sampled(start_index:end_index) = sampled_class_y;
        for j = 1:count_min
            MXs_sampled{start_index + j - 1} = MXs{sampled_indices(j)};
        end

        start_index = end_index + 1;
        
    end
    
    % Make the one hot vector
    trainY_sampled = oneHotEncoder(trainy_sampled, K);
end

function [trainX, trainy, valX, valy]= load_data()
    % Load data
    load('assignment3_X.mat')
    
    val_ind = [2 5 29 35 36 37 45 55 56 61 88 93 118 162 171 198 231 235 251 252 275 292 298 306 406 417 423 461 515 551 556 582 589 591 611 711 808 817 825 847 857 862 866 875 937 977 980 1026 1499 1665 2073 2203 2298 2387 2729 2736 3230 3643 3771 4106 4583 4626 4638 4648 4657 4666 4716 4726 4759 4777 4800 4806 4985 5021 5045 5103 5165 5194 5307 5310 5339 5354 5382 5404 5466 5490 5534 5560 5561 5562 5564 5594 5604 5613 5615 5630 5693 5697 5703 5709 5719 5726 5730 5737 5765 5778 5787 5795 5819 5851 5938 5960 5990 6003 6055 6153 6157 6180 6357 6433 6589 6614 6677 6727 6906 7094 7137 7151 7188 7201 7228 7370 7494 7497 7500 7504 7510 7514 7524 7545 7550 7560 7565 7566 7571 7576 7585 7588 7599 7650 7651 7653 7657 7660 7668 7689 7704 7709 7710 7711 7712 7716 7718 7719 7723 7726 7738 7739 8133 8297 8517 8636 11097 12736 14282 15215 15280 15695 16028 16269 17057 17061 17076 17078 17080 17081 17084 17088 17097 17102 17105 17107 17113 17140 17152 17157 17195 17237 17242 17254 17255 17274 17293 17326 17358 17359 17366 17367 17373 17377 17378 17380 17382 17400 17408 17417];
    valX = X(:,val_ind);
    valy = ys(val_ind);
    
    X(:,val_ind) = [];
    ys(val_ind) = [];
    
    trainX = X;
    trainy = ys;
end

function MF = MakeMFMatrix(F, n_len)
    % Outputs MF of size (n_len-k+1)*n_f X n_len*dd
    [dd, k, n_f] = size(F);
    
    %  r X c
    n_rows = (n_len-k+1)*n_f;
    n_columns = n_len*dd;
    
    % Allocate memory
    MF = zeros(n_rows, n_columns);
    V_F = VectorizeF(F, n_f, dd, k);
    
    % Create MF
    base_rows = 1:n_f;
    base_columns = 1:(dd * k);
    
    for j = 1:(n_rows/n_f) % for every row in V_F
        rows = n_f * (j - 1) + base_rows;
        columns = (j-1) * dd + base_columns;
        MF(rows, columns) = V_F;
    end
end

function V_F = VectorizeF(F, n_f, d, k)
    V_F = zeros(n_f, d * k);
    
    % Vectorize F
    for i = 1:n_f
        f = F(:,:,i);
        V_F(i,:) = f(:);
    end
end

function MX = MakeMXMatrix(x_input, d, k, n_f)
    % Outputs MX of size (n_len-k+1)*nf X k*n_f*d

    n_len = size(x_input, 1) / d;

    % Allocate memory
    MX = sparse((n_len - k + 1) * n_f, (k  * n_f * d));
    
    % Reshape
    X_input = reshape(x_input, [d, n_len]);
    
    % Create MX
    base_columns = 1:k;
    start_row = 1;
    
    for i = 1:(n_len-k+1)
        X = X_input(:, (i - 1) + base_columns);
        end_row = start_row + n_f - 1;
        MX(start_row:end_row, :) = kron(eye(n_f), X(:)');
        start_row = end_row + 1;
    end
end

function loss = ComputeLoss(X_batch, Ys_batch, MFs, W)
    % Computes the loss
    
    % Evalute the classifier
    P = EvaluateClassifier(X_batch, MFs, W);

    % Cross entropy loss
    loss = -sum(log(sum(P .* Ys_batch))) / size(X_batch, 2);
end

function [P, X_1, X_2] = EvaluateClassifier(X_batch, MFs, W)
    % First layer
    X_1 = max(MFs{1} * X_batch, 0);

    % Second layer
    X_2 = max(MFs{2} * X_1, 0);
    
    % Fully connected layer
    S = W * X_2;
    
    % Soft max
    P = SoftMax(S);
end

function P = SoftMax(S)
    P = exp(S) ./ sum(exp(S));
end

function Y = oneHotEncoder(y, C)
    Y = (y==(1:C))';
end

function [grad_W, grad_F1, grad_F2] = ComputeGradients(X_1, X_2, Y, P, W, Fs, MXs, indices, n_len)
    % === Step 0 ===
    % Find dimensions and allocate memory
    [d, k_1, n_1] = size(Fs{1});
    [~, k_2, n_2] = size(Fs{2});
    n_len_1 = n_len - k_1 + 1;
    
    grad_F1 = zeros(size(Fs{1}));
    grad_F2 = zeros(size(Fs{2}));

    % === Step 1 ===
    % Calculate necessary variables
    G = -(Y-P);
    n = size(Y, 2);
    
    % === Step 2 ===
    % Calc gradient of fully connected layer
    grad_W = G * X_2' / n;
    
    % === Step 3 ===
    % Propogate one step back
    G = W' * G;                 % n_2 * n_len_2 X n
    G = G .* (X_2 > 0); % Indicator function
    
    % === Step 4 ===
    % Second convolutional layer
    for i = 1:n
        g_i = G(:,i);                                       % Eq 35
        x_i = X_1(:,i);                                     % Eq 36
        M_x_i = MakeMXMatrix(x_i, n_1, k_2, n_2);    
        v = g_i' * M_x_i;                                   % Eq 37
        v = reshape(v, [n_1, k_2, n_2]);
        grad_F2 = grad_F2 + v / n;                          % Eq 38
    end
    
    % === Step 5 ===
    % Propogate another step back
    MF_2 = MakeMFMatrix(Fs{2}, n_len_1);
    G = MF_2' * G;
    G = G .* (X_1 > 0); % Indicator function

    % === Step 6 ===
    % First convolutional layer
    for j = 1:n
        g_j = G(:,j);                                       % Eq 41       
        M_x_j = MXs{indices(j)};
        v = g_j' * M_x_j;                                   % Eq 37
        v = reshape(v, [d, k_1, n_1]);
        grad_F1 = grad_F1 + v / n;                          % Eq 38
    end
end

function loss = Compute_loss(X, Y, ConvNet, n_lens)
    n_layers = 2;

    MFs = cell(1,n_layers);
    for i = 1:n_layers
        MFs{i} = MakeMFMatrix(ConvNet.F{i}, n_lens{i});
    end
    
    loss = ComputeLoss(X, Y, MFs, ConvNet.W);

end

function Gs = NumericalGradient(X_inputs, Ys, ConvNet, n_lens, h)
    try_ConvNet = ConvNet;
    Gs = cell(length(ConvNet.F)+1, 1);

    for l=1:length(ConvNet.F)
        try_convNet.F{l} = ConvNet.F{l};

        Gs{l} = zeros(size(ConvNet.F{l}));
        nf = size(ConvNet.F{l},  3);

        for i = 1:nf        
            try_ConvNet.F{l} = ConvNet.F{l};
            F_try = squeeze(ConvNet.F{l}(:, :, i));
            G = zeros(numel(F_try), 1);

            for j=1:numel(F_try)
                F_try1 = F_try;
                F_try1(j) = F_try(j) - h;
                try_ConvNet.F{l}(:, :, i) = F_try1; 

                l1 = Compute_loss(X_inputs, Ys, try_ConvNet, n_lens);

                F_try2 = F_try;
                F_try2(j) = F_try(j) + h;            

                try_ConvNet.F{l}(:, :, i) = F_try2;
                l2 = Compute_loss(X_inputs, Ys, try_ConvNet, n_lens);            

                G(j) = (l2 - l1) / (2*h);
                try_ConvNet.F{l}(:, :, i) = F_try;
            end
            Gs{l}(:, :, i) = reshape(G, size(F_try));
        end
    end

    % compute the gradient for the fully connected layer
    W_try = ConvNet.W;
    G = zeros(numel(W_try), 1);
    for j=1:numel(W_try)
        W_try1 = W_try;
        W_try1(j) = W_try(j) - h;
        try_ConvNet.W = W_try1; 

        l1 = Compute_loss(X_inputs, Ys, try_ConvNet, n_lens);

        W_try2 = W_try;
        W_try2(j) = W_try(j) + h;            

        try_ConvNet.W = W_try2;
        l2 = Compute_loss(X_inputs, Ys, try_ConvNet, n_lens);            

        G(j) = (l2 - l1) / (2*h);
        try_ConvNet.W = W_try;
    end
    Gs{end} = reshape(G, size(W_try));

end

function [accuracy] = DifferenceGradients(analyticalG, numericalG, tol)
    relative_error = ComputeRelativeError(analyticalG, numericalG);
    errors = (relative_error > tol);
    accuracy = 100 * (1-sum(errors(:)) / numel(relative_error));
end

function [relative_error] = ComputeRelativeError(analyticalX, numericalX)
    relative_error = abs(analyticalX - numericalX) ./ ...
        max(eps, abs(analyticalX) + abs(numericalX));
%     abs_error = abs(analyticalX - numericalX);
end

function acc = ComputeAccuracy(X, y, MFs, W)
    y_hat = Prediction(X, MFs, W);
    acc = sum(y_hat == y) / length(y);
end

function y_hat = Prediction(X, MFs, W)
    [~, y_hat] = max(EvaluateClassifier(X, MFs, W));
    y_hat = y_hat';
end
