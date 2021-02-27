%% Clear Console 
clc;
%% Clear Workspace 
clear;
%% Close Figures
close all;

%% Load Dataset

load Abalone.mat;

%% Prepare Labels

abalone_labels = abalone_table.Properties.VariableNames (:,1:8);

%% Prepare Gender Groups

abalone_gender = findgroups(abalone_table (:, 1));

%% Prapare Measurements

abalone_measurements = horzcat(abalone_gender, abalone_table {:, 2:8}) ;

%% Function Call Finds The Centered Data Matrix

variables_centered = normalize(abalone_measurements);

%% Request User Input To Select An Attribute  

prompt = 'Choose one attribute to see its analytics: \n 1. Gender (1:F, 2:I, 3:M)\n 2. Length\n 3. Diameter\n 4. Height\n5. Whole Weight\n 6. Shucked Weight \n 7. Viscera Weight\n 8. Shell Weight\n 9. All (For PCA) \n';
attribute = input(prompt);

%% Create A New Matrix Due To User Input

if attribute == 1
    selected_measurements = abalone_gender;
elseif attribute == 9  
        % Normalize The Scale For PCA 
        selected_measurements = variables_centered;
else
    selected_measurements = abalone_measurements(:, attribute);
end

%% Find Statics 

if attribute < 9 
    fprintf('\nAttribute   : %s\n', abalone_labels{:, attribute});
else
    fprintf('\nAttribute   : All(Centered)\n');
end

% Size Of Selected Attribute

variables_size = size(selected_measurements);

%  Mean Value Of Selected Attribute

variables_mean = mean(selected_measurements);
fprintf('Mean        : %f\n', variables_mean);

% Median Value Of Selected Attribute

variables_median = median(selected_measurements);
fprintf('Median      : %f\n', variables_median);

% Total Sum Of Selected Atribute

variables_sum = sum(selected_measurements);
fprintf('Sum         : %f\n', variables_sum);

% Maximum Value of Selected Attribute 

variables_max = max(selected_measurements);
fprintf('Max Value   : %f\n', variables_max);

% Range Of Selected Attribute

variables_range = range(selected_measurements);
fprintf('Range       : %f\n', variables_range);

% Variance Of Selected Attribute

variables_variance = var (selected_measurements);
fprintf('Variance    : %f\n', variables_variance);

% Deviation Of Selected Attribute

variables_deviation = std(selected_measurements);
fprintf('Deviation   : %f\n', variables_deviation);

% Skewness Of Selected Attribute (flag = 1), By The Default))
% Positive
% Mode < Median < Mean
% Normal
% Skewness Is Not Exactly Zero, It Is Nearly Zero
% Negative Skewness
% Mean < Median < Mode
variables_skewness = skewness(selected_measurements);
fprintf('Skewness    : %f\n', variables_skewness);

% Kurtosis of Selected Attribute (flag = 1), By The Default))
% Mesokurtic (0 Or Close To 0) Normal Distribution
% Lepokurtic (Positive): The Leptokurtic Distribution Shows Heavy Tails On Either Side, Indicating Large Outliers
% Platykurtic (Negative): The Flat Tails Indicate The Small Outliers In A Distribution
variables_kurtosis = kurtosis(selected_measurements);
fprintf('Kurtosis    : %f\n', variables_kurtosis);

%% Create A Structure

abaloneInput.time = (0:74)';
abaloneInput.signals.dimensions = 4;
abaloneInput.signals.values = abalone_measurements;

%% Function Call: Return Outliers Count

variables_numberofoutliers = findOutliers(selected_measurements);
fprintf('Outliers    : %d\n\n', variables_numberofoutliers);

%% Calculate Eigenvalues And Eigenvectors Of The Covariance Matrix

% "coeff" Are The Principal Component Vectors Of The Covariance 
% Matrix These Are The "Eigenvectors" ([coeff,score,latent] = pca(X))  
covariance_matrix = cov (selected_measurements);
[eigenvectors, eigenvalues] = eig (covariance_matrix);
eigenvectors_cov = flip(eigenvectors,2);

% Multiply The Original Data By The Principal Component Vectors To Get The Projections Of The Original Data 
% Principal Component Vector Space As Known As "score" ([coeff,score,latent] = pca(X)) 
projections = selected_measurements * eigenvectors_cov;

%% Figure 1: Visualize Summary Statistics With Box Plot

figure('Name','Visualize Summary Statistics With Box Plot','NumberTitle','off', 'Units','normalized','Position',[0.3 0.3 0.3 0.5]);

if attribute < 9
    subplot(2,1,1);
    boxplot(selected_measurements, 'Widths', 0.10);
    set(gca,'xticklabel', abalone_labels(:, attribute), 'fontsize', 12);
    
    xlabel(abalone_labels(:, attribute), 'fontweight', 'bold', 'fontsize', 12);
    ylabel('Magnitude', 'fontweight', 'bold', 'fontsize', 12);
    title('Visualization Of Summary Statistics', 'fontweight', 'bold', 'fontsize', 15);
        
    subplot(2,1,2);
    boxplot(variables_centered(:, attribute), 'Widths', 0.10);

    xlabel(abalone_labels(:, attribute), 'fontweight', 'bold', 'fontsize', 12);
    ylabel('Magnitude', 'fontweight', 'bold', 'fontsize', 12);
    title('Visualization Of Summary Statistics (Centered)', 'fontweight', 'bold', 'fontsize', 15);
    set(gca,'xticklabel', abalone_labels(:, attribute), 'fontsize', 12);
else
    boxplot(selected_measurements, 'Widths', 0.10);
    
    xlabel('Dimensions', 'fontweight', 'bold', 'fontsize', 12);
    ylabel('Magnitudes', 'fontweight', 'bold', 'fontsize', 12);
    title('Visualization Of Summary Statistics (Centered)', 'fontweight', 'bold', 'fontsize', 15);
    set(gca,'xticklabel', abalone_labels, 'fontsize', 12);
end
 

%% Figure 2: View Covariance Principal Components

if (attribute > 8) 
    
    figure('Name','View Covariance Principal Components','NumberTitle','off', 'Units','normalized','Position',[0.3 0.3 0.3 0.5]);
    
    % PC1 vs PC2 vs PC3
    subplot(2,2,1);
    biplot(eigenvectors_cov (:, 1:3), 'scores', projections (:, 1:3), 'VarLabels', abalone_labels);
    xlabel ('PC 1');
    ylabel ('PC 2');
    zlabel ('PC 3');
    title ('PC-1 vs PC-2 vs PC-3');
    
    % PC2 vs PC3 vs PC4
    subplot(2,2,2);
    biplot(eigenvectors_cov (:, 2:4), 'scores', projections (:, 2:4), 'VarLabels', abalone_labels, 'ObsLabels', abalone_rings_string');
    xlabel ('PC 2');
    ylabel ('PC 3');
    zlabel ('PC 4');
    title ('PC-2 vs PC-3 vs PC-4');
    
    % PC4 vs PC5 vs PC6
    subplot(2,2,3);
    biplot(eigenvectors_cov (:, 4:6), 'scores', projections (:, 4:6), 'VarLabels', abalone_labels, 'ObsLabels', abalone_rings_string');
    xlabel ('PC 4');
    ylabel ('PC 5');
    zlabel ('PC 6');
    title ('PC-4 vs PC-5 vs PC-6');
    
    % PC6 vs PC7 vs PC8
    subplot(2,2,4);
    biplot(eigenvectors_cov (:, 6:8), 'scores', projections (:, 6:8), 'VarLabels', abalone_labels, 'ObsLabels', abalone_rings_string');
    xlabel ('PC 6');
    ylabel ('PC 7');
    zlabel ('PC 8');
    title ('PC-6 vs PC-7 vs PC-8');

    %% Figure 3: View Covariance Principal Components Variance

    figure('Name','View Covariance Principal Components Variance And Deviation','NumberTitle','off', 'Units','normalized','Position',[0.3 0.3 0.3 0.5]);
    
    % "latent" Principal Component Variances, That Is The Eigenvalues Of
    % The Covariance Matrix Of X, Returned As A Column Vector ([coeff,score,latent] = pca(X))  
    variance = var(projections)';
    variance_percentage = 100*variance/sum(variance)';

    subplot(3,1,1);
    bar(variance_percentage, 'FaceColor', [0 .5 .5], 'EdgeColor', [0 .9 .9], 'LineWidth', 1.5);
    set(gca,'xticklabel', abalone_labels, 'fontsize', 10);
    xlabel('Covariance Principal Component', 'fontweight', 'bold', 'fontsize', 12);
    ylabel('Propotion Of Variance (%)', 'fontweight', 'bold', 'fontsize', 12);
    
    %% Figure 3: View Covariance Principal Components Deviation
    
    deviation = std(projections);
    deviation_percentage = 100*deviation/sum(deviation);
    
    subplot(3,1,2);
    bar(deviation_percentage, 'FaceColor', [0 .5 .5], 'EdgeColor', [0 .9 .9], 'LineWidth', 1.5);
    set(gca,'xticklabel', abalone_labels, 'fontsize', 10);
    xlabel('Covariance Principal Component', 'fontweight', 'bold', 'fontsize', 12);
    ylabel('Standart Deviation (%)', 'fontweight', 'bold', 'fontsize', 12);
    
    %% Figure 3: View Covariance Principal Components Proportion
    
    propotion = cumsum(variance/sum(variance));
    propotion_percentage = 100*deviation/sum(deviation);
    
    subplot(3,1,3);
    bar(propotion_percentage, 'FaceColor', [0 .5 .5], 'EdgeColor', [0 .9 .9], 'LineWidth', 1.5);
    set(gca,'xticklabel', abalone_labels, 'fontsize', 10);
    xlabel('Covariance Principal Component', 'fontweight', 'bold', 'fontsize', 12);
    ylabel('Cumulative Propotion (%)', 'fontweight', 'bold', 'fontsize', 12);
    
    % Display The Columns of Projections Are Orthogonal 
    disp(corrcoef(projections));
    
end

%% Function Call: User Input (COMMENT THIS SECTION BEFORE TRAINING THE MODEL)

abalone_sample = userinput();

%% Function Call: Abalone Predict (COMMENT THIS SECTION BEFORE TRAINING THE MODEL)

abalone_predict = abalonePredict(abalone_sample);
fprintf('\nRing Count -> %s\n', abalone_predict{:});

%% Function Call: JARQUE-BERA Normality Test (COMMENT THIS SECTION BEFORE TRAINING THE MODEL)

[abaloneJB, abalonePVal] = jarqueBera(abalone_measurements);
fprintf('\nJBtest Abalone(ref)    : %f, JValue Abalone(ref)      : %f \n', jarqueBera(abaloneJB), jarqueBera(abalonePVal));

[SampleJB, SamplePVal] = jarqueBera(abalone_sample);
fprintf('JBtest Sample          : %f, PValue Sample            : %f \n', SampleJB, SamplePVal);

%% Partition The Data Into A Training 

if (attribute > 8) 
    
    sum_variance = 0;
    idx = 0;

    % Components Explain More Than 95% Of All Variability
    while sum_variance < 95
        idx = idx + 1;
        sum_variance = sum_variance + variance_percentage(idx);
    end
    
    % Discriminant Linear Classification Training 
    AbaloneTrainedModel = fitcdiscr(abalone_measurements(:,1:idx),abalone_rings_string);
    
    % Tree Classification 
    %AbaloneTrainedModel = fitctree(abalone_measurements(:,1:idx),abalone_rings_string);
    
    % Pass A Transformed Test To The Trained Model
    varianceTest = projections(:,1:idx);

    abalone_rings_predicted = predict(AbaloneTrainedModel,varianceTest);
    
    % Save The Classification Model
    saveLearnerForCoder(AbaloneTrainedModel,'AbaloneTrainedModel');
    
end

%% Function: JARQUE-BERA TEST

function [JBtest, pValue] = jarqueBera(measurements)
JBtest = length(measurements)*(((skewness(measurements).^2)/6)+(((kurtosis(measurements)-3).^2)/24));
pValue = 1-chi2cdf(JBtest,2);
end

%% Function: Predict Species Of Abalone Using Trained Model

function abaloneLabel = abalonePredict(measurements) %#codegen
    model = loadLearnerForCoder('AbaloneTrainedModel');
    abaloneLabel = predict(model,measurements);   
end

%% Function: Retrieve Sample Data From The User

function abaloneSampleData = userinput() 
    fprintf('\nEnter A Measurement\n');
    fprintf('-----------------------\n');
    Gender = input('1. Gender (1:F, 2:I, 3:M)  :');
    Length = input('2. Length                  :');
    Diameter = input('3. Diameter                :');
    Height = input('4. Height                  :');
    WholeWeight = input('5. Whole Weight            :');
    ShuckedWeight = input('6. Shucked Weight          :');
    VisceraWeight = input('7. Viscera Weight          :');
    ShellWeight = input('8. Shell Weight            :');
    abaloneSampleData = [Gender, Length, Diameter, Height, WholeWeight, ShuckedWeight, VisceraWeight, ShellWeight];
end

%% Function: Find The Number Of Outliers

function outliers = findOutliers(measurements)
TF = isoutlier(measurements, 'quartiles');
[row, column] = size(measurements);
outliers = 0;
    for i = 1:row
        for j = 1:column
            if (TF(i,j) == 1)
                outliers = outliers + 1;
            end
        end 
    end
end

%% Function Find The Centered Data Matrix Using Elementwise Multiplication

%function centered = centeredMatrix(measurements)
%    n = size(measurements);
%    centered = measurements - mean(measurements)/std(measurements).*ones(n);
%end