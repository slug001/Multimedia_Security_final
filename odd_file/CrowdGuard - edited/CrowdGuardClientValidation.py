# Copyright (C) 2022-2024 TU Darmstadt
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------
# Primary author: Phillip Rieger <phillip.rieger@trust.tu-darmstadt.de>
# Co-authored-by: Torsten Krauss <torsten.krauss@uni-wuerzburg.de>
# ------------------------------------------------------------
from enum import Enum
import math
from copy import deepcopy
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import kstest, levene, ttest_ind
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from torch import cosine_similarity


class DistanceMetric(str, Enum):
    """Enum to identify distance metrics necessary in this project"""
    COSINE = 'cosine'
    EUCLIDEAN = 'euclid'


class DistanceHandler:
    """Helper, that calculates distances between two tensors."""
    """一個輔助類，封裝了計算兩個張量之間不同類型距離的邏輯。"""

    @staticmethod   # 計算兩個張量 t1 和 t2 之間的歐氏距離。它首先將張量展平 (flatten)，然後計算差值的 L2 範數。
    def __get_euclid_distance(t1: torch.Tensor, t2: torch.Tensor) -> float:
        t = t1.view(-1) - t2.view(-1)
        return torch.norm(t, 2).cpu().item()

    @staticmethod  # 計算兩個張量 t1 和 t2 之間的餘弦距離（1 - 餘弦相似度）。同樣會先展平張量。
    def __get_cosine_distance(t1: torch.Tensor, t2: torch.Tensor) -> float:
        t1 = t1.view(-1).reshape(1, -1)
        t2 = t2.view(-1).reshape(1, -1)
        return 1 - cosine_similarity(t1, t2).cpu().item()

    @staticmethod 
    def get_distance(distance, t1: torch.Tensor, t2: torch.Tensor) -> float:
        """Factory Method for Distances"""
        if distance == DistanceMetric.COSINE:
            return DistanceHandler.__get_cosine_distance(t1, t2)
        if distance == DistanceMetric.EUCLIDEAN:
            return DistanceHandler.__get_euclid_distance(t1, t2)

        raise Exception(f"Extractor for {distance} not implemented yet.")

"""
用途： 包含了客戶端執行模型驗證的完整流程。
一個驗證客戶端會接收到一組本地模型和一個全局模型，然後使用自己的本地數據來評估這些模型，最終輸出一組它認為是可疑（被後門污染）的模型的索引。
"""
class CrowdGuardClientValidation:

    @staticmethod
    def __distance_global_model_final_metric(distance_type: str, prediction_matrix,
                                             prediction_global_model, sample_indices_by_label,
                                             own_index):
        """
        Calculates the distance matrix containing the metric for CrowdGuard
        with dimensions label x model x layer x values

        計算 HLBIM (Hidden Layer Backdoor Inspection Metric) 矩陣。 
        這是 CrowdGuard 的關鍵創新點。

        輸入：
            distance_type: 'cosine' 或 'euclid'。
            prediction_matrix: 一個多維列表/張量，儲存了每個樣本在每個待驗證本地模型的每一層的隱藏層輸出 (DLOs)。維度大致是 [num_samples, num_local_models, num_layers, feature_dim_of_layer]。
            prediction_global_model: 類似 prediction_matrix，但只包含全局模型的 DLOs。維度大致是 [num_samples, num_layers, feature_dim_of_layer]。
            sample_indices_by_label: 一個字典，key 是數據標籤，value 是具有該標籤的樣本在驗證集中的索引列表。
            own_index: 當前執行驗證的這個客戶端，其自己的模型在 prediction_matrix 的第二個維度（即本地模型列表）中的索引。
        """

        sample_count = len(prediction_matrix)
        model_count = len(prediction_matrix[0])
        layer_count = len(prediction_matrix[0][0])

        # We create a distance matrix with distances between global and local models
        # of the dimensions sample x model x layer x values
        global_distance_matrix = [[[0.] * layer_count for _ in range(model_count)]
                                  for _ in range(sample_count)]
        # 1. calculate distances between predictions of global model and each local model
        for s_i, s in enumerate(prediction_matrix):
            g = prediction_global_model[s_i]
            for m_i, m in enumerate(s):
                for l_i, l in enumerate(m):
                    distance = DistanceHandler.get_distance(distance_type, l, g[
                        l_i])  # either euclidean or cosine distance
                    global_distance_matrix[s_i][m_i][l_i] = distance  # line 18

        # 2. Sort the sample-wise distances by the label of the sample
        for label, sample_list in sample_indices_by_label.items():
            # First pick the samples from the global predictions
            global_distance_matrix_for_label_helper = [
                [[0.] * len(sample_list) for _ in range(layer_count)] for _ in
                range(model_count)]

            s_i_new = 0
            for s_i, s in enumerate(global_distance_matrix):
                if s_i not in sample_list:
                    continue
                for m_i, mi in enumerate(s):
                    for l_i, l in enumerate(mi):
                        global_distance_matrix_for_label_helper[m_i][l_i][s_i_new] = l
                s_i_new += 1

        # We produce the first relative matrix
        sample_relation_matrix = [[[0.] * layer_count for _ in range(model_count)] for _ in
                                  range(sample_count)]

        # 3. divide by distances of this client to use its values as reference
        for s_i, s in enumerate(global_distance_matrix):
            distances_for_own_models_predictions = s[own_index]
            for m_j, mj in enumerate(s):
                for l_i, l in enumerate(mj):
                    relation = 0
                    if distances_for_own_models_predictions[l_i] != 0:
                        relation = l / distances_for_own_models_predictions[l_i]
                    sample_relation_matrix[s_i][m_j][l_i] = relation  # line 21

        # We produce the Label average
        # We produce a matrix with not all samples, but mean all the samples, so that we have a
        # Matrix per label
        sample_relation_matrix_for_label = {}

        # 4. Transpose matrix as preparation for averaging
        for label, sample_list in sample_indices_by_label.items():
            sample_relation_matrix_for_label[label] = [[0.] * layer_count for _ in
                                                       range(model_count)]
            sample_relation_matrix_for_label_helper = [
                [[0.] * len(sample_list) for _ in range(layer_count)] for _ in range(model_count)]
            # transpose dimensions of distance matrix, before we had (sample,model, layer) and
            # we transpose it to (model,layer,sample)
            s_i_new = 0
            for s_i, s in enumerate(sample_relation_matrix):
                if s_i not in sample_list:
                    continue
                for m_j, mj in enumerate(s):
                    for l_i, l in enumerate(mj):
                        sample_relation_matrix_for_label_helper[m_j][l_i][s_i_new] = l
                s_i_new += 1

            # 5. Average over all samples from the same label (basically kick-out the last
            # dimension)
            for m_j, mj in enumerate(sample_relation_matrix_for_label_helper):
                for l_i, l in enumerate(mj):
                    sample_relation_matrix_for_label[label][m_j][l_i] = np.mean(l).item()

        avg_sample_relation_matrix_squared_negative_models_first = {}

        # 6. subtract 1 (mainly for cosine distances) and square (but keep the sign)
        for label, label_values in sample_relation_matrix_for_label.items():
            avg_sample_relation_matrix_squared_negative_models_first[label] = [[0.] * layer_count
                                                                               for _ in
                                                                               range(model_count)]
            for m_j, mj in enumerate(label_values):
                for l_i, l in enumerate(mj):
                    x = l - 1
                    relation = x * x
                    relation = relation if x >= 0 else relation * (-1)
                    avg_sample_relation_matrix_squared_negative_models_first[label][m_j][
                        l_i] = relation
        return avg_sample_relation_matrix_squared_negative_models_first

    @staticmethod
    def __predict_for_single_model(model, local_data, device):
        """
        Returns
        - A matrix with Deep Layer Outputs with dimensions sample x layer x values.
        - The labels for all samples in the client's training dataset
        - The number of layers defined in the model

        用途： 使用給定的單個模型 (model) 在其本地驗證數據 (local_data) 上進行前向傳播，並提取所有預定義的隱藏層的輸出 (DLOs)。
        輸入：
            model: 一個 nn.Module 實例，且必須有 predict_internal_states 方法。
            local_data: 通常是一個 DataLoader，提供驗證用的批次數據。
            device: 'cuda' 或 'cpu'。
        輸出：
            predictions: 一個列表，每個元素對應一個樣本，該元素又是一個列表，包含該樣本在模型各層的 DLO 張量。
            sample_label_list: 所有驗證樣本的真實標籤列表。
            num_layers: 模型中被記錄的隱藏層的數量。
        """
        num_layers = None
        sample_label_list = []
        predictions = []
        model.eval()
        model = model.to(device)
        number_of_previous_samples = 0
        for batch_id, batch in enumerate(local_data):
            data, target = batch
            data, target = data.to(device), target.to(device)
            output = model.predict_internal_states(data)
            if num_layers is None:
                num_layers = len(output)
            assert num_layers == len(output)

            for layer_output_index, layer_output_values in enumerate(output):
                for idx in range(target.shape[0]):
                    sample_idx = number_of_previous_samples + idx
                    assert len(predictions) >= sample_idx
                    if len(predictions) == sample_idx:
                        assert layer_output_index == 0
                        predictions.append([])

                    if layer_output_index == 0:
                        expected_predictions = sample_idx + 1
                    else:
                        expected_predictions = number_of_previous_samples + target.shape[0]
                    assert_msg = f'{len(predictions)} vs. {sample_idx} ({idx} {batch_id} '
                    assert_msg += f'{layer_output_index} {number_of_previous_samples})'
                    assert len(predictions) == expected_predictions, assert_msg
                    assert_msg = f'{len(predictions[sample_idx])} {layer_output_index} '
                    assert_msg += f'{sample_idx} {batch_id} {idx} {number_of_previous_samples}'
                    assert len(predictions[sample_idx]) == layer_output_index, assert_msg
                    value = layer_output_values[idx].clone().detach().cpu()
                    predictions[sample_idx].append(value)
            number_of_previous_samples += target.shape[0]
            for t in target:
                sample_label_list.append(t.detach().clone().cpu().item())
        model.cpu()
        return predictions, sample_label_list, num_layers

    @staticmethod
    def __do_predictions(models, global_model, local_data, device):
        """
        Returns
        - The Deep Layer Outputs for all models in a matrix of dimension
          sample x model x layer x value
        - The Deep Layer Outputs of the global model int he dimension sample x layer x value
        - A dict containing lists of sample indices for each label class
        - The number of layers from the model

        用途： 對傳入的所有本地模型列表 (models) 和全局模型 (global_model)，
            分別調用 __predict_for_single_model 來獲取它們在 local_data 上的 DLOs。
        輸出：
            all_models_predictions: 所有本地模型的 DLOs。
            global_model_predictions: 全局模型的 DLOs。
            sample_indices_by_label: 樣本按標籤分類的索引。
            n_layers: 層數。
        """
        all_models_predictions = []
        for model_index, model in enumerate(models):
            predictions, _, _ = CrowdGuardClientValidation.__predict_for_single_model(model,
                                                                                      local_data,
                                                                                      device)
            for sample_index, layer_predictions_for_sample in enumerate(predictions):
                if sample_index >= len(all_models_predictions):
                    assert model_index == 0
                    assert len(all_models_predictions) == sample_index
                    all_models_predictions.append([])
                all_models_predictions[sample_index].append(layer_predictions_for_sample)
        tmp = CrowdGuardClientValidation.__predict_for_single_model(global_model, local_data,
                                                                    device)
        global_model_predictions, sample_label_list, n_layers = tmp
        sample_indices_by_label = {}
        for s_i, label in enumerate(sample_label_list):
            if label not in sample_indices_by_label.keys():
                sample_indices_by_label[label] = []
            sample_indices_by_label[label].append(s_i)

        return all_models_predictions, global_model_predictions, sample_indices_by_label, n_layers

    @staticmethod
    def __prune_poisoned_models(num_layers, total_number_of_clients, own_client_index,
                                distances_by_metric, verbose=False):
        """
        核心功能：執行迭代剪枝和統計檢驗來識別可疑模型（對應論文 Algorithm 2 和 Algorithm 4）。
        輸入：
            num_layers: 模型層數。
            total_number_of_clients: 本輪參與的客戶端總數。
            own_client_index: 當前驗證客戶端自己的模型在列表中的索引。
            distances_by_metric: 一個字典，key 是距離類型 ('cosine', 'euclid')，value 是對應的 HLBIM 矩陣。
            verbose: 是否打印詳細日誌。
        
        """
        detected_poisoned_models = []
        for distance_type in distances_by_metric.keys():

            # First load the distance Matrix for this client and the samples by labels.
            distance_matrix_la_m_l = distances_by_metric[distance_type]

            # We put all of our labels into one big row.
            layer_length = num_layers * len(distance_matrix_la_m_l)
            dist_matrix_m_lcon = [[0.] * layer_length for _ in range(total_number_of_clients)]
            label_count = 0
            for label_x, dist_matrix_m_l_for_label in distance_matrix_la_m_l.items():
                for model_idx, model_values in enumerate(dist_matrix_m_l_for_label):
                    for layer_idx, layer in enumerate(model_values):
                        dist_matrix_m_lcon[model_idx][layer_idx + label_count * num_layers] = layer
                label_count = label_count + 1

            dist_matrix_m_l = dist_matrix_m_lcon

            client_indices = [x for x, value in enumerate(dist_matrix_m_l) if
                              x != own_client_index]
            pruned_indices = []
            has_malicious_model = True
            new_round_needed = True
            prune_idx = 0

            max_pruning_count = int(math.floor((len(dist_matrix_m_l) - 1) / 2))

            while has_malicious_model and new_round_needed:
                # unique
                pruned_indices_local = deepcopy(pruned_indices)
                # Ignore the own label again and the pruned indices
                pruned_cluster_input_m_l = [value for x, value in
                                            enumerate(dist_matrix_m_l) if
                                            x != own_client_index and x not in pruned_indices]
                pruned_client_indices = [x for x, value in
                                         enumerate(dist_matrix_m_l) if
                                         x != own_client_index and x not in pruned_indices]

                if len(pruned_cluster_input_m_l) <= 1:
                    break

                layer_values = {}

                for m in pruned_cluster_input_m_l:
                    for l_i, l in enumerate(m):
                        if l_i not in layer_values.keys():
                            layer_values[l_i] = []
                        layer_values[l_i].append(l)

                median_layer_values = []

                for l_i, l_values in layer_values.items():
                    median_layer_values.append(np.median(l_values).item())

                median_graph = list(median_layer_values)

                pca_list = []
                for m in pruned_cluster_input_m_l:
                    pca_list.append(m)

                pca_list.append(median_graph)

                scaled_data = preprocessing.scale(pca_list)

                pca = PCA()
                pca.fit(scaled_data)
                pca_data = pca.transform(scaled_data)

                cluster_input = []
                cluster_input_plain = []
                pca_one_data = pca_data.T[0]
                for pca_one_value in pca_one_data:
                    cluster_input.append([pca_one_value])
                    cluster_input_plain.append(pca_one_value)

                # Significance tests
                median_val = np.median(cluster_input_plain)
                if verbose:
                    print(f'cluster_input_plain={cluster_input_plain}')
                x_values = []
                y_values = []
                for value in cluster_input_plain:
                    # Split the samples into two groups
                    distance_value = abs(value - median_val)
                    if value >= median_val:
                        x_values.append(distance_value)
                    else:
                        y_values.append(distance_value)
                print(f'Distance: {distance_type}, use y {len(y_values)}: {y_values}')
                print(f'Distance: {distance_type}, use x {len(x_values)}: {x_values}')

                # Statistical tests
                t_value, t_p_value = ttest_ind(x_values, y_values)
                ks_value, ks_p_value = kstest(x_values, y_values)
                barlett_value, bartlett_p_value = levene(x_values, y_values)

                # Outlier tests
                # Creating boxplot
                bp_result = plt.boxplot(cluster_input_plain, whis=5.5)
                fliers = bp_result['fliers'][0].get_ydata()
                outlier_boxplot = len(fliers)
                plt.close()

                # Outlier based on variance
                deviation_mean = np.mean(cluster_input_plain)
                deviation_std = abs(np.std(cluster_input_plain))

                max_dist_rule_factor = 0
                for cip in cluster_input_plain:
                    cip_abs = abs(cip - deviation_mean)
                    rule_factor = cip_abs / deviation_std
                    if max_dist_rule_factor < rule_factor:
                        max_dist_rule_factor = rule_factor

                outlier_three_sigma = max_dist_rule_factor

                has_malicious_model_t_threshold = False
                if t_p_value < 0.01:
                    has_malicious_model_t_threshold = True
                has_malicious_model_ks_threshold = False
                if ks_p_value < 0.01:
                    has_malicious_model_ks_threshold = True
                has_malicious_model_bartlett_threshold = False
                if bartlett_p_value < 0.01:
                    has_malicious_model_bartlett_threshold = True

                has_boxplot_outlier = False
                has_three_sigma_outlier = False

                if outlier_boxplot > 0:
                    has_boxplot_outlier = True
                if outlier_three_sigma >= 3:
                    has_three_sigma_outlier = True

                # Choose exit criterium
                has_malicious_model = (has_malicious_model_t_threshold
                                       or has_malicious_model_ks_threshold
                                       or has_malicious_model_bartlett_threshold
                                       or has_boxplot_outlier
                                       or has_three_sigma_outlier)

                ac_e = AgglomerativeClustering(n_clusters=2, distance_threshold=None,
                                               compute_full_tree=True,
                                               metric="euclidean", memory=None,
                                               connectivity=None,
                                               linkage='single',
                                               compute_distances=True).fit(cluster_input)
                ac_e_labels: list = ac_e.labels_.tolist()
                median_value_cluster_label = ac_e_labels[-1]
                ac_e_malicious_class_indices = [idx for idx, val in enumerate(ac_e_labels) if
                                                val != median_value_cluster_label]

                for m_j, value in enumerate(pruned_client_indices):
                    if m_j in ac_e_malicious_class_indices:
                        pruned_indices_local.append(value)

                pruned_indices_local = list(set(pruned_indices_local))

                # If we now prune more than half, we stop and remove the best items from the last
                # pruning list.
                pruned_too_much = True
                if len(pruned_indices_local) > max_pruning_count:
                    dist_values_of_pruned_models = []
                    for midx in ac_e_malicious_class_indices:
                        dist_to_median = abs(cluster_input[midx][0] - cluster_input[-1][0])
                        dist_values_of_pruned_models.append(dist_to_median)

                    sorted_dist_values_of_pruned_models = list(dist_values_of_pruned_models)
                    sorted_dist_values_of_pruned_models.sort()

                    sorted_ac_e_malicious_class_indices = []
                    for sdv in sorted_dist_values_of_pruned_models:
                        dvidx = dist_values_of_pruned_models.index(sdv)
                        for m_j, value in enumerate(pruned_client_indices):
                            if m_j == ac_e_malicious_class_indices[dvidx]:
                                sorted_ac_e_malicious_class_indices.append(value)
                    overflowed_count = len(pruned_indices_local) - max_pruning_count
                    for oc in range(overflowed_count):
                        # Get the values of the clusters and remove the nearest ones
                        # from pruned_indices_local
                        pruned_indices_local.remove(sorted_ac_e_malicious_class_indices[-1])
                        del sorted_ac_e_malicious_class_indices[-1]
                    pruned_too_much = False

                still_pruning = len(pruned_indices) < len(pruned_indices_local)
                new_round_needed = still_pruning and pruned_too_much
                if has_malicious_model and new_round_needed:
                    pruned_indices = pruned_indices_local

                prune_idx += 1

            # Analyze the voting
            for _, value in enumerate(client_indices):
                if value in pruned_indices:
                    detected_poisoned_models.append(value)

        return list(set(detected_poisoned_models))

    @staticmethod
    def validate_models(global_model, models, own_client_index, local_data, device):
        """
        用途：這是客戶端驗證的入口函式。
        步驟：
            調用 __do_predictions 獲取所有模型（本地和全局）在當前客戶端本地數據上的 DLOs。
            對於每種距離度量（Cosine 和 Euclidean）：
                a. 調用 __distance_global_model_final_metric 計算 HLBIM 矩陣。
            調用 __prune_poisoned_models 根據計算出的 HLBIM 矩陣來識別可疑模型。
        輸出： 
            一個列表，包含被識別為可疑的本地模型的索引。
        """
        tmp = CrowdGuardClientValidation.__do_predictions(models, global_model, local_data, device)
        prediction_matrix, global_model_predictions, sample_indices_by_label, num_layers = tmp
        distances_by_metric = {}
        for dist_type in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN]:
            calculated_distances = CrowdGuardClientValidation.__distance_global_model_final_metric(
                dist_type,
                prediction_matrix,
                global_model_predictions,
                sample_indices_by_label,
                own_client_index)
            distances_by_metric[dist_type] = calculated_distances
        result = CrowdGuardClientValidation.__prune_poisoned_models(num_layers, len(models),
                                                                    own_client_index,
                                                                    distances_by_metric)
        return result
