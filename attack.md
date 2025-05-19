# 論文筆記：透過下毒後門關鍵層實現聯邦學習後門攻擊 (ICLR 2024)

這篇筆記整理了 ICLR 2024 的論文「BACKDOOR FEDERATED LEARNING BY POISONING BACKDOOR-CRITICAL LAYERS」。

## 論文基本資訊

*   **論文標題 (英文)：** BACKDOOR FEDERATED LEARNING BY POISONING BACKDOOR-CRITICAL LAYERS
*   **論文標題 (中文翻譯)：** 透過下毒後門關鍵層實現聯邦學習後門攻擊
*   **發表會議：** ICLR 2024
*   **論文連結：** https://arxiv.org/pdf/2308.04466

## 摘要

> 聯邦學習 (Federated Learning, FL) 已被廣泛部署，以在分散式設備上對敏感數據進行機器學習訓練，同時保護數據隱私。然而，FL 的去中心化學習範式和異質性進一步擴大了後門攻擊的攻擊面。現有的 FL 攻擊和防禦方法通常著眼於整個模型，卻忽略了一個現象：**後門關鍵層 (Backdoor-Critical, BC) layers** 的存在——即模型中一小部分主導其漏洞的層。攻擊這些 BC layers 可以達到與攻擊整個模型相當的效果，但被最先進 (SOTA) 防禦機制檢測到的機率卻遠低得多。
>
> 本文提出了一種通用的**層替換分析 (Layer Substitution Analysis)** 方法，從攻擊者的角度識別和驗證 BC layers。基於識別出的 BC layers，作者精心設計了一種新的後門攻擊方法，該方法能在各種防禦策略下自適應地尋求攻擊效果和隱蔽性之間的平衡。大量實驗表明，作者提出的 BC layer 感知後門攻擊方法，即使只有 10% 的惡意客戶端，也能在七種 SOTA 防禦機制下成功植入後門，並且其性能優於最新的後門攻擊方法。

## 核心技術詳細說明

### 1. 後門關鍵層 (Backdoor-Critical Layers, BC Layers)

*   **核心概念：** 模型中並非所有層對後門的形成都同等重要。存在一個小的層子集 (BC layers)，它們的權重變化對後門任務的成功率 (Backdoor Success Rate, BSR) 有著不成比例的巨大影響。
*   **直觀理解：** 較淺的層學習低級特徵（如邊緣），較深的層學習更複雜的概念。後門任務（將觸發器與目標標籤關聯）更可能嵌入到能編碼複雜概念的較深層。例如，論文中 `fc1.weight` (全連接層權重) 可能是一個 BC layer。

### 2. 層替換分析 (Layer Substitution Analysis, LSA) – 識別 BC Layers

這是一個由攻擊者（惡意客戶端）在其本地執行的三步驟過程 (參考論文 Figure 2)：

1.  **本地訓練 (Local Training)**
    *   惡意客戶端收到全局模型 `w`。
    *   在乾淨數據集 `D_clean,train` 上訓練 `w`，得到「良性本地模型」`w_benign`。
    *   再將 `w_benign` 在帶觸發器的「中毒數據集」`D_poison,train` 上訓練，得到已學會後門的「惡意本地模型」`w_malicious`。

2.  **前向層替換 (Forward Layer Substitution)**
    *   **目標：** 找出從 `w_malicious` 中移除後，BSR 下降最多的層。
    *   **過程：** 迭代地將 `w_malicious` 中的每一層 `l` 替換為 `w_benign` 中對應的層 `l`，得到混合模型 `w_b2m(l)`。
    *   **評估：** 計算 `ΔBSR_b2m(l) = BSR_malicious - BSR_b2m(l)`。
    *   **排序：** 根據 `ΔBSR_b2m(l)` 從高到低對所有層排序。

3.  **後向層替換 (Backward Layer Substitution)**
    *   **目標：** 確認一組最小的 BC layers `L*`，將它們從 `w_malicious` 植入 `w_benign` 後能達到足夠高的 BSR。
    *   **過程：** 從 `w_benign` 開始，按步驟 2 的順序，逐步將 `w_malicious` 中的層複製到 `w_benign` 中，形成 `w_m2b(L*)`。
    *   **停止條件：** 當 `w_m2b(L*)` 的 BSR 達到預設閾值 (例如 `τ * BSR_malicious`) 時停止。此時 `L*` 即為 BC layers。

### 3. 基於 BC Layers 的後門攻擊方法

#### a. 逐層下毒 (Layer-wise Poisoning, LP) 攻擊

*   **目標防禦：** 基於距離 (如 FLTrust, MultiKrum) 和基於反演的防禦。
*   **核心思想：** 僅在 BC layers (`L*`) 中植入惡意權重，非 BC layers 則使其權重接近良性模型，以減少異常性。
*   **實現 (參考 Equation 4)：**
    ```
    ῶ^(i) = λ ⋅ v ο u_malicious + ReLU(1 - λ) ⋅ v ο u_average + (1 - v) ο u_average
    ```
    其中：
    *   `ῶ^(i)`: 惡意客戶端 `i` 上傳的更新。
    *   `v`: BC layer 選擇向量。
    *   `u_malicious`: 本地惡意模型的層表示。
    *   `u_average`: 估計的「良性客戶端本地模型的平均層表示」。
    *   `λ`: 隱蔽性控制超參數。
    *   `ο`: 逐元素相乘。
*   **自適應層控制：** 若模擬被拒絕，可減少 `L*` 的大小。

#### b. 逐層翻轉 (Layer-wise Flipping, LF) 攻擊

*   **目標防禦：** 基於符號 (如 RLR) 的防禦。
*   **核心思想：** 預先主動翻轉 BC layers (`L*`) 中參數的更新方向。當 RLR 防禦試圖「修正」時，反而會將它們翻轉回攻擊者期望的惡意狀態。
*   **實現 (參考論文 §4.2 公式)：**
    ```
    ῶ^(i)_LFA = -(w_m2b(L*) - w) + w
    ```
    其中 `w` 是當前全局模型，`w_m2b(L*)` 是包含惡意 BC 層的目標模型。

## 實作部分 (Implementation Details)

*   **實驗環境：**
    *   框架：PyTorch
    *   硬體：NVIDIA RTX A5000 GPU
    *   FL 設置：默認 100 客戶端，10% 惡意，每輪選 10% 訓練。
    *   數據集：Fashion-MNIST, CIFAR-10
    *   模型：五層 CNN (Table A-5), ResNet18, VGG19
    *   數據分佈：默認 non-IID (q=0.5)
*   **超參數範例：**
    *   LSA 識別閾值 `τ = 0.95`
    *   LP 攻擊 `λ`: CIFAR-10 為 1, Fashion-MNIST 為 0.5
*   **評估指標：**
    *   `Acc`: 主任務準確率
    *   `BSR`: 後門成功率
    *   `BAR`: 良性客戶端接受率
    *   `MAR`: 惡意客戶端接受率 (越高越隱蔽)
*   **對比的防禦機制：**
    *   距離型：FLTrust, FLAME, MultiKrum
    *   符號型：RLR
    *   其他：FLDetector, FLARE
*   **對比的攻擊方法：**
    *   BadNets (基線)
    *   DBA (Distributed Backdoor Attacks)
    *   SRA (Subnet Replacement Attack)
    *   Scaling Attack
    *   Constrain Loss Attack

## 實驗結果亮點

1.  **隱蔽性 (Table 1)：** LP 攻擊在 MultiKrum 和 FLAME 防禦下均能實現非常高的 MAR。
2.  **有效性 (Table 2)：** 在多種 SOTA 防禦下，LP 攻擊均能實現高 BSR，同時保持較好的主任務 Acc。LF 攻擊在 RLR 防禦下表現出色。
3.  **BC Layers 的重要性 (Ablation Study, Table 3, Figure 7)：** 針對 LSA 識別出的 BC layers 進行攻擊，其 BSR 遠高於隨機選擇非 BC layers 進行攻擊。
4.  **參數敏感性分析 (Figures 4, 5, 6)：** 展示了 `τ`, `λ` 和 BC layer 識別間隔對攻擊效果的影響。

## 總結

這篇論文提出，後門攻擊的有效性主要集中在少數「後門關鍵層」(BC layers)。基於此，他們設計了識別這些層的方法 (LSA) 以及兩種針對性的攻擊策略 (LP 和 LF)。實驗證明這些方法在保持攻擊效果的同時，顯著提高了對現有聯邦學習防禦機制的隱蔽性。
