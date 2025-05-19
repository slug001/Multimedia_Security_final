# Multimedia_Security_final: 聯邦學習中的分布式後門攻防研究

**組員:**
*   R13922119
*   R13944069

---

## 專案概述 (Project Overview)

## Related work
### **攻擊**
### **BACKDOOR FEDERATED LEARNING BY POISONINGBACKDOOR-CRITICAL LAYERS (ICLR 2024)**
*   **簡介:**
>現有的 FL 攻擊和防禦方法通常著眼於整個模型，卻忽略了一個現象：後門關鍵層 (Backdoor-Critical, BC) layers 的存在——即模型中一小部分主導其漏洞的層。攻擊這些 BC layers 可以達到與攻擊整個模型相當的效果，但被最先進 (SOTA) 防禦機制檢測到的機率卻遠低得多。   
>
>本文提出了一種通用的層替換分析 (Layer Substitution Analysis) 方法，從攻擊者的角度識別和驗證 BC layers。基於識別出的 BC layers，作者精心設計了一種新的後門攻擊方法，該方法能在各種防禦策略下自適應地尋求攻擊效果和隱蔽性之間的平衡。大量實驗表明，作者提出的 BC layer 感知後門攻擊方法，即使只有 10% 的惡意客戶端，也能在七種 SOTA 防禦機制下成功植入後門，並且其性能優於最新的後門攻擊方法。
*   **LINK:** https://arxiv.org/pdf/2308.04466
*   **CODE:** https://github.com/zhmzm/Poisoning_Backdoor-critical_Layers_Attack

### **防禦**
### **CrowdGuard: Federated Backdoor Detection inFederated Learning (NDSS 2024)**
*   **簡介:**
>現有的 FL 防護措施有各種局限性：它們可能僅限於特定的數據分佈，或者由於排除良性模型或添加噪聲而降低全局模型的準確性，容易受到自適應的、了解防禦機制的對手攻擊，或者需要服務器訪問本地模型，從而允許數據推斷攻擊。
>
>本文提出了一種名為 CrowdGuard 的新型防禦機制，它有效地減輕了 FL 中的後門攻擊，並克服了現有技術的不足。它利用客戶端對各個模型的反饋，分析隱藏層中神經元的行為，並通過迭代剪枝方案消除中毒模型。CrowdGuard 採用位於服務器的堆疊式聚類方案來增強其對惡意客戶端反饋的韌性。評>估結果表明，CrowdGuard 在各種情境（包括 IID 和 non-IID 數據分佈）下均達到了 100% 的真陽性率和真陰性率。此外，CrowdGuard 能夠抵禦自適應攻擊者，同時保持受保護模型的原始性能。為確保機密性，CrowdGuard 在客戶端和服務器端都利用可信執行環境 (TEEs) 構建了一個安全且保護>隱私的架構。
*   **LINK:** https://www.ndss-symposium.org/wp-content/uploads/2024-233-paper.pdf
*   **CODE:** https://github.com/trust-tuda/crowdguard?tab=readme-ov-file

---
### **其他論文**
### **DBA: Distributed Backdoor Attacks against Federated Learning**

*   **簡介:** 分布式後門攻擊 (DBA) 是一種針對聯邦學習的先進後門攻擊策略。其核心思想是將一個設計好的全局觸發器 (Global Trigger) 分解成多個局部模式 (Local Patterns)。這些局部模式被分配給不同的、相互串通的惡意客戶端。每個惡意客戶端僅在其本地訓練數據中植入分配給它的特定局部模式，並參與聯邦學習的訓練過程。
*   **LINK:** https://openreview.net/forum?id=rkgyS0VFvr

### **BayBFed: Bayesian Backdoor Defense for Federated Learning**
*  **BayBFed: 用於聯邦學習的貝葉斯後門防禦**
*  **簡介:**  利用客戶端更新的概率分佈來檢測 FL 中的惡意更新。BayBFed 計算客戶端更新的概率度量，該算法可以利用這種概率度量來有效地檢測和過濾掉惡意更新。  
*  第一階段：後驗計算 (Posterior Computation)： 利用分層 Beta-Bernoulli 過程 (Hierarchical Beta-Bernoulli Process, HBBP) 來為每個客戶端的模型更新（權重）計算一個概率度量（具體來說是 Beta 後驗分佈的參數）。這個過程能夠追蹤客戶端更新中的變化。  
*  第二階段：檢測與過濾 (Detection and Filtering)： 採用中國餐館過程 (Chinese Restaurant Process, CRP) 的一個改編版本，作者稱之為 CRP-Jensen。該模塊使用第一階段計算出的概率度量作為輸入，通過計算客戶端更新分佈與現有簇（良性更新組）分佈之間的 Jensen-Shannon 散度 (Jensen-Divergence, JD) 來區分惡意更新和良性更新。惡意更新通常會表現出與良性簇顯著不同的 JD 值。
*  **LINK** https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10179362  

