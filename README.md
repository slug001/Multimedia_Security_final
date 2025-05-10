# Multimedia_Security_final: 聯邦學習中的分布式後門攻防研究

**組員:**
*   R13922119
*   R13944069

---

## 專案概述 (Project Overview)

## Related work
### **DBA: Distributed Backdoor Attacks against Federated Learning**

*   **簡介:** 分布式後門攻擊 (DBA) 是一種針對聯邦學習的先進後門攻擊策略。其核心思想是將一個設計好的全局觸發器 (Global Trigger) 分解成多個局部模式 (Local Patterns)。這些局部模式被分配給不同的、相互串通的惡意客戶端。每個惡意客戶端僅在其本地訓練數據中植入分配給它的特定局部模式，並參與聯邦學習的訓練過程。
*   **LINK:** https://openreview.net/forum?id=rkgyS0VFvr

### **BayBFed: Bayesian Backdoor Defense for Federated Learning**
*  **BayBFed: 用於聯邦學習的貝葉斯後門防禦**
*  **簡介:**  利用客戶端更新的概率分佈來檢測 FL 中的惡意更新。BayBFed 計算客戶端更新的概率度量，該算法可以利用這種概率度量來有效地檢測和過濾掉惡意更新。  
*  第一階段：後驗計算 (Posterior Computation)： 利用分層 Beta-Bernoulli 過程 (Hierarchical Beta-Bernoulli Process, HBBP) 來為每個客戶端的模型更新（權重）計算一個概率度量（具體來說是 Beta 後驗分佈的參數）。這個過程能夠追蹤客戶端更新中的變化。  
*  第二階段：檢測與過濾 (Detection and Filtering)： 採用中國餐館過程 (Chinese Restaurant Process, CRP) 的一個改編版本，作者稱之為 CRP-Jensen。該模塊使用第一階段計算出的概率度量作為輸入，通過計算客戶端更新分佈與現有簇（良性更新組）分佈之間的 Jensen-Shannon 散度 (Jensen-Divergence, JD) 來區分惡意更新和良性更新。惡意更新通常會表現出與良性簇顯著不同的 JD 值。  

