# 40Q+ Benchmark Molecular Systems Research Report

> **Date**: 2026-03-09
> **Purpose**: 為 Flow-Guided Krylov Pipeline (AR-NF + SKQD) 選擇 40+ qubit 規模的基準分子系統
> **Method**: 基於 2024-2026 年文獻的系統調研

---

## 1. Executive Summary

本報告調研了 2024-2026 年量子化學領域的基準分子系統，為我們的 AR-NF + SKQD pipeline 建立從 20Q 到 50Q+ 的漸進式基準階梯。核心發現：

1. **IBM 的 de facto 標準**：N2/cc-pVDZ CAS(10,26) = 52 qubits，已被多篇論文採用
2. **CISD 覆蓋率在 40Q 急劇下降**：CAS(10,20) 時 CISD 僅覆蓋 0.003% 配置空間，NF 的價值在此顯現
3. **過渡金屬系統**是最具化學意義的 benchmark：[2Fe-2S]、Cr2、Fe-porphyrin
4. **PySCF 可行性確認**：所有推薦系統均可用 PySCF CASSCF 設定

---

## 2. 文獻調研：各研究團隊使用的基準系統

### 2.1 IBM SQD/QSCI/SqDRIFT 系列

| 論文 | 分子 | Active Space | Qubits | 方法 |
|------|------|-------------|--------|------|
| Robledo-Moreno et al., Science Advances 2025 | N2/cc-pVDZ | CAS(10,~26) frozen-core | 58 | SQD |
| Robledo-Moreno et al., Science Advances 2025 | [2Fe-2S] | CAS(30,20) | 45 | SQD |
| Robledo-Moreno et al., Science Advances 2025 | [4Fe-4S] | CAS(54,36) | 77 | SQD |
| IBM+Lockheed, JCTC 2025 | CH2/cc-pVDZ | CAS(6,23) | 52 | SQD |
| IBM SqDRIFT, Science 2026 | half-Mobius C13Cl2 | 32 electrons | 72-100 | SqDRIFT |
| SqDRIFT (arXiv 2508.02578) | coronene | CAS(24,24) | 48 | SqDRIFT |
| SqDRIFT (arXiv 2508.02578) | naphthalene | pi-space | ~20 | SqDRIFT |
| Cuprate chain (arXiv 2512.04962) | CuO plaquettes | variable | variable | SQD |
| SKQD (arXiv 2501.09702) | Anderson impurity | 42e,42o | 85 | SKQD |

**關鍵觀察**：
- IBM 統一使用 cc-pVDZ basis set + frozen-core 近似
- N2 和 [2Fe-2S] 是其核心 benchmark
- 最新 SqDRIFT 論文已推進到 coronene (48Q) 和 Anderson model (85Q)

### 2.2 QiankunNet (Nature Communications 2025)

| 分子 | Active Space | 備註 |
|------|-------------|------|
| H2O/cc-pVDZ | CAS(10,24) | 48 spin orbitals, 驗證 transformer 架構 |
| N2 | 大 active space | FCI-level accuracy |
| 多分子 benchmark | up to 30 spin orbitals | 99.9% FCI 精度 |

**架構**: decoder-only transformer + MCTS sampling + batched autoregressive。
**與我們的關聯**: 我們的 Phase 4a AR transformer 直接受此啟發。

### 2.3 NNQS-SCI (SC24, Tang et al.)

| 分子 | Active Space | Hilbert Space | 備註 |
|------|-------------|--------------|------|
| N2/large basis | up to 76 spin orbitals | ~1.4 x 10^12 | 首次 NNQS 達到 FCI 精度 |
| Cr2/large basis | up to 76 spin orbitals | — | 首次 NNQS 精確計算 Cr2 |
| 多系統 | up to 152 spin orbitals | >10^14 | 極限規模 |

**關鍵成就**：
- N2 ground state: -109.278862 Ha
- Cr2 ground state: -2086.400429 Ha
- Adaptive SCI + Transformer decoder，memory 壓縮 90%

### 2.4 AB-SND (arXiv 2508.12724, August 2025)

自適應基底 + 樣本化神經對角化。使用 autoregressive NN 採樣 + 參數化基底變換。
在原始計算基底中 ground state 不集中時特別有效。

### 2.5 SandboxAQ GPU-DMRG (JCTC 2025)

| 系統 | CAS 大小 | 備註 |
|------|---------|------|
| PAH (acenes) | up to CAS(82,82) orbital-optimized | DGX-H100 |
| Iron-sulfur complexes | CAS(37,30) to CAS(82,82) | 最大規模 benchmark |
| Cr2 | various | 標準 benchmark |

**重要**: SandboxAQ 證明 GPU-DMRG 可以在數天內完成 CAS(82,82)。
這意味著 CAS(30,20) 對 DMRG 而言是 "trivially classical"。
我們的 pipeline 需要瞄準 DMRG 尚需大量計算的區域 (CAS(40+, 30+))。

### 2.6 SHCI Benchmarks (Sharma group)

| 系統 | Active Space | 備註 |
|------|-------------|------|
| Cr2/cc-pVDZ-DK | (28e, 76o) | relativistic, 標準 benchmark |
| Mn-Salen | (28e, 22o) | 過渡金屬配合物 |
| F2 | (14e, 108o) | 大 active space |
| butadiene | (22e, 82o) | 共軛系統 |

### 2.7 Chemically Decisive Benchmarks (arXiv 2601.10813, Jan 2026)

這篇論文提出了專門為量子效用設計的分子基準階梯：

| 分子 | 類型 | 化學意義 |
|------|------|---------|
| N2 | 多參考鍵斷裂 | triple bond dissociation |
| FeS | 高自旋過渡金屬 | 自旋態能量學 |
| [2Fe-2S] | 生物相關鐵硫簇 | 抗鐵磁耦合 |
| U2 | 錒系元素鍵 | 極端多參考 + 相對論 |

### 2.8 Reinholdt QSCI 批判 (JCTC 2025)

**關鍵發現**：QSCI 比古典 SCI 需要多一個數量級的 determinants。
- HCI 用 4,260 determinants 達到 0.1 Ha 精度
- QSCI 需要 50,551 determinants 達到相同精度
- 這強化了我們使用 SKQD (而非 SQD) 的決策正確性

---

## 3. 推薦基準階梯

### CISD 覆蓋率分析

以下數據展示為何 NF 在大 active space 中不可或缺：

| System | CISD Configs | Total Configs | CISD Coverage |
|--------|-------------|---------------|---------------|
| N2/STO-3G CAS(14,10) 20Q | 610 | 14,400 | 4.236% |
| N2/cc-pVDZ CAS(10,10) 20Q | 876 | 63,504 | 1.379% |
| N2/cc-pVDZ CAS(10,12) 24Q | 1,716 | 627,264 | 0.274% |
| N2/cc-pVDZ CAS(10,15) 30Q | 3,501 | 9,018,009 | 0.039% |
| N2/cc-pVDZ CAS(10,20) 40Q | 7,876 | 240,374,016 | **0.003%** |
| N2/cc-pVDZ CAS(10,26) 52Q | 15,436 | 4,327,008,400 | **0.0000%** |
| Cr2 CAS(12,20) 40Q | 9,955 | 1,502,337,600 | **0.001%** |
| [2Fe-2S] CAS(22,20) 40Q | 13,960 | 28,210,561,600 | **0.000%** |

**結論**：在 40Q 時，CISD 僅覆蓋 ~0.003% 的配置空間。Ground state wavefunction 中的 triples/quadruples 貢獻顯著，NF 必須能夠找到這些配置。

---

### Step 1: ~24 Qubits (可立即執行)

#### 1a. N2/cc-pVDZ CAS(10,12) — 24 qubits

| Item | Value |
|------|-------|
| **分子** | N2 (氮分子) |
| **幾何結構** | R(N-N) = 1.10 A (equilibrium) |
| **基底組** | cc-pVDZ |
| **Active Space** | CAS(10,12): 10 electrons, 12 orbitals (frozen 1s core) |
| **Spin Orbitals / Qubits** | 24 |
| **配置數** | 627,264 |
| **CISD 覆蓋率** | 0.274% |
| **對角化策略** | NF+SKQD (truncate to ~15K basis) |
| **參考能量** | DMRG/SHCI available |
| **為何好** | 從現有 CAS(10,10) 自然延伸；IBM benchmark 的縮小版；已有大量文獻參考 |
| **使用此系統的論文** | IBM SQD tutorial, 多篇 DMRG benchmark |

**PySCF 設定**:
```python
from pyscf import gto, scf, mcscf
mol = gto.M(atom='N 0 0 0; N 0 0 1.10', basis='cc-pvdz', symmetry=True)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 12, 10)  # 12 orbitals, 10 electrons
mc.kernel()
```

#### 1b. Cr2/STO-3G CAS(12,12) — 24 qubits

| Item | Value |
|------|-------|
| **分子** | Cr2 (鉻二聚體) |
| **幾何結構** | R(Cr-Cr) = 1.68 A (equilibrium) |
| **基底組** | STO-3G |
| **Active Space** | CAS(12,12): 12 electrons (3d + 4s), 12 orbitals |
| **Spin Orbitals / Qubits** | 24 |
| **配置數** | 853,776 |
| **CISD 覆蓋率** | 0.213% |
| **對角化策略** | NF+SKQD |
| **參考能量** | DMRG available |
| **為何好** | 經典多參考 benchmark；3d-3d 金屬鍵；STO-3G 足以測試方法 |
| **使用此系統的論文** | JACS 2022 "Closing a Chapter", NNQS-SCI, DMRG-DSRG 2025 |

**PySCF 設定**:
```python
mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.68', basis='sto-3g', spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 12, 12)  # 3d+4s orbitals
mc.kernel()
```

**注意**: Cr2 的 SCF 收斂困難，可能需要 `mf.max_cycle = 200` 和 `mc.natorb = True`。

---

### Step 2: ~30 Qubits (挑戰性)

#### 2a. N2/cc-pVDZ CAS(10,15) — 30 qubits

| Item | Value |
|------|-------|
| **分子** | N2 |
| **幾何結構** | R(N-N) = 1.10 A |
| **基底組** | cc-pVDZ |
| **Active Space** | CAS(10,15): 10 electrons, 15 orbitals |
| **Spin Orbitals / Qubits** | 30 |
| **配置數** | 9,018,009 |
| **CISD 覆蓋率** | 0.039% |
| **對角化策略** | NF + importance truncation to ~15K |
| **Sparse H 記憶體** | ~13.4 GB (fits in 128GB UMA) |
| **為何好** | IBM CAS(10,26) 的中間站；CISD 已不足 (0.039%) |
| **使用此系統的論文** | DMRG-TCC, QiankunNet |

#### 2b. benzene/STO-3G CAS(6,15) — 30 qubits

| Item | Value |
|------|-------|
| **分子** | C6H6 (苯) |
| **幾何結構** | D6h, R(C-C) = 1.40 A, R(C-H) = 1.08 A |
| **基底組** | STO-3G |
| **Active Space** | CAS(6,15): 6 pi electrons, 15 orbitals (pi + sigma correlation) |
| **Spin Orbitals / Qubits** | 30 |
| **配置數** | 207,025 |
| **CISD 覆蓋率** | 0.853% |
| **對角化策略** | NF+SKQD |
| **為何好** | 經典有機化學 benchmark；pi 電子關聯；文獻豐富 |
| **使用此系統的論文** | QPE for benzene (PCCP 2024), ALCI, SandboxAQ |

---

### Step 3: ~40 Qubits (目標)

#### 3a. N2/cc-pVDZ CAS(10,20) — 40 qubits [PRIMARY TARGET]

| Item | Value |
|------|-------|
| **分子** | N2 |
| **幾何結構** | R(N-N) = 1.10 A (+ stretched 1.50, 2.00, 2.50, 3.00 A) |
| **基底組** | cc-pVDZ |
| **Active Space** | CAS(10,20): 10 electrons, 20 orbitals |
| **Spin Orbitals / Qubits** | 40 |
| **配置數** | 240,374,016 (~2.4 x 10^8) |
| **CISD 覆蓋率** | **0.003%** |
| **對角化策略** | NF MUST — truncate to ~15K diag basis |
| **Sparse H 記憶體** | ~358 GB (exceeds 128GB UMA — need streaming) |
| **參考能量** | SHCI/DMRG extrapolated FCI available |
| **為何好** | 直接與 IBM benchmark 可比；triple bond 斷裂是經典強關聯問題；CISD 完全不足 |
| **使用此系統的論文** | IBM SQD (Science Advances 2025), NNQS-SCI (SC24) |

**技術挑戰**:
- 需要 AR-NF 產生 triples + quadruples (CISD 僅 0.003%)
- Sparse H construction 需要 streaming (358 GB > 128 GB)
- SKQD diag basis 上限 15K — NF + importance ranking 必須精準

#### 3b. Cr2/cc-pVDZ CAS(12,20) — 40 qubits

| Item | Value |
|------|-------|
| **分子** | Cr2 |
| **幾何結構** | R(Cr-Cr) = 1.68 A |
| **基底組** | cc-pVDZ (or Ahlrichs SVP) |
| **Active Space** | CAS(12,20): 12 electrons (3d6×2), 20 orbitals |
| **Spin Orbitals / Qubits** | 40 |
| **配置數** | 1,502,337,600 (~1.5 x 10^9) |
| **CISD 覆蓋率** | **0.001%** |
| **對角化策略** | NF MUST |
| **參考能量** | DMRG (M=28000, JACS 2022), SHCI (28e/76o) |
| **為何好** | 量子化學最困難的 benchmark 之一；六重鍵；極端多參考 |
| **使用此系統的論文** | JACS 2022, NNQS-SCI (SC24), DMRG-DSRG 2025 |

**注意**: Cr2 是 "closing a chapter" 級別的困難系統。即使 DMRG 也需要 M=28000 (bond dimension) 才能收斂。
我們的 NF+SKQD 在此系統上如果能達到 chemical accuracy (1.6 mHa)，將是有力的成果。

#### 3c. [2Fe-2S] CAS(22,20) — 40 qubits [SECONDARY TARGET]

| Item | Value |
|------|-------|
| **分子** | [Fe2S2(SCH3)4]^2- (methyl-capped iron-sulfur cluster) |
| **幾何結構** | 見 IBM 論文 supplementary |
| **基底組** | STO-3G or cc-pVDZ (with ECP for Fe) |
| **Active Space** | CAS(22,20): Fe 3d electrons + bridging S 3p |
| **Spin Orbitals / Qubits** | 40 |
| **配置數** | 28,210,561,600 (~2.8 x 10^10) |
| **CISD 覆蓋率** | **0.000%** |
| **對角化策略** | NF MUST + aggressive truncation |
| **參考能量** | DMRG-CASSCF available |
| **為何好** | 生物學相關 (鐵硫蛋白)；IBM 核心 benchmark；抗鐵磁耦合 |
| **使用此系統的論文** | IBM SQD (Science Advances 2025), FCIQMC-CASSCF 2021 |

**PySCF 注意**:
- Fe 需要 ECP (effective core potential)，如 `cc-pVDZ-PP` 或 Stuttgart ECP
- 抗鐵磁態需要 broken-symmetry initial guess
- active space 選擇需要包含 Fe 3d + bridging S 3p

#### 3d. Fe-porphyrin CAS(8,20) — 40 qubits

| Item | Value |
|------|-------|
| **分子** | Fe(II)-porphine (model porphyrin) |
| **幾何結構** | D4h, standard from literature |
| **基底組** | cc-pVDZ (Fe with ECP) |
| **Active Space** | CAS(8,20): Fe 3d^6 + porphyrin pi |
| **Spin Orbitals / Qubits** | 40 |
| **配置數** | 23,474,025 (~2.3 x 10^7) |
| **CISD 覆蓋率** | 0.024% |
| **對角化策略** | NF + importance truncation |
| **參考能量** | DMRG(34,35), HCISCF(44,44) |
| **為何好** | 自旋交叉 (spin crossover)；生物化學核心問題 |
| **使用此系統的論文** | HCISCF (JCTC 2017), FCIQMC 2021, MC-PDFT |

---

### Step 4: ~50 Qubits (雄心目標)

#### 4a. N2/cc-pVDZ CAS(10,26) — 52 qubits [IBM STANDARD]

| Item | Value |
|------|-------|
| **分子** | N2 |
| **幾何結構** | R(N-N) = 1.10 A (and stretched) |
| **基底組** | cc-pVDZ, frozen 1s core |
| **Active Space** | CAS(10,26): all non-core orbitals |
| **Spin Orbitals / Qubits** | 52 |
| **配置數** | 4,327,008,400 (~4.3 x 10^9) |
| **CISD 覆蓋率** | **0.0000%** |
| **對角化策略** | NF MUST |
| **參考能量** | SCI: -109.22802921665716 Ha (IBM tutorial) |
| **為何好** | IBM de facto 標準 benchmark；直接可比較結果 |
| **使用此系統的論文** | IBM SQD (Science Advances 2025), SKQD (arXiv 2501.09702), Reinholdt JCTC 2025 |

#### 4b. coronene CAS(24,24) — 48 qubits

| Item | Value |
|------|-------|
| **分子** | C24H12 (coronene, 冠烯) |
| **幾何結構** | D6h, 標準文獻幾何 |
| **基底組** | STO-3G (pi orbitals only) |
| **Active Space** | CAS(24,24): 24 pi electrons, 24 pi orbitals |
| **Spin Orbitals / Qubits** | 48 |
| **配置數** | 7,312,459,672,336 (~7.3 x 10^12) |
| **CISD 覆蓋率** | ~0% |
| **對角化策略** | NF MUST |
| **參考能量** | DMRG extrapolated |
| **為何好** | SqDRIFT 論文的核心 benchmark；PAH 系統 |
| **使用此系統的論文** | SqDRIFT (arXiv 2508.02578), SandboxAQ GPU-DMRG |

#### 4c. CH2/cc-pVDZ CAS(6,23) — 52 qubits

| Item | Value |
|------|-------|
| **分子** | CH2 (methylene) |
| **幾何結構** | 標準 (singlet + triplet) |
| **基底組** | cc-pVDZ |
| **Active Space** | CAS(6,23): 6 electrons, 23 orbitals |
| **Spin Orbitals / Qubits** | 46 (triplet) or 52 (with all spin channels) |
| **配置數** | ~10^8 |
| **參考能量** | SCI available (IBM+Lockheed 2025) |
| **為何好** | IBM+Lockheed 實際在量子硬體上驗證的系統；singlet-triplet gap 精度 |
| **使用此系統的論文** | IBM+Lockheed JCTC 2025 |

---

## 4. 優先推薦 (Priority Ranking)

### Tier 1: 必須實現 (核心 benchmark)

| Priority | System | Qubits | Rationale |
|----------|--------|--------|-----------|
| **P1** | N2/cc-pVDZ CAS(10,12) | 24 | 從現有 CAS(10,10) 自然延伸，驗證 NF 在中等規模的價值 |
| **P2** | N2/cc-pVDZ CAS(10,20) | 40 | 40Q 核心 target，直接與 IBM 可比 |
| **P3** | N2/cc-pVDZ CAS(10,26) | 52 | IBM de facto 標準，如能達到 chemical accuracy 即為強成果 |

**理由**: N2 是最乾淨的 benchmark — 幾何簡單 (diatomic)、PySCF 設定容易、文獻參考能量豐富、IBM 論文直接可比。通過逐步增加 active space (10→12→15→20→26)，可以系統性地展示 NF+SKQD 的 scaling 行為。

### Tier 2: 化學意義展示

| Priority | System | Qubits | Rationale |
|----------|--------|--------|-----------|
| **P4** | Cr2/STO-3G CAS(12,12) | 24 | 經典多參考 benchmark，展示方法在強關聯系統的能力 |
| **P5** | Cr2/cc-pVDZ CAS(12,20) | 40 | 40Q 多參考 benchmark，化學界最困難的問題之一 |
| **P6** | Fe-porphyrin CAS(8,20) | 40 | 生物化學相關，自旋交叉問題 |

### Tier 3: 擴展驗證

| Priority | System | Qubits | Rationale |
|----------|--------|--------|-----------|
| **P7** | benzene CAS(6,15) | 30 | 有機化學 pi 關聯 |
| **P8** | [2Fe-2S] CAS(22,20) | 40 | IBM 核心 benchmark，但 PySCF 設定較複雜 |
| **P9** | coronene CAS(24,24) | 48 | SqDRIFT benchmark，PAH 系統 |

---

## 5. PySCF 可行性分析

### 所有推薦系統的 PySCF 設定範例

```python
# === Step 1: N2/cc-pVDZ CAS(10,12) ===
from pyscf import gto, scf, mcscf

mol = gto.M(
    atom='N 0 0 0; N 0 0 1.10',
    basis='cc-pvdz',
    symmetry=True,  # D2h symmetry breaks pi degeneracy
    verbose=4
)
mf = scf.RHF(mol).run()
# 12 orbitals, 10 active electrons (frozen 1s core)
mc = mcscf.CASSCF(mf, 12, 10)
mc.kernel()
# FCI energy in this active space
e_fci = mc.e_tot


# === Step 1: Cr2/STO-3G CAS(12,12) ===
mol = gto.M(
    atom='Cr 0 0 0; Cr 0 0 1.68',
    basis='sto-3g',
    spin=0,
    verbose=4
)
mf = scf.RHF(mol)
mf.max_cycle = 200
mf.run()
mc = mcscf.CASSCF(mf, 12, 12)
mc.natorb = True  # use natural orbitals for stability
mc.kernel()


# === Step 2: N2/cc-pVDZ CAS(10,15) ===
mol = gto.M(
    atom='N 0 0 0; N 0 0 1.10',
    basis='cc-pvdz',
    symmetry=True,
    verbose=4
)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 15, 10)
mc.kernel()
# Note: CASSCF FCI solver cannot handle 9M configs
# Need: mc.fcisolver = pyscf.fci.selected_ci.SCI(mol)
# Or use our pipeline's SKQD as the solver


# === Step 3: N2/cc-pVDZ CAS(10,20) ===
# PySCF CASSCF will fail with default FCI solver (240M configs)
# Use CASCI with our integrals extraction instead:
mol = gto.M(
    atom='N 0 0 0; N 0 0 1.10',
    basis='cc-pvdz',
    symmetry=True
)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 20, 10)
# Extract integrals for our pipeline
h1e, e_core = mc.get_h1eff()
h2e = mc.get_h2eff()
# Feed h1e, h2e to MolecularHamiltonian


# === Step 3: benzene CAS(6,15) ===
mol = gto.M(
    atom='''
    C  1.40  0.00  0.00
    C  0.70  1.21  0.00
    C -0.70  1.21  0.00
    C -1.40  0.00  0.00
    C -0.70 -1.21  0.00
    C  0.70 -1.21  0.00
    H  2.49  0.00  0.00
    H  1.24  2.15  0.00
    H -1.24  2.15  0.00
    H -2.49  0.00  0.00
    H -1.24 -2.15  0.00
    H  1.24 -2.15  0.00
    ''',
    basis='sto-3g',
    symmetry=True
)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 15, 6)  # 6 pi electrons, 15 orbitals
mc.kernel()


# === Step 3: Fe-porphyrin CAS(8,20) ===
# Simplified Fe-porphine model (FeN4H4)
mol = gto.M(
    atom='''
    Fe 0.000  0.000 0.000
    N  2.000  0.000 0.000
    N  0.000  2.000 0.000
    N -2.000  0.000 0.000
    N  0.000 -2.000 0.000
    ''',  # Simplified; use full porphine geometry for production
    basis={'Fe': 'cc-pvdz', 'N': 'cc-pvdz', 'H': 'cc-pvdz', 'C': 'cc-pvdz'},
    spin=0,  # or 2 for triplet, 4 for quintet
    verbose=4
)
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 20, 8)
mc.kernel()


# === Step 4: N2/cc-pVDZ CAS(10,26) ===
# This is the IBM standard - 26 orbitals = all non-core
mol = gto.M(
    atom='N 0 0 0; N 0 0 1.10',
    basis='cc-pvdz',
    symmetry=True
)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, 26, 10)
# Can only extract integrals, NOT solve FCI (4.3B configs)
h1e, e_core = mc.get_h1eff()
h2e = mc.get_h2eff()
```

### PySCF 限制與解決方案

| 問題 | 限制 | 解決方案 |
|------|------|---------|
| FCI solver | CAS > ~18 orbitals 無法直接 solve | 使用 CASCI 提取積分，用我們的 SKQD solve |
| Cr2 SCF 收斂 | 高自旋金屬二聚體 | `max_cycle=200`, 分數佔據 initial guess |
| [2Fe-2S] | 抗鐵磁態 | broken-symmetry DFT initial guess → CASSCF |
| N2 MO 退化 | pi 軌道旋轉不確定 | `symmetry=True` (D2h) 固定 |
| 大 active space CASSCF | orbital optimization costs | 用 CASCI (固定 MO) + HF orbitals 或 DMRG orbitals |

---

## 6. 技術挑戰與解決策略

### 6.1 記憶體限制 (128 GB UMA)

| System | Full H (dense) | Sparse H (~100 nnz/row) | Feasible? |
|--------|---------------|------------------------|-----------|
| CAS(10,12) 24Q | 2.9 TB | 0.93 GB | Sparse OK |
| CAS(10,15) 30Q | 591 TB | 13.4 GB | Sparse OK |
| CAS(10,20) 40Q | 420 PB | 358 GB | Need streaming/chunked |
| CAS(12,20) 40Q | 16 EB | 2,239 GB | Need streaming/chunked |
| CAS(10,26) 52Q | 136 EB | 6,448 GB | Need streaming/chunked |

**結論**: 24-30Q 的 sparse H 可以完全放入記憶體。40Q+ 需要：
1. NF truncation 到 ~15K configs 再建 H
2. sigma-vector (matrix-free) 方法
3. 或 streaming sparse H construction (已有 C2+C3 chunked connections)

### 6.2 NF 品質要求

| System | Total Configs | Diag Basis Limit | NF Must Find |
|--------|--------------|------------------|--------------|
| CAS(10,12) | 627K | 15K | top 2.4% configs |
| CAS(10,15) | 9M | 15K | top 0.17% configs |
| CAS(10,20) | 240M | 15K | top 0.006% configs |
| CAS(10,26) | 4.3B | 15K | top 0.0003% configs |

NF 必須在指數級的配置空間中找到 ground state 的 dominant determinants。
Autoregressive transformer 是關鍵 — non-autoregressive NF 無法捕捉 inter-orbital correlations。

### 6.3 參考能量策略

| Qubit Range | Reference Method | Availability |
|-------------|-----------------|-------------|
| 20-24Q | PySCF CASCI FCI (exact) | Direct computation |
| 24-30Q | SHCI (epsilon extrapolation) | Arrow/Dice codes |
| 30-40Q | DMRG (bond dim extrapolation) | Block2/ITensor |
| 40-52Q | DMRG or SHCI literature values | Published results |

**建議**: 對於 Step 3+ (40Q+)，使用文獻中已發表的 DMRG/SHCI 參考能量，而非自行計算。
IBM 的 N2/cc-pVDZ SCI 參考能量 -109.22802921665716 Ha 可直接使用。

---

## 7. 實施路線圖

### Phase A: 立即可做 (1-2 週)

1. 在 `src/hamiltonians/molecular.py` 增加 factory functions:
   - `create_n2_ccpvdz_cas_hamiltonian(cas=(10,12))` — 24Q
   - `create_cr2_hamiltonian(basis='sto-3g', cas=(12,12))` — 24Q
2. 在 `tests/` 增加 integration tests (slow):
   - N2/cc-pVDZ CAS(10,12) chemical accuracy test
   - Cr2/STO-3G CAS(12,12) convergence test
3. 驗證 AR-NF 在 627K config space 的表現

### Phase B: 中期目標 (2-4 週)

1. 增加 CAS(10,15) = 30Q benchmark
2. 增加 benzene CAS(6,15) = 30Q
3. streaming sparse H construction for >1M configs
4. 驗證 NF truncation quality (NF top-15K vs exact top-15K overlap)

### Phase C: 40Q 目標 (4-8 週)

1. N2/cc-pVDZ CAS(10,20) = 40Q — 核心里程碑
2. sigma-vector (matrix-free) SKQD for >100M configs
3. 比較 NF+SKQD vs Direct-CI+SKQD (NF 的價值在此應明顯)
4. 與 IBM SQD 結果直接比較

### Phase D: 50Q+ 雄心 (8+ 週)

1. N2/cc-pVDZ CAS(10,26) = 52Q — IBM 標準
2. Cr2/cc-pVDZ CAS(12,20) = 40Q
3. [2Fe-2S] CAS(22,20) = 40Q
4. 論文撰寫

---

## 8. 與競爭者的對比

| Method | Scale Demonstrated | Our Advantage |
|--------|-------------------|---------------|
| IBM SQD | 77Q ([4Fe-4S]) | 不需要量子硬體 |
| IBM SqDRIFT | 100Q (half-Mobius) | 不需要量子硬體 |
| QiankunNet | 30 spin orbs, 99.9% FCI | 我們也用 AR transformer，可直接比較 |
| NNQS-SCI | 152 spin orbs | 他們用超算 (SC24)，我們用單 DGX Spark |
| AB-SND | molecular systems | 我們有 SKQD (Krylov)，他們用 SBD |
| GPU-DMRG | CAS(82,82) | DMRG 是金標準，我們的方法若能接近是成功 |

**我們的定位**: Classical NF+SKQD pipeline running on single DGX Spark。
如果在 40-52Q 達到 chemical accuracy (1.6 mHa)，且比 Direct-CI 顯著更好，
就足以發表 — 展示 autoregressive NF 的價值。

---

## Sources

### IBM SQD/QSCI/SqDRIFT
- [Chemistry beyond the scale of exact diagonalization (Science Advances 2025)](https://www.science.org/doi/10.1126/sciadv.adu9991)
- [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702)
- [Quantum chemistry with provable convergence via SqDRIFT](https://arxiv.org/html/2508.02578)
- [IBM SQD Tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-quantum-diagonalization)
- [IBM half-Mobius molecule (Science 2026)](https://research.ibm.com/blog/half-mobius-molecule)
- [IBM+Lockheed CH2 study (JCTC 2025)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00075)
- [Cuprate chain SQD convergence](https://arxiv.org/html/2512.04962)
- [SKQD for Heisenberg models](https://arxiv.org/abs/2512.17141)

### QiankunNet
- [Solving many-electron Schrodinger equation with transformer (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-63219-2)
- [Bridging Transformer and Tensor Networks (JCTC 2024)](https://pubs.acs.org/doi/10.1021/acs.jctc.4c01703)

### NNQS-SCI
- [NNQS-SCI: Tackling Trillion-Dimensional Hilbert Space (SC24)](https://dl.acm.org/doi/10.1145/3712285.3759800)

### AB-SND
- [Adaptive-basis sample-based neural diagonalization](https://arxiv.org/html/2508.12724)

### SandboxAQ GPU-DMRG
- [Orbital optimization of large active spaces via AI-accelerators (JCTC 2025)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00571)
- [Quarter petaFLOPS DMRG on DGX-H100 (JCTC 2024)](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00903)

### SHCI
- [Semistochastic Heat-Bath CI (JCTC 2017)](https://pubs.acs.org/doi/10.1021/acs.jctc.6b01028)
- [SHCI GitHub (Arrow)](https://github.com/QMC-Cornell/shci)

### Cr2
- [The Chromium Dimer: Closing a Chapter (JACS 2022)](https://pubs.acs.org/doi/10.1021/jacs.2c06357)
- [DMRG-DSRG for Cr2 (arXiv 2025)](https://arxiv.org/abs/2503.01299)
- [NNQS optimization for Cr2](https://arxiv.org/html/2404.09280)

### Iron-Sulfur Clusters
- [Importance of electron correlation in 2Fe-2S (JCTC 2024)](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00781)
- [Stochastic-CASSCF for iron-sulfur clusters (JCTC 2021)](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00589)

### Fe-Porphyrin
- [Cheap and near exact CASSCF with large active spaces (JCTC 2017)](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00900)
- [FCIQMC/DMRG for Fe(II) porphyrin (Int J Quantum Chem 2021)](https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26454)

### Chemically Decisive Benchmarks
- [Chemically decisive benchmarks on the path to quantum utility (arXiv 2026)](https://arxiv.org/abs/2601.10813)

### Critical Analysis
- [Critical Limitations in QSCI Methods (JCTC 2025)](https://arxiv.org/abs/2501.07231)
- [PennyLane Top 20 Molecules](https://pennylane.ai/blog/2024/01/top-20-molecules-for-quantum-computing)

### Acenes/PAH
- [Active Learning CI for PAH excitation energies (JCTC 2021)](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00769)
- [Benzene QPE workflow (PCCP 2024)](https://pubs.rsc.org/en/content/articlehtml/2024/cp/d4cp03454f)
