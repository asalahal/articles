Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                             BMC Bioinformatics
https://doi.org/10.1186/s12859-024-05987-0
   RESEARCH                                                                                                                                                                                                     Open Access
MGATAF: multi-                                                                                                                                                                                                                                                                                                        channel graph attention
net work with adaptive fusion for cancer-  drug
response predic tion
Dhek raSaeed                                                1*, HuanlaiXing                                                                                                                                                                                      1*, Barak atAlBadani                                                 2, LiFeng                                                               1, RaeedAl‑Sabri                                                                                                                 3, MonirAbdullah                                                                                         4 and
AmirRehman                                                                                                          1
*Correspondence:                                                 Abstract
dhekra@my.swjtu.edu.cn;
hxx@home.swjtu.edu.cn                                            Background:                                                                                                                                Drug response prediction iscritical inprecision medicine todetermine
1 School ofComputing                                             themost effective andsafe treatments forindividual patients. Traditional predic‑
andArtificial Intelligence,                                      tion methods relying ondemographic andgenetic data often fall shor tinaccu‑
Southwest Jiaotong University,                                   racy androbustness. Recent graph‑based models, whilepromising, frequently
Chengdu611756, Sichuan, China
2 School ofComputer Science                                      neglect thecritical role ofatomic interactions andfail tointegrate drug fingerprints
andEngineering, Central South                                    withSMILES forcomprehensive molecular graph construction.
University, Changsha410083,
Hunan, China                                                     Results:                                                                                                                                          We introduce multimodal multi‑                                                                                                                                                                                                                                                                                                                                                                                                                                                               channel graph attention network withadap                                                                                                                         ‑
3 Faculty ofComputer Sciences                                    tive fusion (MGATAF), aframework designed toenhance drug response predic‑
andInformation Systems, Thamar                                   tions bycapturing bothlocal andglobal interactions amonggraph nodes. MGATAF
University, Dhamar87246, Yemen
4 College ofComputing                                            improves drug representation byintegrating SMILES andfingerprints, resulting inmore
and                                                                   Information Technology, precise predictions ofdrug effects. The methodology involves constructing multimodal
University ofBisha, Bisha67714,                                  molecular graphs, employing multi‑                                                                                                                                                                                            channel graph attention networks tocapture
Saudi Arabia
                                                                 diverse interactions, andusing adaptive fusion tointegrate these interactions atmul‑
                                                                 tiple abstraction levels. Empirical results demonstrate MGATAF’s superior per formance
                                                                 compared totraditional andother graph‑based techniques. Forexample, ontheGDSC
                                                                 dataset, MGATAF achieved a5.12% improvement inthePearson correlation coefficient
                                                                 (PCC ), reaching 0.9312 withanRMSE of0.0225. Similarly, innew cell‑line tests, MGATAF
                                                                 outper formed baselines withaPCC of0.8536 andanRMSE of0.0321 ontheGDSC
                                                                 dataset, andaPCC of0.7364 withanRMSE of0.0531 ontheCCLE dataset.
                                                                 Conclusions:                                                                                                                                                                                                                                            MGATAF significantly advances drug response prediction byeffectively
                                                                 integrating multiple molecular data types andcapturing complex interactions. This
                                                                 framework enhances prediction accuracy andoffers arobust tool forpersonalized
                                                                 medicine, potentially leading tomore effective andsafer treatments forpatients. Future
                                                                 research can expand onthis work byexploring additional data modalities andrefining
                                                                 theadaptive fusion mechanisms.
                                                                 Keywords:                                                                      Graph neural network, Drug response prediction, Precision medicine,
                                                                 Bioinformatics
                                                              © The Author(s) 2025. Open Access  This article is licensed under a Creative Commons Attribution 4.0 International License, which permits
                                                              use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original
                                                              author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third
                                                              party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the mate‑
                                                              rial. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or
                                                              exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://
                                                              creat iveco mmons. org/ licen ses/ by/4. 0/.

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                             Page 2 of 24
                      Introduction
                      Cancer remains one of the most significant global health challenges, affecting millions
                      of individuals worldwide and contributing to a substantial number of deaths annually.
                      Effective treatment strategies are crucial for improving patient outcomes, yet predicting
                      how a particular treatment will affect a patient’s tumor response is complex. Precision
                      medicine aims to address this challenge by tailoring treatments to individual patients,
                      but early prediction methods often fall short due to their reliance on limited data sets,
                      such as genetic information or protein expression levels, which do not fully capture the
                      intricate biology of cancer [1–3]. Over the past few years, the growing accessibility of
                      molecular data obtained from individuals affected by cancer has instigated the advance-
                      ment of various extensive drug response initiatives. These endeavors, namely the GDSC
                      [4] and CCLE [5], aim to incorporate a broad spectrum of data sources to augment
                      the precision of prognosticating drug responses. To this end, molecular profiling has
                      emerged as a primary technique in determining the efficacy of cancer treatments [6].
                      It involves a comprehensive analysis of the genetic and molecular characteristics of a
                      patient’s tumor to gain a more profound understanding of its underlying biology. By uti-
                      lizing advanced computational methods and combining multiple sources of molecular
                      data, it is possible to derive a holistic view of the tumor and predict how it will respond
                      to different therapies [7]. Through the integration of various data sources, researchers
                      can overcome one of the significant limitations of early prediction methods, which rely
                      on a single type of data, leading to inadequate insights into the underlying biology of
                      cancer. With a more comprehensive understanding of the molecular profile of a patient’s
                      cancer, doctors can determine the most effective treatment regimen, which is essential
                      for precision medicine. The ability to predict drug response accurately can significantly
                      impact the lives of cancer patients, making large-scale drug response projects [4,                                                                     5,                          8]
                      critical in advancing cancer research and treatment.
                         In this context, several machine learning algorithms have been employed, includ-
                      ing linear regression [9,                                                                                                                                               10], decision trees [11,                                                                                                                                     12], random forests [13], support vec                                                                                                                               -
                      tor machines (SVMs) [14,                                                                                                                                                                        15], and neural networks [16–21]. However, these traditional
                      machine learning algorithms suffer from various limitations that impede their effec-
                      tiveness [22,                                                                                                            23]. For instance, the random forest algorithm’s main limitations include
                      over-fitting and the limited information derived from genomic data alone. Likewise, the
                      SVM algorithm’s main limitations involve being sensitive to the selection of parameters,
                      the constraints of binar y classification, and the quality and complexity of the input data.
                      Despite the algorithm used, a common challenge in using machine learning for drug
                      response prediction (DRP) is the complexity of the data and the intricate relationships
                      between features and drug response, often resulting in overfitting. Unlike simple algo-
                      rithms like linear regression or decision trees, complex algorithms like neural networks
                      are challenging to interpret, which makes it difficult to understand the underlying mech-
                      anisms that drive the predictions.
                         To overcome the limitations of traditional machine learning in predicting cancer drug
                      response, Convolutional Neural Networks (CNNs) [3,                                                                                24,                                             25] have emerged as a prom                                         -
                      ising alternative. Convolutional Neural networks (CNNs) are engineered to automati-
                      cally capture and assimilate salient features from input data, effectively manage complex
                      and high-dimensional data, possess the ability to withstand noise and variability, and

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                           Page 3 of 24
                 incorporate the spatial relationships that exist between features. As a result, CNNs are
                 particularly adept at analyzing cancer genomics data and hold promise for enhanc-
                 ing the precision and accuracy of DRP. Nevertheless, the performance of CNN mod-
                 els is partially contingent upon the structure of the molecular data, thereby producing
                 a level of prediction accuracy that is constrained. This limitation stems from the fact
                 that CNNs represent drugs as strings, which does not align with the natural structure of
                 drugs. Recently, graph neural networks (GNNs) have demonstrated significant potential
                 in cancer drug response prediction [26–32], showcasing encouraging outcomes. GNNs
                 are specifically designed to handle data organized in a graph structure, where the nodes
                 represent unique entities and the edges depict the connections or associations among
                 them. In the domain of cancer drug response, drug composition can be represented as
                 a graphical structure where atoms are nodes and the bonds between them are edges.
                 Employing GNNs in this context offers significant benefits, as they can effectively inte-
                 grate graph-based information into the model. This integration enables GNNs to make
                 more precise predictions, enhancing the accuracy of the overall analysis. Addition-
                 ally, GNNs can handle sparse and noisy data, making them well-suited for molecular
                 data. Existing graph neural network (GNN) models for DRP encounter several limita-
                 tions that hinder their performance. Firstly, many GNNs have limited model capacity,
                 often being shallow networks that primarily capture local graph structures. This defi-
                 ciency restricts their ability to handle intricate relationships among nodes, potentially
                 leading to decreased accuracy in DRP. Secondly, the prevalent reliance on a single-
                 layer attention mechanism in most GNN models proves insufficient for capturing the
                 nuanced interconnections within graphs. These weaknesses undermine their capability
                 to accurately leverage complex relationships and dependencies among nodes, limiting
                 their suitability for DRP. Furthermore, prevalent graph-based methods often overlook
                 the significance of integrating drug fingerprints alongside drug SMILES in constructing
                 molecular graphs. Drug fingerprints, which encapsulate structural and physicochemi-
                 cal properties, offer valuable insights into molecular characteristics that influence drug
                 responses. These fingerprints encode essential information about molecular structure,
                 such as functional groups, bond types, and spatial arrangements, providing a nuanced
                 understanding of drug interactions at the atomic level. Despite the rich information
                 embedded within drug fingerprints, their incorporation into graph-based models has
                 been limited. Instead, many existing approaches rely solely on drug SMILES represen-
                 tations, which primarily capture molecular connectivity. SMILES strings offer a stand-
                 ardized representation of molecular structures but lack detailed molecular properties
                 essential for accurate drug response predictions. Current graph-based methods often
                 ignore drug fingerprints, leading to suboptimal predictive performance. Integrating
                 drug fingerprints with SMILES enhances the models’ comprehensiveness and predictive
                 power, resulting in more accurate drug efficac y predictions .
                    In response to these constraints, we introduce a Multimodal Multi-channel Graph
                 Attention Network with Adaptive Fusion (MGATAF), an innovative framework for
                 predicting cancer drug response. MGATAF stands out for its ability to effectively cap-
                 ture the intricate relationships between drugs and genes, as well as the dependencies
                 between different drugs and genes. The current innovative approach is the process of
                 leveraging multi-channel graph attention and adaptive fusion within MGATAF. Firstly,

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                      Page 4 of 24
                   MGATAF constructs multimodal molecular graphs that incorporate both SMILES and
                   drug fingerprints, providing a comprehensive representation of drug molecules. Then,
                   multi-channel graph attention mechanisms are applied to capture complex interactions
                   among graph nodes , allowing MGATAF to learn multiple orders of relationships . Finally,
                   adaptive fusion techniques integrate these interactions at various levels of abstraction,
                   enhancing prediction performance. By employing multi-channel graph attention with
                   adaptive fusion, MGATAF offers superior predictive capabilities compared to existing
                   methods. These contributions collectively position MGATAF as a promising advance-
                   ment in the field of drug response prediction, offering enhanced predictive capabilities
                   for precision medicine initiatives .
                     The main contributions of our proposed approach, as outlined in this study, are sum-
                   marized as following:
                     1.                                  Introduction of MGATAF Framework: MGATAF, a Multi-channel Graph Attention
                         Network with Adaptive Fusion, is introduced to enhance predictive accuracy in drug
                         response tasks by capturing local and global drug–cell interactions .
                     2.                          Novel Attention and Fusion Modules: MGATAF features a multi-channel graph
                         attention mechanism and an adaptive fusion module, integrating diverse molecular
                         data to produce robust drug–cell interaction representations .
                     3.                                    Superior Empirical Per formance: Experiments on GDSC and CCLE datasets show
                         MGATAF outperforms state-of-the-art methods, achieving better Pearson correla-
                         tion coefficients and root mean square errors .
                     4.                                     Linking Technical Innovation to Clinical Impact: Ablation studies confirm the effi            -
                         cacy of the attention and fusion modules, contributing to more accurate predictions.
                         These improvements have clinical relevance, offering more precise drug efficacy pre-
                         dictions to guide treatment decisions and streamline drug development.
                   Related work
                   Cancer drug response prediction is crucial in precision medicine, guiding the selec-
                   tion of optimal therapeutic inter ventions. A wide range of methods has been employed,
                   including traditional approaches, classical machine learning models, and recent
                   advances in graph neural networks (GNNs). This section reviews these methods, with
                   an emphasis on machine learning and GNN approaches relevant to our proposed model.
                   Traditional approaches
                   Traditional methods, such as genomic profiling, biomarker-based approaches, and
                   in   vitro/in vivo models, have played a significant role in early drug response prediction
                   research. Genomic profiling utilizes high-throughput technologies to identify genetic
                   alterations in cancer cells, assuming a correlation between these alterations and drug
                   responses [33,                                                                                    34]. For example, EGFR mutations are associated with positive responses
                   to EGFR-targeted therapies. While effective in some cases, the predictive power of
                   these methods is limited by the quality of genetic data, high costs, and computational
                   complexity.

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                 Page 5 of 24
                       Biomarker-based approaches aim to predict responses based on specific biological
                    markers like proteins or gene expressions [35,                                                                                                                                                                                           36]. However, their generalizability is
                    often constrained by the cancer type and available biomarkers, reducing their broader
                    applicability. Similarly, in  vitro cell line models [37,                                                                                       38], while useful for screening drug
                    sensitivity, fail to capture the complex biological characteristics of tumors in patients,
                    limiting their clinical relevance.
                    Machine learning‑based approaches
                    Classical machine learning models, including support vector machines (SVMs) and
                    matrix factorization techniques, have been applied to drug response prediction with
                    var ying degrees of success. Dong et al. [39] employed SVMs with recursive feature selec                                                                                                                               -
                    tion to predict drug response across multiple cell lines , while Wang et al. [40] introduced
                    a similarity-regularized matrix factorization (SRMF) method to enhance the predictive
                    power by leveraging the similarity between cell lines and drugs. Despite their utility,
                    these approaches often struggle to capture the full complexity of molecular interactions,
                    leading to limited prediction accurac y.
                       Several computational models have been developed to predict miRNA-disease asso-
                    ciations by integrating heterogeneous biological data and employing advanced machine-
                    learning techniques .
                       Chen et     al. [41] introduced the DBNMDA model, which utilizes deep-belief net                                                                                                                 -
                    works for miRNA-disease association prediction. By pre-training restricted Boltzmann
                    machines on feature vectors constructed from all miRNA-disease pairs and fine-tuning
                    with both positive and selected negative samples, the model effectively leverages infor-
                    mation from both known and unknown associations. This approach reduces the impact
                    of limited known associations on prediction accuracy and achieves superior perfor-
                    mance, with high AUC scores in various cross-validation settings and successful case
                    studies .
                       Ha et   al. [42] proposed IMIPMF, employing probabilistic matrix factorization to pre                                                       -
                    dict miRNA-disease associations. By drawing an analogy to recommender systems, their
                    method addresses the challenge of predicting associations for new miRNAs and diseases
                    without prior known associations. IMIPMF demonstrates high performance with a reli-
                    able AUC value, highlighting its effectiveness despite only considering known miRNA-
                    disease associations and miRNA expression data. Another innovative framework is
                    NCMD, which [43] utilizes node2vec-based neural collaborative filtering for miRNA-
                    disease association prediction. This method learns low-dimensional vector representa-
                    tions of miRNAs and diseases using Node2vec and combines the linear capabilities of
                    generalized matrix factorization with the nonlinear abilities of a multilayer perceptron.
                    Extensive experiments and case studies validate its effectiveness in discovering novel
                    miRNA-disease associations. Ha [44] introduced MDMF, a computational framework
                    that predicts miRNA-disease associations using matrix factorization with a disease simi-
                    larity constraint. By integrating heterogeneous information and evaluating performance
                    through global and local leave-one-out cross-validation, MDMF achieves significant
                    improvements over previous methods. Case studies on major human cancers further

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                     Page 6 of 24
                   demonstrate its efficiency in uncovering miRNA-disease associations and deciphering
                   the roles of miRNAs in disease pathogenesis.
                     In the SM AP framework proposed by Ha [45], miRNA-disease associations are identi                                                                                    -
                   fied by applying recommendation algorithms with miRNA and disease similarity con-
                   straints. By measuring comprehensive similarity values based on miRNA functional
                   similarity, disease semantic similarity, and Gaussian interaction profile kernel similarity,
                   SMAP effectively integrates known associations and similarities to achieve high AUC
                   scores in cross-validation and case studies, serving as a guide for elucidating disease
                   pathogenesis and biomarkers. Ha [46] also presented MLMD, a metric learning-based
                   model for predicting miRNA-disease associations. MLMD learns miRNA-disease met-
                   rics to uncover novel associations as well as miRNA-miRNA and disease-disease simi-
                   larities. The model demonstrates outstanding performance compared to state-of-the-art
                   methods, with reliable AUC scores in cross-validation frameworks and successful case
                   studies confirming its practicality and feasibility.
                     Furthermore, Ha [47] extended computational approaches to lncRNA-disease asso                                                                                          -
                   ciations by proposing EMFLDA, a matrix factorization method that applies lncRNA
                   expression profiles. By effectively incorporating heterogeneous biological datasets and
                   using expression profiles as weights, EMFLDA improves model accuracy and perfor-
                   mance. The model outperforms previous methods in AUC scores and plays a pivotal role
                   in extracting disease biomarkers.
                   Graph neural network approaches
                   Recent advancements in graph neural networks (GNNs) have shown great promise in
                   drug response prediction by effectively modeling the intricate relationships between
                   drugs and cancer cell lines .
                     Several studies have applied GNNs to drug response tasks. For instance, Yang et    al.
                   [48] developed GPDRP, a multimodal framework leveraging drug molecular graphs and
                   gene pathway activity for drug response prediction, while Wang et  al. [49] proposed the
                   XMR model, an explainable multimodal neural network for predicting drug efficacy.
                   While attention-based GNNs [29,                                                                                       50] have improved predictive accuracy by enhancing
                   the model’s ability to learn from molecular representations, their reliance on single-layer
                   attention mechanisms has limited their capacity to capture complex relationships across
                   multiple layers. These models often fail to incorporate diverse neighboring orders of
                   information within the graph structure.
                     To overcome this limitation, the Multi-channel Graph Attention module introduced in
                   our work seeks to incorporate knowledge from multiple neighboring orders into the final
                   prediction. This allows for a more comprehensive understanding of drug–cell interac          -
                   tions by considering both proximal and distant relationships within the molecular graph,
                   improving prediction accuracy. Unlike AMD-GNN [51], which primarily addresses the
                   over-smoothing issue in deep graph neural networks for tasks such as node classifica-
                   tion, our method, MGATAF, is specifically designed for molecular graph representation.
                   While AMD-GNN employs decoupled propagation and transformation mechanisms,
                   MGATAF integrates both SMILES and molecular fingerprints through a multi-graph
                   attention framework. This allows MGATAF to capture molecular interactions across

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Page 7 of 24
                                                                                                                                Table1                                                                                                                          Description of the GDSC and CCLE datasets
                                                                                                                                Characteristic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 GDSC                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              CCLE
                                                                                                                                Before Preprocessing
                                                                                                                                Number of Cell Lines                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    1074                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              1036
                                                                                                                                Number of Drugs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      250                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                24
                                                                                                                                After Preprocessing
                                                                                                                                Number of Cell Lines                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    948                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                436
                                                                                                                                Number of Drugs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      223                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                23
                                                                                                                                Drug–Cell Line Pairs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            172,114                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    10,464
                                                                                                                                Missing Pairs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           18.6%                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               None
                                                                                                                                Table2                                                                                                                          Atom and cell‑line feature descriptions
                                                                                                                                Features                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Description
                                                                                                                                Atom
                                                                                                                                Atom type                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        C, N, O, S, F, etc. (one‑hot)
                                                                                                                                Degree                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        0–10 (one           ‑hot)
                                                                                                                                Implicit valence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0–6 (one           ‑hot)
                                                                                                                                Formal charge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Formal charge number (integer)
                                                                                                                                Radical electrons                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Number of radical electrons (integer)
                                                                                                                                Hybridization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                SP, SP2, SP3, SP3D, SP3D2 (one‑hot or null)
                                                                                                                                Aromatic                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              Whether the atom is in an aromatic system (binar y)
                                                                                                                                Hydrogens                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0–4 (one           ‑hot)
                                                                                                                                Ring                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 Whether the atom is in a ring (binar y)
                                                                                                                                Chirality                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            R, S (one‑hot or null)
                                                                                                                                Cell-line
                                                                                                                                Gene expression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 Expression levels of genes in cell lines
                                                                                                                                Copy number variation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   Variations in the number of copies of a gene
                                                                                                                                Somatic mutations                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                Presence or absence of mutations in genes
                                                                                                                                multiple data types, offering a unique approach for analyzing biochemical structures.
                                                                                                                                These differences reflect the distinct goals and applications of the two methods .
                                                                                                                                Dataset and preprocessing
                                                                                                                                Datasets
                                                                                                                                To evaluate our model, we conducted experiments using two datasets: the Genomics of
                                                                                                                                Drug Sensitivity in Cancer (GDSC) [52] and the Cancer Cell Line Encyclopedia (CCLE)
                                                                                                                                [5]. Dataset descriptions are presented in Tables                                          1 and 2. In this study, we used the
                                                                                                                                GDSC and CCLE datasets independently for all analyses. Future work could explore the
                                                                                                                                potential benefits of integrating these datasets to leverage complementar y information.
                                                                                                                                                 (1) GDSC is a large-scale initiative that screens cancer drugs to assess their efficac y on
                                                                                                                                numerous cancer cell lines , while also providing corresponding omics and drug response
                                                                                                                                data. In our study, we use version 6.0 of the GDSC dataset, which contains the half max-
                                                                                                                                imal-inhibitor y concentration (IC50) values for drug–cell line pairs, covering 250 drugs
                                                                                                                                and 1074 cell lines. Additionally, the cancer cell lines in the GDSC dataset are described
                                                                                                                                by their genetic and omics features, such as copy number variations and mutation sta-
                                                                                                                                tuses. Drugs are identified by their names and compound ID (CID), which can be used

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                                                                                                                     Page 8 of 24
                                           for cross-referencing with other databases. The molecular structures of the drugs were
                                           sourced from PubChem [53].
                                                 (2) The CCLE dataset [5] provides comprehensive genomic and pharmacological data
                                           for human cancer cell lines . This dataset includes experimental data such as drug targets ,
                                           dosage information, log(IC50) values, and effective area measurements for drug–cell
                                           pairs involving 24 drugs and 1036 cell lines. In our study, we utilize the log(IC50) value
                                           as the primar y measure of drug sensitivity.
                                           Preprocessing
                                           Following the methodology described in [54], we selected only drugs with available
                                           IC50 values. For the molecular structure of the drugs, we obtained SMILES strings from
                                           PubChem and represented them as molecular graphs, where atoms form the nodes and
                                           bonds define the edges. Each atom node was described using a 78-dimensional fea-
                                           ture vector. We removed cell lines lacking omics data, as well as drugs that had iden-
                                           tical compound IDs (CID) in PubChem. Additionally, drugs without a corresponding
                                           PubChem ID in the GDSC database were excluded. After preprocessing, the GDSC
                                           dataset contained 172,114 drug–cell pairs derived from 223 drugs and 948 cell lines.
                                           Of the 223     ×                         948      =                           211,404 possible drug–cell interactions , approximately 18.6% were
                                           missing  . 9e IC50 values of each drug–cell pair were normalize d to a range betwe en
                                           0 and 1. Similarly, the CC LE dataset , after preprocessing , contained 11,104 drug–cell
                                           pairs involving 23 drugs and 436 cell lines . For the                                                                                                       24      ×                          436      =                            10,464 possible drug–
                                           cell pairs   , no interaction w a s missing    . At the input stage, dr ug s were repre sente  d using
                                           their c                                                                             anonic                                                                             al SMI                                                 L                                                                     E                                                                                                          S for                                                                             mat                                                                                                                                                                                         , and cell line                                                                     s were enco                                                                                                                              de                                                                                                                                         d a                                                                                                                     s a binar                                                                                                                                                                                                                                                                                                                                   y 735-dimensional
                                           vector.
                                           Method
                                           The MGATAF framework comprises three primary modules, namely the GNN-based
                                           node representation module, the multi-channel graph attention module, and the adap-
                                           tive fusion module, as illustrated in Fig . 1. In this particular section, our initial focus will
                                           be on presenting an outline of the framework in its entirety, and this will be followed by
                                           a detailed description of each individual module included in it.
                                           Overview
                                           In Fig . 1, the GNN-based node representation module is depicted, which accepts the drug
                                           graph and employs a GNN-based network with multiple layers to obtain the drug graph’s
                                           representation. This module retains all node embeddings from each layer. The multi-chan-
                                           nel attention graph module begins by creating a virtual super node cs  , which is linked to all
                                           dr ug no des  . A g raph neural network is then use d to generate the g raph representation of
                                           the sup er no de                             cs  . Next, the hidden state of                                                    cs  , corresponding to the graph embedding of
                                           the no de emb e dding layer, is up date d using GRU. An attention me chanism is utilize d to
                                           determine the weight of each dr ug   ’s g raph emb e dding  .  e CC L and cnger print features
                                           are generate      d by the c   ancer cell line emb     e      dding and cnger     pr   int enco     ding mo     dule   s          , re   sp     e      c                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    -
                                           tively.  en, an adaptive f  usion mo  dule is prop  o s e  d to match the dr  ug re pre s ent ation and
                                           f                         u         s           e                           d C               C                                   L and cnger                       pr               int re           pre             s           ent           ations                                              .  i         s mo                         dule extrac           t                       s the cor               rel         ation f                 rom b                         oth
                                           lo cal and global p ersp e ctives , resulting in a comprehensive representation of cancer and

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                Page 9 of 24
                                Fig.1                                                 MGATAF framework
                             drug responses. This enables effective integration of the encoded drug representation with
                             the encoded features of cancer cell lines and fingerprints. Furthermore, an attention mech-
                             anism layer is employed to identif y the most significant features of the fused embeddings.
                             The output is then generated using a fully connected layer.
                             GNN‑based node representation module
                             GNNs are an effective approach for extracting knowledge from data that is structured as
                             graphs [55]. GNNs possess the capability to learn from graph-structured data, such as
                             social networks , molecular structures , and other types of networks . By effectively capturing
                             the complex relationships between nodes within a graph, GNNs can leverage this informa-
                             tion to make predictions or provide recommendations . Given the drug graph Gd         =                                   (A           d       ,     X    )           ,
                             where                                                                             X is the node feature matrix and Ad is the adjacency matrix of graph. A GCN func                                                                                                                                                                                                                                           -
                             tion i  s applie      d to generate the output re  pre   s  ent  ation                                Z  ld of the l                                                                                 -th layer, which can be rep-
                             resented as follows:
                                          (l                )            1              1    l         −1) l                )
                                       Zd       =                       ReLU     (      ˜D−d2A˜dD−˜d2Z   (˜dW       (˜dl,                                                                             (1)
                             where W       (˜  l                ) is the weight matrix of l-th layer in GCN, the initial                                   Z   (0               )=                     X,           ReLU indicate
                                              d                                                                                                               d
                             the Relu activation function, Ad           =                            Ad         +                         Id˜        ,        and Dd˜ me         ans the di     agonal de      g            re               e matr        i               x of
                             Ad˜   .
                             Multi‑channel graph attention module
                             This section describes the multi-channel graph attention module, which is one of
                             the key components of our proposed framework . The Multi-channel graph attention

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                                Page 10 of 24
                                module is designed to effectively learn representations that take into account multiple
                                levels of information about the relationships between nodes in a graph. This module
                                utilizes the power of graph neural networks (GNNs) in combination with a multi-
                                channel graph attention technique to learn and extract informative representations
                                of graphs. Specifically, a graph’s intricate relationships are captured by applying mul                            -
                                tiple layers of attention. The underlying idea is to use multiple layers of attention to
                                capture different levels of relationships between nodes in the graph, thereby allow                            -
                                ing us to capture and encode complex and multi-scale patterns in the graph. Each
                                layer of attention captures different types of relationships from different layers in the
                                network . This enables us to capture all essential dependencies for modeling complex
                                graphs. O verall, our proposed multi-channel graph attention module aims to enhance
                                the expressive power of our framework, enabling us to learn more informative and
                                powerful representations of graph-structured data. By incorporating the various
                                types of relationships captured by different layers of attention, the proposed model
                                can acquire a more comprehensive and intricate graph representation. This is because
                                the different layers of attention are designed to capture distinct levels and types of
                                relationships between nodes in the graph, allowing the model to learn and incorpo                                -
                                rate more intricate patterns and dependencies among the nodes. Thus, by combining
                                these different types of relationships, the model can learn more complex graph rep                            -
                                resentations that can capture the intricacies of the data. Specifically, this module will
                                receive the output representation of each layer,                                                             Z  ld  ={             z1,      z2,     ...,      zN     }, zi          ∈RD,                                                  where
                                N is the number of nodes, and D                                                                                             is the feature vector dimension. This module gener                -
                                ates new node features for each layer received from the GNN-based node representa                         -
                                tion module and produces a new set of node features,                                                                    Z  ′  ={                       z′,     z′,     ...,     z′, z′i              RD′.
                                                                                                                                                                        1     2              N      }   i
                                    The attention coefficient eij    , indicating the signiecance of node j                                                                                                                                                                                                                                                                                                                                                                          ’s features to node i                                                         ,
                                can be calculated as follows:
                                          eij        =α(Wzi     ,      Whjj,                                                                                                                                            (2)
                                where W              ∈                 RF′×F is a weight matrix, and a represents a shared attention mechanism:
                                RF  ′       ×                 RF  ′        →                           R. In the experiment , this mechanism is implemented as a single-layer
                                feedforward neural network.
                                    Using the attention mechanism, the model can assign di erent weights to drugs
                                de  p     ending on their rele       v    ance to the g    raph emb     e     dding           .  i  s help  s to c   apture the com                                                                                                                                                                                                                                                                                                                  -
                                plex relationship s b etwe en dr ug s and their interactions w ithin the g raph.  e result                                                                                                                                                                                                                                                               -
                                ing g      raph emb       e        dding i  s a comp   ac   t re   pre    s   ent   ation of the entire g      raph that c    an b       e u  s   e        d
                                for pre  diction ta sk  s   . To ensure the co e cient s are e a sily comparable acro ss di erent
                                no         de    s                , the S        of         tMa          x f         unc    tion i   s u   s    e         d to nor     mali      ze them for all choice    s of j                                                                                                                                                                                                                                                                                                                                                                                                                                          :
                                          α                              e               ij   )       =∑exp(eij)
                                             ij         =                              softmaxj   (     exp(e                                                                                                           (3)
                                                                                               k ∈Ni                ik    )
                                where Ni is the set of no de i’s neighbors in the graph. Additionally, a is parameterized
                                by a weight vector ϕ              ∈               R2D                 ′, with the nonlinearity Leaky ReLU ser ving as the activation
                                function.

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                                                                                                                                                          Page 11 of 24
                                                                                                            LeakyReLU                   (ϕT[Wz            i            �                  Wzj  ]))
                                                                 αij             =                                                                                          exp(∑exp(LeakyReLU                   (ϕ T     [Wz            i            �                  Wz]))                                                           (4)
                                                                                             k  ∈N                 i                                                                                            k
                                                 where .T i s the transp   o  sition op   eration.  en, the output of no   de re pre  s ent ation i based
                                                 on the multi-head attention mechanism is as follows:
                                                                                                                                                   
                                                                 z′                                           �          α k                                                                                                                                                                                                           (5)
                                                                    i            =�                          Kk  =1                                σij   W     k   zi
                                                                                                            j∈N                 i
                                                 where    K and                              σ represent the number of attention mechanism heads and the nonlinear
                                                 function, respectively. Additionally,                                                                                    σ       kij are the normalized attention coe cients com                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      -
                                                 pute    d by the k-th attention mechanism (                                                                                           σk         ),          and  W    k   i s the cor  re  sp   onding weig ht matr  i    x
                                                 for the input line ar transformation. In the Wnal layer of the network    , an average multi-
                                                 head attention mechanism is applied as follows.
                                                                                                                                                          
                                                                                                1      �K           �
                                                                 h′                                                             α k                                                                                                                                                                                                    (6)
                                                                     i           =                           σK                       ij  W     k   zj
                                                                                                        =1         j∈N       i
                                                 where                σ is the activation function.
                                                       Finally, the multi-channel g raph attention mo dule is the g raph emb e dding for multi-
                                                 level representing the embedding of multiple layers, as follows:
                                                                                                                                                                                                                              
                                                                                                                                                           1      �                    �                                      
                                                                [hl1  ,      hl2  ,    ...,      hlN  ]=h                                           li             =                                  σKKα kij   W     k    zj                                                                                                           (7)
                                                                                                                                                                      =1                 ∈N       i
                                                 where l is the layer number.
                                                 Cancer cell line embedding and fingerprints encoding
                                                 For a given cancer cell line C, let’s denote C as a matrix with dimensions (N,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                F), where N
                                                 is the number of features (e.g., the number of features) and F is the number of features
                                                 per cell line. The output of a convolutional layer can be computed as follows:
                                                                                         (                  F∑∑w                                                                               )
                                                                Cij          =                                 σ         C(i        +l                  −1),k                                                    ·      Wj ,k            ,l              +                         bj                                                    (8)
                                                                                                k =1         l =1
                                                Cij i   s the output fe      ature map at p         o    sition i and channel j,                                                                                                         σ i   s the ac    tiv      ation f         unc    tion (e.g                   .,
                                                 ReLU),                                                                   w is the size of the convolution filter (kernel),                                                      Wj  ,k                   ,l is the weight of the j-th filter
                                                 at feature k and filter position l, and                                                                               bj is the bias term for the j-th filter.
                                                       After flattening the output of the convolutional layer, the output of the fully connected
                                                 layer can be expressed as follows:
                                                                                                                                                   
                                                                                              �M
                                                                Ci          =                           σ Cj         ·      Wij         +                      bi                                                                                                                                                                      (9)
                                                                                               =1

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                  Page 12 of 24
                           Ci is the output of the i-th neuron in the fully connected layer,                                                   σ is the activation function
                            (e.g    ., ReLU, Sig moid), M is the number of inputs to the fully connected layer (flattened
                            feature maps from the convolutional layer), Cj is the j-th input to the fully connected
                            layer, Wij is the weight conne cting the j-th input to the i-th neuron, and                                                                  bi is the bias
                            term for the i-th neuron.
                               For a given fingerprint F, let’s denote F as a matrix with dimensions (M,                                                                                                                                                                                                                                                                                                                                                                                                      K), where M is
                            the number of features (e.g ., the number of features) and K is the number of features per
                            cell line. The output of a convolutional layer can be computed as follows:
                                                  (               K∑∑w                                       )
                                     Fij          =                                 σC(i        +l                  −1),k                                                    ·      Wj ,k            ,l              +                        bj(10)
                                                      k =1    l =1
                            Fij i  s the output fe     ature map at p       o    sition i and channel j,                            σ i  s the ac   tiv     ation f       unc   tion (e.g                .,
                            ReLU),                                                                   w is the size of the convolution filter (kernel), Wj, ,l is the weight of the j-th filter
                            at feature k and filter position l, and                            bj is the bias term for the j-th filter.
                               After flattening the output of the convolutional layer, the output of the fully connected
                            layer can be expressed as follows:
                                                      M                            
                                                    �                              
                                     Fi          =                           σCj         ·      Wij         +                       bi                                                   (11)
                                                      =1
                            Fi is the output of the                                                                                                                                                                                                                 i-th neuron in the fully connected layer, σ is the activation function
                            (e.g    ., ReLU, Sig moid), M is the number of inputs to the fully connected layer (flattened
                            feature maps from the convolutional layer), Fj is the                                                                                        j-th input to the fully connected
                            layer, Wij is the weight conne cting the j-th input to the i-th neuron, and                                                                  bi is the bias
                            term for the i-th neuron.
                               Then, the cancer cell lines embedding and fingerprints embedding are concatenated as
                            follows:
                                     X              =                              concat  [C                                   ,      F    ]                                            (12)
                            where C is the cancer cell embeddings and F is the fingerprints embeddings.
                            Adaptive fusion module
                            Adaptive fusion is a technique in neural networks that involves the merging of several data
                            sources into a single output. The objective of this approach is to improve the precision and
                            reliability of the neural network by integrating diverse data sources, such as drug represen-
                            tation and cancer cell line representation, thereby enhancing its accuracy and stability. By
                            doing so, the model can extract complementary information from different sources and
                            produce more comprehensive and reliable predictions. The combination process involves
                            dynamically weighting the different data sources based on their relative importance and rel-
                            evance to the task at hand. The adaptive nature of the technique ensures that the fusion
                            process is tailored to the specific input data and task , making it more effective and efficient
                            than fixed fusion methods. As shown in Fig. 2, the proposed adaptive fusion module inte-
                            grates multiple input representations from both representations into a single output. This
                            process can be done using a weighted sum operation. The weights assigned to each input

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                    Page 13 of 24
                          are adjusted based on the performance of the network on a given cancer and drug response
                          prediction task . This allows the network to adapt to changing conditions and improve its
                          accuracy over time. Adaptive fusion can also be used to reduce overfitting by combining
                          multiple sources of information into one output.
                             In the last layer, the study combines the representations of the drug and cancer cell line to
                          capture their overall relationship. This comprehensive method allows us to create a power-
                          ful representation by capturing the correlation between them. To accomplish this , the study
                          employs an adaptive fusion module that integrates the two-view features using a control
                          gate weight. This weight is determined by the underlying temporal properties of the fea-
                          tures, which are processed through a fully connected layer followed by a Sigmoid function.
                          The output of the Sigmoid function represents the control gate weight, which is then used
                          to modif y the two-view features. This modification ensures that the most informative fea-
                          tures are emphasized while irrelevant features are suppressed. By employing this technique,
                          the study aims to enhance the accurac y and robustness of the neural network by effectively
                          integrating multiple sources of information. The two modified characteristics are then com-
                          bined. The AF module is formulated as follows:
                                  gout             =              σ(Wg     (concat               [X                                    ,     H   ])                   +                         bg     ),(13)
                                               ⨂              ⨁          ⨂
                                  H    =(h                1                         2),                                                                                     (14)
                          where    H is the fingerprint representations,                               contact represents concatenation,                           Wg is the
                         FC l ayer      ’s trainable p arame ter, and                  bg is the bias term.              σ is the SoftMax function, h rep                                                                                                                                                                                                                                                                                                -
                          resents the learned embeddings for the input data, while gout 1 and                                                    gout 2 are the gating
                         mechanisms responsible for modulating the importance of the fused features in the
                         adaptive fusion module.
                             Finally, an attention mechanism is applied and fully connected layer as follows:
                                                               (       [                  ])
                                  eij          =                          LeakyReLUa TWhi �Wh                        j                                                      (15)
                         where       eij represents the attention score between node i and node j,                                            a is the weight vector,
                          W is the weight matr i  x   , hi and                   hj are the feature ve ctors of nodes i and j, and                                 ∥ denotes
                          concatenation. Normalize the attention scores using the softmax function:
                            Fig.2                                                 Adaptive fusion module structure

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                               Page 14 of 24
                                    α          ∑      exp(e     ij)
                                       ij         =          exp(e     ik    )                                                                                                        (16)
                                                    k ∈ Ni
                           where αij is the normalized attention coe cient , and                                             Ni denotes the neighbors of node i.
                               Then, Compute the final output embedding:
                                                                        
                                    h′                    �αij Whj                                                                                                                  (17)
                                       i           =                           σ
                                                    j∈     Ni
                           where h ′i i   s the up        d   ate         d fe      ature ve         c    tor for no         de                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        i, and σ is an activation function, such as
                           ReLU.
                                e f ully conne cte d layer for the emb e dding                                      H       ′ can be describe d as follows : Flatten
                           the output embe dding                     H       ′:
                                    h′            =                                    Flatten(H                 ′  )                                                                 (18)
                            en, apply the fully conne cte d layer :
                                                                              
                                                     �M
                                    Oi          =                           σh′j  Wij         +                      bi                                                             (19)
                                                      =1
                           where       Oi i s the output of the i-th neuron in the fully connected layer,                                                   σ is the activation
                           f         unc    tion (e.g                     ., ReLU, Sig        moid), M is the number of inputs to the fully connected layer,                              h ′j
                           is the j-th input to the fully connected layer,                                 Wij is the weight connecting the j-th input to
                           the i-th neuron,              bi is the bias term for the i-th neuron.
                           Experiments
                           Baselines
                           The study assesses the effectiveness of MGATAF compared to various established mod-
                           els as baselines,such as tCNNS [56] and variants of popular graph neural network-based
                           models, including GCN [57], GIN [58], GAT [59], and SuperGAT [60]. The tCNNS
                           model employs SMILES strings to represent drugs and utilizes a convolutional layer to
                           extract features of drugs. Additionally, it employs another convolutional layer to extract
                           features of cancer cell lines from genetic attribute vectors. Ultimately, a fully connected
                           layer is employed to predict the response of the drug–cell interaction.
                               On the other hand, the GCN, GIN, GAT, and SuperGAT models adopt a unique
                           approach by representing drugs as graphs and cancer cell lines as one-hot vectors. These
                           models employ graph convolutional layers to extract essential characteristics from both
                           drugs and cancer cell lines. Subsequently, the drug and cancer cell line features are com-
                           bined, and the models predict the IC50 value.
                               The study compares the performance of these models against MGATAF, which is a
                           proposed approach that combines the drug and cancer cell line representations at vari-
                           ous levels using the multi-channel graph attention module and adaptive fusion module.
                           This allows MGATAF to capture the cross-correlation between the drug and cancer cell
                           line representations and provide an efficient final representation for predicting the IC50
                           value.

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                   Page 15 of 24
                         Experimental settings
                         To facilitate re-implementation of our MGATAF model, we provide detailed informa                                -
                         tion on the parameters for each module. The MGATAF framework is composed of
                         three primar y components:
                             1.              A GNN‑based node representation module that employs a multi-layer Graph Con                                                                                                                                                                                                                                                                                                                           -
                                  volutional Network (GCN) with ReLU activation functions, retaining node embed-
                                  dings from each layer; weights are initialized using Xavier initialization, and an L2
                                  regularization term with a weight decay of 5     ×                        10−4           is applied.
                             2.                                  A                 multi‑                                                                      channel graph attention module that uses multi-head attention mecha                                                                                                                                                                                                                               -
                                  nisms with 8 heads in the first layer and 1 head in the second layer, leveraging
                                  LeakyReLU activation with a negative slope of 0.2; attention coefficients are com-
                                  puted using shared weight matrices W              ∈                 RF  ′ ×F      and attention vectors                ϕ              ∈R2D                 ′.
                             3.              An adaptive fusion module that concatenates drug representations with cancer cell
                                  line and fingerprint embeddings (obtained via convolutional and fully connected lay-
                                  ers with ReLU activations), applies a gating mechanism using a Sigmoid function to
                                  adaptively weight features and incorporates an attention mechanism followed by a
                                  fully connected layer to produce the final output.
                         The learning parameters are adjusted to maximize accuracy on the validation sam                             -
                         ples, with optimal values carefully selected. The entire model is trained using the
                         Adam optimizer with a learning rate of 0.0005, a dropout rate of 0.3 applied to pre                                     -
                         vent overfitting, and early stopping if validation loss does not decrease for 50 con                                   -
                         secutive epochs, over a maximum of 300 epochs. Training and experimentation are
                         conducted using the NVIDIA GeForce GTX 1080 Ti graphics card. In comparison,
                         the baseline models adhere to the same parameter settings. The experimental results
                         are presented, with the best outcomes highlighted for clarity.
                         Evalution metrics
                         To assess the performance of our model, we employ two metrics: the Pearson cor                               -
                         relation coefficient (PCC) and root mean square error (RMSE). The PCC is a widely
                         employed statistical measure that gauges the strength of the linear association
                         between two variables. In particular, a PCC value of −                                                                                                                1 denotes a flawless negative
                         correlation, 0 represents no correlation, and 1 signifies a perfect positive correlation.
                         The PCC is calculated using the following equation:
                                                         ∑n        x              i        −x )(¯y                             i    −¯y)
                                  PCC      =    √∑            i=1                                 (∑
                                                         n    (xi   −                       ¯x)2n(yi−¯y)2
                                                         i=1                      i=1
                         where  x and                     Y represent the sets of true ln(IC50) and predicted ln(IC50), respectively.                         n
                         is the number of data points , xi and                          yi are the true ln(IC50) and predicted ln(IC50) of the
                         ith data point, respectively, and                     o¯ and     y¯ are the means of x and Y, respectively.
                             The RMSE is another measure of error used to evaluate the accuracy of a model in
                         predicting quantitative data. It is calculated as follows:

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Page 16 of 24
                                                                                                                           Table3                                                                                                                          Per formance comparison on the GDSC and CCLE datasets in the mixed test experiment
                                                                                                                           Method                                                                                                                                                                                                    GDSC dataset                                                                                                                                                                                                                                                                                                                   CCLE dataset
                                                                                                                                                                                                                                                                                                                                     P CC                                                                                                                                                                                                                                                                                                                                RMSE                                                                                                                                                                                                                                                                                          P CC                                                                                                                                                                                                                                                                                                                        RMSE
                                                                                                                           tCNNS [56]                                                                                                                                                                                                                                                                                                                                                                                                                                                                0.8890                                                                                                                                                                                                                                                                              0.0312                                                                                                                                                                                                                                                                    0.7483                                                                                                                                                                                                                                                                      0.0612
                                                                                                                           GCN [57]                                                                                                                                                                                                                                                                                                                                                                                                                                                                0.9118                                                                                                                                                                                                                                                                              0.0273                                                                                                                                                                                                                                                                    0.7828                                                                                                                                                                                                                                                                      0.0508
                                                                                                                           GIN [58]                                                                                                                                                                                                                                                                                                                                                                                                                                                                0.9264                                                                                                                                                                                                                                                                              0.0252                                                                                                                                                                                                                                                                    0.7556                                                                                                                                                                                                                                                                      0.0542
                                                                                                                           GAT [59]                                                                                                                                                                                                                                                                                                                                                                                                                      0.9065                                                                                                                                                                                                                                                                              0.0289                                                                                                                                                                                                                                                                    0.7741                                                                                                                                                                                                                                                                      0.0520
                                                                                                                          SuperGAT [60]                                                                                                                                                                                                                                                                                                                                 0.8800                                                                                                                                                                                                                                                                              0.0333                                                                                                                                                                                                                                                                    0.7750                                                                                                                                                                                                                                                                      0.0519
                                                                                                                          MGATAF                                                                                                                                                                                                                                                                                                                                                        0.9312                                                                                                                                                                                                                                                                          0.0225                                                                                                                                                                                                                                                                                                     0.7859                                                                                                                                                                                                                                                                          0.0503
                                                                                                                           Bold: our method.
                                                                                                                           Table4                                            Performance comparison on the GDSC and CCLE datasets in the new cell‑line test
                                                                                                                           experiment
                                                                                                                           Method                                                                                                                                                                                                     GDSC dataset                                                                                                                                                                                                                                                                                                                  CCLE dataset
                                                                                                                                                                                                                                                                                                                                      P CC                                                                                                                                                                                                                                                                                                                           RMSE                                                                                                                                                                                                                                                                                             P CC                                                                                                                                                                                                                                                                                                                        RMSE
                                                                                                                           tCNNS [56]                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0.3490                                                                                                                                                                                                                                                                        0.0576                                                                                                                                                                                                                                                                      0.3469                                                                                                                                                                                                                                                                      0.0692
                                                                                                                           GCN [57]                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0.8399                                                                                                                                                                                                                                                                        0.0363                                                                                                                                                                                                                                                                      0.7279                                                                                                                                                                                                                                                                      0.0563
                                                                                                                           GIN [58]                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0.8460                                                                                                                                                                                                                                                                        0.0358                                                                                                                                                                                                                                                                      0.7252                                                                                                                                                                                                                                                                      0.0575
                                                                                                                           GAT [59]                                                                                                                                                                                                                                                                                                                                                                                                                         0.8312                                                                                                                                                                                                                                                                        0.0380                                                                                                                                                                                                                                                                      0.7078                                                                                                                                                                                                                                                                      0.0580
                                                                                                                           SuperGAT [60]                                                                                                                                                                                                                                                                                                                                    0.8289                                                                                                                                                                                                                                                                        0.0378                                                                                                                                                                                                                                                                      0.7027                                                                                                                                                                                                                                                                      0.0585
                                                                                                                           MGATAF                                                                                                                                                                                                                                                                                                                                                          0.8536                                                                                                                                                                                                                                                                            0.0321                                                                                                                                                                                                                                                                                                        0.7364                                                                                                                                                                                                                                                                                                        0.0531
                                                                                                                           Bold: our method.
                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                         ∑n
                                                                                                                                                                                                                                                  1                                                                                                                                           2
                                                                                                                                                                   RMSE                                                      =                    √n                                                                      (x             i       −                      yi   )
                                                                                                                                                                                                                                                                                              =1
                                                                                                                           Here, xi and                                                                                                 yi represent the true ln(IC50) and predicted ln(IC50) of the                                                                                                                                                                                                                                                                                                                                                                                                                                                           ith data point,
                                                                                                                           respectively, and n is the number of data points in the sample.
                                                                                                                           Experimental results
                                                                                                                           The study conducted an assessment and comparison of MGATAF’s overall effectiveness
                                                                                                                           with several baseline models, specifically tCNNS and different DRP models. The study
                                                                                                                           employed two metrics, PCC and RMSE, to evaluate the performance of these models.
                                                                                                                           The metrics were calculated using identical benchmark datasets and settings for all
                                                                                                                           approaches. In Tables     3 and 4, the study presents the outcomes of our experiments in
                                                                                                                           relation to the baseline models, with the best results being emphasized in bold to facili-
                                                                                                                           tate easy comparison.
                                                                                                                                           Mixed Test: In this experiment, the study assesses the performance of MGATAF
                                                                                                                           using all available drugs and cell lines in the training phase, meaning that all drugs and
                                                                                                                           cell lines have been seen at least once during training. This study only consider the
                                                                                                                           172114 drug–cell pairs for which response data is provided by GDSC. The data is shuf           -
                                                                                                                           fled and split into 80% as a training set, 10% as a validation set, and 10% as a test set.
                                                                                                                           Table      3 shows that graph neural network-based models outperform the convolutional

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                Page 17 of 24
                  network-based model in both PCC and RMSE. Moreover, our approach achieves the
                  best performance compared to all baseline models for both PCC and RMSE .
                     New Cell Line Test E xperiment : In the tests for new cells, the drugs and cells are sepa        -
                  rated into the training , validation, and test datasets instead of the interaction pairs. This
                  simulates the scenario where new cell lines are introduced and need to be predicted. The
                  results for these tests are shown in Table    4. The performance of all models was not as
                  good as in the mixed experiment, indicating that it is more challenging to predict cancer
                  drug responses for new cell lines. However, MGATAF outperforms all baseline methods
                  in terms of both PCC and RMSE in this experiment, demonstrating its effectiveness in
                  predicting responses for unseen cells .
                     We have conducted statistical significance tests to strengthen the comparison between
                  our MGATAF model and the baseline methods. Specifically, we trained and evaluated
                  each model five times using different random seeds to account for variability due to ini-
                  tialization and data shuffling and applied paired two-tailed t-tests to compare the perfor-
                  mance of MGATAF against each baseline method on both the GDSC and CCLE datasets
                  for the Pearson Correlation Coefficient (PCC) and Root Mean Square Error (RMSE)
                  metrics. The results indicate that in the mixed test experiment, the improvements in
                  PCC and reductions in RMSE achieved by MGATAF overall baseline models were sta-
                  tistically significant, with p-values less than 0.01 on the GDSC dataset and less than 0.05
                  on the CCLE dataset ; similarly, in the new cell-line test experiment, MGATAF’s perfor-
                  mance gains over the baseline methods were statistically significant, with p-values less
                  than 0.05 for both PCC and RMSE metrics on both datasets. These statistical tests con-
                  firm that the superior performance of MGATAF is not due to random chance but is sta-
                  tistically significant, reinforcing the effectiveness of our proposed model in comparison
                  to existing methods.
                  Discussion
                  In our study, we developed the MGATAF model to predict drug responses in cancer cell
                  lines by integrating drug representations, cancer cell line embeddings, and fingerprint
                  features. The superior performance of MGATAF, as evidenced by the highest Pearson
                  Correlation Coefficient (PCC) and lowest Root Mean Square Error (RMSE) in both the
                  mixed test and new cell line test experiments (Tables  3 and 4), has significant biological
                  implications. We can summarize the biological significance of the MGATAF model as
                  follows:
                     1.                                           Enhanced Drug Response Prediction: The high PCC values indicate that MGATAF
                        can accurately predict the sensitivity of cancer cell lines to various drugs. This sug-
                        gests that our model effectively captures the complex biological interactions between
                        drug compounds and cancer cell genotypes . Accurate predictions can aid in selecting
                        the most effective drugs for specific cancer types, thereby advancing precision medi-
                        cine.
                     2.                                Generalization to New Cell Lines: In the new cell line test experiment, MGATAF
                        outperforms all baseline models, demonstrating its robustness and generalization
                        capability to unseen cell lines. This is biologically significant as it indicates the mod-
                        el’s potential to predict drug responses in newly discovered or less-characterized

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                        Page 18 of 24
                             cancer cell lines, facilitating the exploration of treatment options for rare or resistant
                             cancers .
                         3.                                                 Interpretation of Molecular Mechanisms: The multi-channel graph attention mech                                                                                                                                                                                                 -
                             anism in MGATAF allows for the identification of important molecular substruc-
                             tures within drug compounds that contribute to their efficacy. By assigning attention
                             weights to different parts of the drug graphs, the model highlights which molecular
                             features are most influential, providing insights into the mechanisms of action at a
                             molecular level.
                         4.                                       Integration of Multi‑                                                                                                                                                                                                                      Omic Data: The adaptive fusion module combines drug repre                                        -
                             sentations with cancer cell line genomic data and fingerprint features. This integra-
                             tive approach reflects the multifactorial nature of drug responses, considering both
                             the genetic makeup of the cancer cells and the chemical properties of the drugs . Such
                             integration is crucial for understanding the complex biological pathways involved in
                             drug sensitivity and resistance.
                         5.                         Potential for Drug Repositioning: The ability of MGATAF to predict responses
                             across a wide range of drugs suggests its utility in drug repositioning efforts . By iden-
                             tif ying unexpected sensitivities, the model can propose existing drugs as candidates
                             for new therapeutic applications, accelerating the development of effective cancer
                             treatments .
                         6.                          Facilitating Biomarker Discovery: The attention mechanisms and feature impor    -
                             tance scores generated by MGATAF can help identify key genetic markers and
                             molecular features associated with drug responses. This can guide experimental
                             studies aiming to validate potential biomarkers for prognosis or as targets for new
                             drugs .
                      Ablation study
                      This section conducted three ablation studies to investigate the effects of multi-channel
                      attention, adaptive fusion, and the number of layers on the overall performance of the
                      model.
                      Effect of multi‑channel attention
                      This section presents an ablation study conducted to investigate the effect of multi-chan-
                      nel attention on the final representation of the model and its impact on the model’s per-
                      formance. Multi-channel attention is a mechanism used in deep learning to selectively
                      weight the importance of different channels in a multi-channel input. This study focuses
                      on the impact of multi-channel attention on the performance of the model.
                         To conduct the experiment, the model was trained both with and without multi-chan-
                      nel attention, namely MGATAF-MA , and the final representation from the last layer of
                      the model was used for the analysis . The representation of each layer was not kept to iso-
                      late the effect of multi-channel attention on the final representation of the model.
                         The results of the experiment shown in Tables 3 and 4 demonstrate that the inclusion
                      of multi-channel attention had a positive effect on the performance of the model. This

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                              Page 19 of 24
                  indicates that the use of multi-channel attention improves the final representation of the
                  model and leads to better performance on the task .
                    It is important to note that ablation studies are typically conducted in conjunction
                  with other experiments to provide a more comprehensive understanding of the model’s
                  behavior. In this case, it would be useful to evaluate the performance of the model in
                  comparison to other attention mechanisms or without any attention mechanism to fur-
                  ther validate the effectiveness of multi-channel attention.
                    In summary, the ablation study conducted on the effect of multi-channel attention
                  demonstrated that including multi-channel attention improves the final representation
                  of the model and leads to better performance on the task . This finding underscores the
                  importance of incorporating attention mechanisms in deep learning models to selec-
                  tively weight the importance of different channels and improve the model’s performance.
                  Effect of adaptive fusion
                  This section is dedicated to exploring the impact of adaptive fusion, which is a technique
                  used in deep learning to merge multiple sources of information into a single output,
                  on the performance of a model in cancer-drug response tasks. Specifically, the study is
                  interested in exploring the differences between the concatenation operation and adap-
                  tive fusion operation in terms of performance.
                    The concatenation operation involves simply concatenating the features from different
                  modalities into a single feature vector. This method can be effective in some cases, but
                  it has the disadvantage of not being able to weigh the importance of different modalities
                  according to their relevance to the task . On the other hand, an adaptive fusion operation
                  is designed to dynamically adjust the importance of different modalities based on their
                  relevance to the task , which can lead to better performance.
                    To conduct this ablation study, the method first trains the model using the concatena-
                  tion operation as the fusion operation, namely MGATAF-AF. The study then modifies
                  the model to use an adaptive fusion operation and compares the performance of the two
                  models on a test set. The study aims to determine which fusion operation is more effec-
                  tive in capturing the complex relationships between drugs, proteins, and genes, which
                  are critical for accurate drug response prediction.
                    The results of the study shown in Tables 3 and                     4 demonstrate that the adaptive fusion
                  operation leads to improved performance compared to the concatenation operation.
                  This suggests that the ability to dynamically adjust the importance of different modali-
                  ties is an important factor in achieving high performance in our tasks. Therefore, it
                  concludes that using an adaptive fusion operation can effectively capture the complex
                  relationships between drugs, proteins, and genes, and improve the accuracy and robust-
                  ness of the model.
                    Overall, the findings of this study highlight the importance of carefully selecting and
                  tuning fusion operations in cancer-drug response tasks to achieve the best possible per-
                  formance. By selecting the appropriate fusion operation, the method can improve the
                  accurac y and robustness of our models, which can ultimately lead to better personalised
                  treatments for patients as shown in Figs . 3, 4.

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                       Page 20 of 24
                          Fig.3                                                 PCC per formance on the effect of multi‑                                                                                                                                        channel attention and adaptive fusion on the GDSC and CCLE
                          datasets in the mixed test experiment and the new cell‑line test experiment
                          Fig.4                                                 ERMS per formance on the effect of multi‑                                                                                                                                        channel attention and adaptive fusion on the GDSC and
                          CCLE datasets in the mixed test experiment and the new cell‑line test experiment
                        Effect of number of layers
                        In this particular ablation study, the focus is on the effect of the number of layers on the
                        performance of a deep learning model. The experiment involves training and evaluating
                        models with var ying numbers of layers , specifically 1, 2, 3, 4, and 5 layers .
                           As shown in Fig. 5, as the number of layers increases, the performance of the model
                        initially improves. This is likely due to the ability of deeper models to capture more com-
                        plex patterns and relationships within the data. However, after a certain number of layers,
                        the performance begins to decline. This phenomenon is known as the “vanishing gradi-
                        ent problem.” As the number of layers increases, it becomes more difficult to propagate
                        gradients through the entire network during training, leading to slower convergence and
                        degraded performance. The findings of this research demonstrate that the optimal number
                        of layers for this particular problem is 3 layers. This suggests that a moderate level of depth
                        is sufficient to capture the relevant patterns in the data , without encountering the vanishing
                        gradient problem that can arise with deeper networks .

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                         Page 21 of 24
                               Fig.5                                                 Ablation study on the number of layers
                            Conclusion
                            Predicting drug response is a vital task in the field of precision medicine; however, tradi-
                            tional approaches have shown limited accurac y and robustness. This research presents a
                            novel methodology known as the Multi-channel Graph Attention Network with Adap-
                            tive Fusion (MGATAF) networks, offering an innovative approach to drug response
                            prediction. The proposed framework efficiently captures the intricate relationships
                            among drugs , proteins , and genes , which are often overlooked by conventional methods .
                            Additionally, it addresses the issue of disregarding the significant interactions between
                            atoms. Our experimental findings indicate that MGATAF surpasses both traditional and
                            graph-based methods, highlighting its potential as a robust tool for precision medicine.
                            By enabling more precise and effective drug response predictions tailored to individual
                            patients, our approach is poised to contribute significantly to the advancement of preci-
                            sion treatments, ultimately improving health outcomes. Given the promising results of
                            this study, we anticipate that it will inspire further research aimed at enhancing patient
                            outcomes in the field of precision medicine.
                            Abbreviations
                            MGAT                                                  Multimodal multi‑                                                                                                                                                                                                                     channel graph attention network
                            PCC                                                                                Pearson correlation coefficient
                            CNNs                                                                                                                   Convolutional neural networks
                            GNNs                                                                                                                    Graph neural networks
                            DRP                                                                                                                    Drug response prediction
                            RMSE                                                       Root                                                                           ‑mean‑square        deviation
                            Acknowledgements
                            The authors are thankfulto the Deanship of Graduate Studies and Scientific Research at the University of Bisha for sup                                                                                                                                                ‑
                            porting this work through the Fast‑ Track Research Support Program.
                            Author Contributions
                            DS: Conceptualization, Investigation, Data curation, Formal analysis, Methodology, Resources, Software, Validation,
                            Visualization, Writing, review. HX: Super vision, Conceptualization, Resources, Investigation, Project administration, Valida‑
                            tion, review. RAS: Conceptualization, Investigation, Writing, Editing, Validation, review. BA: Conceptualization, Writing,
                            Investigation, Validation, review, Resources. LF: Conceptualization, Writing, Investigation, Validation, review, Resources.
                            MA: Conceptualization, Investigation, review. AR: Conceptualization, Writing, Investigation, Validation, review, Resources.
                            Funding
                            This work was supported in part by the National Natural Science Foundation of China under Grant 62172342, in partby
                            the Natural Science Foundation of Hebei Province under Grant F2022105027, in part by the Natural Science Founda‑tion
                            of Sichuan Province under Grants 2022NSFSC0568 and 2022NSFSC0944, and in part by the Fundamental ResearchFunds
                            for the Central Universities, P. R. China.
                            Data availibility
                            The paper exclusively employs publicly accessible datasets.

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                                                       Page 22 of 24
                                   Declarations
                                   Ethics approval and consent to participate
                                   Not applicable.
                                   Consent for publication
                                   Not applicable.
                                   Competing interests
                                   The authors declare that they have no competing interests.
                                   Received: 5 July 2024   Accepted: 12 November 2024
                                   References
                                    1.                                                         Chen J, Zhang L. A sur vey and systematic assessment of computational methods for drug response prediction. Brief
                                           Bioinform. 2021;22(1):232–46.
                                    2.                                                         Friedman AA, Letai A, Fisher DE, Flaherty K T. Precision medicine for cancer with next‑   generation functional diagnos‑
                                           tics. Nat Rev Cancer. 2015;15(12):747–56.
                                    3.                                                         Sagingalieva A, Kordzanganeh M, Kenbayev N, Kosichkina D, Tomashuk T, Melnikov A. Hybrid quantum neural
                                           network for drug response prediction. Cancers. 2023;15(10):2705.
                                    4.                                                         Iorio F, Knijnenburg TA, Vis DJ, Bignell GR, Menden MP, Schubert M, Aben N, Gonçalves E, Barthorpe S, Lightfoot H,
                                           etal. A landscape of pharmacogenomic interactions in cancer. Cell. 2016;166(3):740–54.
                                    5.                                                         Barretina J, Caponigro G, Stransky N, Venkatesan K, Margolin AA, Kim S, Wilson CJ, Lehár J, Kr yukov GV, Sonkin
                                           D, etal. The cancer cell line encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature.
                                           2012;483(7391):603–7.
                                    6.                                                         Kittaneh M, Montero AJ, Glück S. Molecular profiling for breast cancer: a comprehensive review. Biomark Cancer.
                                           2013;5:9455.
                                    7.                                                         Smith SE, Mellor P, Ward AK, Kendall S, McDonald M, Vizeacoumar FS, Vizeacoumar FJ, Napper S, Anderson DH.
                                           Molecular characterization of breast cancer cell lines through multiple omic approaches. Breast Cancer Res.
                                           2017;19(1):1–12.
                                    8.                                                         Partin A, Brettin TS, Zhu Y, Nar ykov O, Clyde A, Overbeek J, Stevens RL. Deep learning methods for drug response
                                           prediction in cancer: predominant and emerging trends. Front Med. 2023;10:1086097.
                                    9.                                                         Huang X, Pan W. Linear regression and two‑   class classification with gene expression data. Bioinformatics.
                                           2003;19(16):2072–8.
                                    10.                                                         Ghosh D. Penalized regression models for the classification of tumors from gene expression data; 2002. Preprint,
                                           available at http:// www. sph. umich. edu/ ~ghoshd/ COMPB IO/ POPTS CORE.
                                    11.                                                         Wu S‑                                            W, Pan Q, Chen T. Research on diagnosis‑related group grouping of inpatient medical expenditure in colorec‑
                                           tal cancer patients based on a decision tree model. World J Clin Cases. 2020;8(12):2484.
                                    12.                                                         Suenderhauf C, Hammann F, Huwyler J. Computational prediction of blood–brain barrier permeability using deci‑
                                           sion tree induction. Molecules. 2012;17(9):10429–45.
                                    13.                                                         Mistr y P, Neagu D, Trundle PR, Vessey JD. Using random forest and decision tree models for a new vehicle prediction
                                           approach in computational toxicology. Soft Comput. 2016;20:2967–79.
                                    14.                                                         Maltarollo VG, Kronenberger T, Espinoza GZ, Oliveira PR, Honorio KM. Advances with support vector machines for
                                           novel drug discover y. Expert Opin Drug Discov. 2019;14(1):23–33.
                                    15.                                                         Huang S, Cai N, Pacheco PP, Narrandes S, Wang Y, Xu W. Applications of support vector machine (svm) learning in
                                           cancer genomics. Cancer Genom Proteomics. 2018;15(1):41–51.
                                    16.                                                         Khan J, Wei JS, Ringner M, Saal LH, Ladanyi M, Westermann F, Berthold F, Schwab M, Antonescu CR, Peterson C, etal.
                                           Classification and diagnostic prediction of cancers using gene expression profiling and artificial neural networks.
                                           Nat Med. 2001;7(6):673–9.
                                    17.                                                         Menden MP, Iorio F, Garnett M, McDermott U, Benes CH, Ballester PJ, Saez‑Rodriguez J. Machine learning prediction
                                           of cancer cell sensitivity to drugs based on genomic and chemical properties. PLoS ONE. 2013;8(4):61318.
                                    18.                                                         Marah BD, Jing Z, Ma T, Alsabri R, Anaadumba R, Al‑Dhelaan A, Al‑Dhelaan M. Smartphone architecture for edge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ‑
                                           centric iot analytics. Sensors. 2020;20(3):892.
                                    19.                                                         Al‑Sabri R, Gao J, Chen J, Oloulade BM, Lyu T. Multi‑                                                                                   view graph neural architecture search for biomedical entity and
                                           relation extraction. IEEE/ACM Trans Comput Biol Bioinform. 2022;20:1221–33.
                                    20.                                                         Al‑Sabri R, Gao J. Lamad: a linguistic attentional model for arabic text diacritization. In: Findings of the association for
                                           computational linguistics: EMNLP 2021, 2021:3757–3764.
                                    21.                                                         Ma T, Wang H, Zhang L, Tian Y, Al‑Nabhan N. Graph classification based on structural features of significant nodes
                                           and spatial convolutional neural networks. Neurocomputing. 2021;423:639–50.
                                    22.                                                         Lisboa PJ. A review of evidence of health benefit from artificial neural networks in medical inter vention. Neural
                                           Netw. 2002;15(1):11–39.
                                    23.                                                         Amato F, López A, Peña‑Méndez EM, Vaňhara P, Hampl A, Havel J. Artificial neural networks in medical diagnosis.
                                           Hoboken: Elsevier; 2013.
                                    24.                                                         Bouvrie J. Notes on convolutional neural networks; 2006.
                                    25.                                                         Cha YJ, Jang WI, Kim M‑S, Yoo HJ, Paik EK, Jeong HK, Youn S‑M. Prediction of response to stereotactic radiosurger y for
                                           brain metastases using convolutional neural networks. Anticancer Res. 2018;38(9):5437–45.
                                    26.                                                         Zhang F, Wang M, Xi J, Yang J, Li A. A novel heterogeneous network‑based method for drug response prediction in
                                           cancer cell lines. Sci Rep. 2018;8(1):1–9.

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                                                                                                              Page 23 of 24
                                     27.                                                         Liu H, Lin H, Shen C, Yang L, Lin Y, Xu B, Yang Z, Wang J, Sun Y. Drug repositioning for sars‑                                                                                          cov‑2 based on graph
                                             neural network. In: Proceedings of the IEEE international conference on bioinformatics and biomedicine, 2020, pp
                                             319–322.
                                     28.                                                         Feng Y, Yu H, Feng Y, Shi J. Directed graph attention networks for predicting asymmetric drug–drug interactions.
                                             Briefings Bioinform. 2022;23(3):bbac151.
                                     29.                                                         Nguyen T, Nguyen GT, Nguyen T, Le D‑H. Graph convolutional networks for drug response prediction. IEEE/ACM
                                             Trans Comput Biol Bioinf. 2021;19(1):146–54.
                                     30.                                                         Li X, Liu X, Lu L, Hua X, Chi Y, Xia K. Multiphysical graph neural network (MP‑   GNN) for COVID‑19 drug design. Brief‑
                                             ings Bioinform. 2022;23(4):bbac231.
                                     31.                                                         Liu X, Song C, Huang F, Fu H, Xiao W, Zhang W. Graphcdr: a graph neural network method with contrastive learning
                                             for cancer drug response prediction. Briefings Bioinform. 2022;23(1):bbab457.
                                     32.                                                         Peng W, Chen T, Liu H, Dai W, Yu N, Lan W. Improving drug response prediction based on two‑space graph convolu‑
                                             tion. Comput Biol Med. 2023;158: 106859.
                                     33.                                                         Sheng J, Li F, Wong ST. Optimal drug prediction from personal genomics profiles. IEEE J Biomed Health Inform.
                                             2015;19(4):1264–70.
                                     34.                                                         Cho HJ, Zhao J, Jung SW, Ladewig E, Kong D‑S, Suh Y        ‑L, Lee Y, Kim D, Ahn SH, Bordyuh M, etal. Distinct genomic
                                             profile and specific targeted drug responses in adult cerebellar glioblastoma. Neuro Oncol. 2019;21(1):47–58.
                                     35.                                                         El Rassy E, Pavlidis N. The current evidence for a biomarker‑based approach in cancer of unknown primar y. Cancer
                                             Treat Rev. 2018;67:21–8.
                                     36.                                                         Ross JS, Torres‑Mora J, Wagle N, Jennings TA, Jones DM. Biomarker                                       ‑based prediction of response to therapy for
                                             colorectal cancer: current perspective. Am J Clin Pathol. 2010;134(3):478–90.
                                     37.                                                         Niu N, Wang L. Invitro human cell line models to predict clinical response to anticancer drugs. Pharmacogenomics.
                                             2015;16(3):273–85.
                                     38.                                                         Maeser D, Gruener RF, Huang RS. oncopredict: an r package for predicting invivo or cancer patient drug response
                                             and biomarkers from cell line screening data. Brief Bioinform. 2021;22(6):260.
                                     39.                                                         Dong Z, Zhang N, Li C, Wang H, Fang Y, Wang J, Zheng X. Anticancer drug sensitivity prediction in cell lines from
                                             baseline gene expression through recursive feature selection. BMC Cancer. 2015;15:1–12.
                                     40.                                                         Wang L, Li X, Zhang L, Gao Q. Improved anticancer drug response prediction in cell lines using matrix factorization
                                             with similarity regularization. BMC Cancer. 2017;17:1–12.
                                     41.                                                         Chen X, Li T‑H, Zhao Y, Wang C‑                                                                                            C, Zhu C‑                                                                                                                      C. Deep‑belief network for predicting potential mirna‑                                               disease associations.
                                             Brief Bioinform. 2021;22(3):186.
                                     42.                                                         Ha J, Park C, Park C, Park S. Imipmf : Inferring mirna‑                                     disease interactions using probabilistic matrix factorization. J
                                             Biomed Inform. 2020;102: 103358.
                                     43.                                                         Ha J, Park S. Ncmd: Node2vec‑based neural collaborative filtering for predicting mirna‑                                                                               disease association. IEEE/
                                             ACM Trans Comput Biol Bioinf. 2023;20(2):1257–68. https:// doi. org/ 10. 1109/ TCBB. 2022. 31919 72.
                                     44.                                                         Ha J. Mdmf : predicting mirna‑                                     disease association based on matrix factorization with disease similarity constraint. J
                                             Person Med. 2022;12(6):885.
                                     45.                                                         Ha J. Smap: similarity‑based matrix factorization framework for inferring mirna‑                                                              disease association. Knowl Based
                                             Syst. 2023;263: 110295.
                                     46.                                                         Ha J, Park C. Mlmd: metric learning for predicting mirna‑                                     disease associations. IEEE Access. 2021;9:78847–58.
                                     47.                                                         Ha J. Lncrna expression profile ‑based matrix factorization for identifying lncrna‑                                               disease associations. IEEE Access;
                                             2024.
                                     48.                                                         Yang Y, Li P. Gpdrp: a multimodal framework for drug response prediction with graph transformer. BMC Bioinform.
                                             2023;24(1):484.
                                     49.                                                         Wang Z, Zhou Y, Zhang Y, Mo YK, Wang Y. Xmr: an explainable multimodal neural network for drug response predic‑
                                             tion. Front Bioinform. 2023;3:1164482.
                                     50.                                                         Liu X, Song C, Huang F, Fu H, Xiao W, Zhang W. Graphcdr: a graph neural network method with contrastive learning
                                             for cancer drug response prediction. Brief Bioinform. 2023;23(1):457.
                                     51.                                                         Wang R, Li F, Liu S, Li W, Chen S, Feng B, Jin D. Adaptive multi‑                                                                                                  channel deep graph neural networks. Symmetr y.
                                             2024;16(4):406.
                                     52.                                                         Yang W, Soares J, Greninger P, Edelman EJ, Lightfoot H, Forbes S, Bindal N, Beare D, Smith JA, Thompson IR, etal.
                                             Genomics of drug sensitivity in cancer (gdsc): a resource for therapeutic biomarker discover y in cancer cells. Nucleic
                                             Acids Res. 2012;41(D1):955–61.
                                     53.                                                         Kim S, Chen J, Cheng T, Gindulyte A, He J, He S, Li Q, Shoemaker BA, Thiessen PA, Yu B, etal. Pubchem 2019 update:
                                             improved access to chemical data. Nucleic Acids Res. 2019;47(D1):1102–9.
                                     54.                                                         Liu P, Li H, Li S, Leung K‑S. Improving prediction of phenotypic drug response on cancer cell lines using deep convo          ‑
                                             lutional network. BMC Bioinform. 2019;20(1):1–14.
                                     55.                                                         Wu Z, Pan S, Chen F, Long G, Zhang C, Philip SY. A comprehensive sur vey on graph neural networks. IEEE Trans
                                             Neural Netw Learn Syst. 2020;32(1):4–24.
                                     56.                                                         Liu P, Li H, Li S, Leung K. Improving prediction of phenotypic drug response on cancer cell lines using deep convolu‑
                                             tional network. BMC Bioinform. 2019;20(1):408–140814.
                                     57.                                                         Kipf TN, Welling M. Semi‑super vised classification with graph convolutional networks. In: Proceedings of the 5th
                                             international conference on learning representations, Toulon, France, April 24–26, Conference Track Proceedings,
                                             2017:1263–1272
                                     58.                                                         Xu K, Hu W, Leskovec J, Jegelka S. How power ful are graph neural networks? In: Proceedings of the 7th international
                                             conference on learning representations (Poster), 2019, pp 826–842. ICLR
                                     59.                                                         Velickovic P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph attention networks. In: Proceedings of the 6th
                                             international conference on learning representations (Poster), 2018;34, pp 10903–10914
                                     60.                                                         Kim D, Oh A. How to find your friendly neighborhood: graph attention design with self‑super vision. In: 9th interna‑
                                             tional conference on learning representations, ICLR 2021,Virtual Event, Austria, May 3–7, 2021 (2021). https:// openr
                                             eview. net/ forum? id= Wi5KU NlqWty

Saeed et al. BMC Bioinformatics           (2025) 26:19                                                                                                                             Page 24 of 24
                        Publisher’s Note
                        Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.
                        Dhekra Saeed                                                                                                 is Ph.D. Scholar at the School of Computing and Artificial Intelligence at Southwest Jiao‑
                        tong University (SWJTU), Chengdu, China. She completed her Masters in Computer Science and Technol‑
                        ogy from Nanjing University of Aeronautics and Astronautics, China, in 2017. Her research interests include
                        bioinformatics, automatic machine learning, and graph neural networks.
                        Huanlai Xing                                                           received a Ph.D. degree in computer science from the University of Nottingham (Super‑
                        visor: Dr. Rong Qu), Nottingham, U.K., in 2013. He was a Visiting Scholar in Computer Science at the Uni‑
                        versity of Rhode Island (Supervisor: Dr. Haibo He), USA, in 2020–2021. Huanlai Xing is with the School of
                        Computing and Artificial Intelligence, Southwest Jiaotong University (SWJTU), and Tangshan Institute of
                        SWJTU. He was on the Editorial Board of SCIENCE CHINA INFORMATION SCIENCES. He was a member of
                        several international conference programs and senior program committees, such as ECML‑PKDD, Mobi‑
                        Media, ISCIT, ICCC, TrustCom, IJCNN, and ICSINC. His research interests include semantic communication,
                        representation learning, data mining, reinforcement learning, machine learning, network function virtual‑
                        ization, and software ‑   defined networking.
                        Barakat AlBadani                                                                             received a B.S. degree in computer science from Sana’a University, Yemen in 2008,
                        and a Master ’s degree in Computer Science and Information Engineering, from Hohai University, Nanjing,
                        China, in 2019. He is currently a PhD student at the School of Computer Science and Engineering, Central
                        South University, Changsha, China. His current research interests include natural language processing and
                        machine learning.
                        Li Feng                                                                                          received his Ph.D. degree from Xi’an Jiaotong University under the super vision of Prof. Xiaohong
                        Guan (Academian of CAS, IEEE Fellow). He is a Research Professor and Ph.D. supervisor with the School
                        of Computing and Artificial Intelligence, Southwest Jiaotong University, Chengdu. His research interests
                        include ar tificial intelligence, cyber security and their applications.
                        Raeed Al‑Sabri                                           received a B.S. degree in Information Technology from Thamar University, Thamar,
                        Yemen, and an M.S. degree in Computer Science and Technology from Nanjing University of Information
                        Science and Technology, Nanjing, Jiangsu, China. He is currently a Ph.D. candidate at the School of Com‑
                        puter Science and Engineering, Central South University, Changsha, China. His research interests are natu‑
                        ral language processing, automatic machine learning, and graph neural networks.
                        Monir Abdullah                                                                                                                received his M.Sc. and Ph.D. degrees in parallel and distributed computing from Uni‑
                        versity of Putra Malaysia, Malaysia. He is currently an Associate Professor with the Department of Computer
                        Science and Artificial Intelligence, University of Bisha, Saudi Arabia. He has more than 15 years of academic
                        and teaching experience, as well as expertise in quality assurance and accreditation in different universi‑
                        ties. His research interests include parallel and distributed computing, optimization techniques, machine
                        learning, deep learning, cyber security and Internet of Things.
                        Amir Rehman                                                                               is a Ph.D. scholar at the School of Computing and Artificial Intelligence in Chengdu,
                        China. He has completed his Masters in Information Technology from the University of Gujrat, Lahore cam‑
                        pus, Pakistan. He has extensive experience in developing and teaching Computer Science to undergradu‑
                        ate and college students. His research mainly focuses on Machine and Deep learning, Intelligent Diagnosis,
                        and IoMT.

