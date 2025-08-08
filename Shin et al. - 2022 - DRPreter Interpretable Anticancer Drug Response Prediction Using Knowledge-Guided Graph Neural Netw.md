International Journal of

_**Molecular Sciences**_


_Article_
# **DRPreter: Interpretable Anticancer Drug Response Prediction** **Using Knowledge-Guided Graph Neural Networks** **and Transformer**


**Jihye Shin** **[1,†]** **, Yinhua Piao** **[2,†]** **, Dongmin Bang** **[1,3]** **, Sun Kim** **[1,2,4,5]** **and Kyuri Jo** **[6,]** *****


1 Interdisciplinary Program in Bioinformatics, Seoul National University, Seoul 08826, Korea
2 Department of Computer Science and Engineering, Institute of Engineering Research, Seoul National
University, Seoul 08826, Korea
3 AIGENDRUG Co., Ltd., Seoul 08826, Korea
4 Interdisciplinary Program in Artificial Intelligence, Seoul National University, Seoul 08826, Korea
5 MOGAM Institute for Biomedical Research, Yongin-si 16924, Korea
6 Department of Computer Engineering, Chungbuk National University, Cheongju 28644, Korea
***** Correspondence: kyurijo@chungbuk.ac.kr

                 - These authors contributed equally to this work.



**Citation:** Shin, J.; Piao, Y.; Bang, D.;


Kim, S.; Jo, K. DRPreter:


Interpretable Anticancer Drug


Response Prediction Using


Knowledge-Guided Graph Neural


Networks and Transformer. _Int. J._


_Mol. Sci._ **2022**, _23_ [, 13919. https://](https://doi.org/10.3390/ijms232213919)


[doi.org/10.3390/ijms232213919](https://doi.org/10.3390/ijms232213919)


Academic Editor: Pablo Minguez


Received: 30 September 2022


Accepted: 8 November 2022


Published: 11 November 2022


**Publisher’s Note:** MDPI stays neutral


with regard to jurisdictional claims in


published maps and institutional affil

iations.


**Copyright:** © 2022 by the authors.


Licensee MDPI, Basel, Switzerland.


This article is an open access article


distributed under the terms and


conditions of the Creative Commons


[Attribution (CC BY) license (https://](https://creativecommons.org/licenses/by/4.0/)


[creativecommons.org/licenses/by/](https://creativecommons.org/licenses/by/4.0/)


4.0/).



**Abstract:** Some of the recent studies on drug sensitivity prediction have applied graph neural
networks to leverage prior knowledge on the drug structure or gene network, and other studies have
focused on the interpretability of the model to delineate the mechanism governing the drug response.
However, it is crucial to make a prediction model that is both knowledge-guided and interpretable, so
that the prediction accuracy is improved and practical use of the model can be enhanced. We propose
an interpretable model called DRPreter (drug response predictor and interpreter) that predicts the
anticancer drug response. DRPreter learns cell line and drug information with graph neural networks;
the cell-line graph is further divided into multiple subgraphs with domain knowledge on biological
pathways. A type-aware transformer in DRPreter helps detect relationships between pathways and a
drug, highlighting important pathways that are involved in the drug response. Extensive experiments
on the GDSC (Genomics of Drug Sensitivity and Cancer) dataset demonstrate that the proposed
method outperforms state-of-the-art graph-based models for drug response prediction. In addition,
DRPreter detected putative key genes and pathways for specific drug–cell-line pairs with supporting
evidence in the literature, implying that our model can help interpret the mechanism of action of the
drug.


**Keywords:** transcriptomics; artificial intelligence; pharmacogenomics; human health; cancer; drug
sensitivity; graph neural networks; Explainable AI; precision medicine; drug discovery


**1. Introduction**


The advances in technology and scientific capability enable the acquisition of large
amounts of personal omics data at a reduced cost [ 1 ]. Consequently, there is a growing
interest in using individualized health data for precision medicine, leading to a number of
data-driven healthcare models [ 2 ]. Pharmacogenomics, one of the branches of precision
medicine, is the study of how a person’s genetic profile influences their response to medications [ 3, 4 ]. Prediction of drug response or efficacy using the omics data of patients before
the actual treatment is crucial because it can help increase clinical success and minimize
adverse drug effects by modifying dosages or selecting alternative medications based
on predicted value for personalized chemotherapy. However, obtaining patients’ tumor
tissues via surgical procedure or biopsy involves safety issues [ 5 ], and performing animal
experiments for clinical trials to infer human drug efficacy leads to ethical and financial
concerns [ 6 ]. In addition, even though correlating the drug response and omics data can
help improve understanding the drugs’ mechanisms of action [ 7 ], many candidate drugs



_Int. J. Mol. Sci._ **2022**, _23_ [, 13919. https://doi.org/10.3390/ijms232213919](https://doi.org/10.3390/ijms232213919) [https://www.mdpi.com/journal/ijms](https://www.mdpi.com/journal/ijms)


_Int. J. Mol. Sci._ **2022**, _23_, 13919 2 of 19


still fail to enter clinical trials during the drug discovery process due to an incomplete
understanding of the mechanisms [ 8, 9 ]. In this respect, an interpretable in silico model for
drug response prediction would be useful for numerous healthcare purposes, especially for
precision medicine and drug discovery [10].
Molecular profiles of cancer cell lines and high-throughput drug sensitivity screening
databases are publicly available [ 11 – 16 ], including CCLE (Cancer Cell Line Encyclopedia) [ 12 ] and GDSC (Genomics of Drug Sensitivity in Cancer) [ 13, 16 ]. Public databases
and improved computing technologies such as machine learning and deep learning have
contributed to the rapid development of models for predicting anticancer drug sensitivity
from cancer cell lines based on their genetic profiles.
The early studies in drug sensitivity prediction utilized machine-learning techniques [ 17 – 19 ] such as a random forest [ 20 ], support vector machine [ 21 ], and matrix
factorization [ 22, 23 ]. However, the traditional machine learning-based models can still
be improved in terms of predictive performance and generalizability [ 3, 24 ]. Matrixfactorization-based models leave nonlinear relationships unaddressed because they attempt to identify interactions between the drugs and cell lines using linear combinations
of latent features. With the capabilities of learning complex nonlinear functions and
high-dimensional representations from raw data, various deep learning techniques have
been utilized for predicting drug responses [ 24 ], and the overall predictive power of drug
sensitivity has been improved [ 25 ]. DeepDR [ 26 ] and MOLI [ 27 ] are drug-specific models
that only use cell features such as somatic mutations, gene expression profile, and copy
number variation to predict the IC50 values of each sample. tCNNs [ 28 ] introduced a
model to predict drug sensitivity for drug–cell pairs using SMILES (Simplified Molecular
Input Line Entry System) [ 29 ] sequences as drug features in addition to the genomic
profiles of cells. The models described above used vector representations in common for
describing cell or drug features.
Graph-based approaches have been introduced in drug-response prediction models
to take advantage of the structural information of a drug or a gene network. A drug
can be represented as a molecular graph consisting of a set of atoms (nodes) and a set of
bonds (edges), and the graph is transformed into a high-level representation by a neural
network [ 30, 31 ]. For example, GraphDRP [ 31 ] obtained drug embeddings using graph
convolutional networks, and cell line embeddings were derived from binary vectors of
genomic aberrations. The state of a cell line can also be characterized as a gene–gene
interaction network where genes (nodes) have node features from omics data such as gene
expression values [ 32 – 34 ]. Reference [ 33 ] proposed an end-to-end drug-response prediction
model, TGDRP, with cell-line graph embedding consisting of genes that have cancer-related
mutations and drug graph embedding obtained by a graph neural network. They also
proposed TGSA, which updates embeddings from TGDRP with similarity information
between cell lines and drugs and predicts the drug response again. Among graph-based
drug-response prediction models, NIHGCN [ 35 ] constructed a cell-line–drug heterogeneous network with cell-line gene expression and drug-fingerprint vectors as node features
to aggregate neighborhood interactions of drug and cell line. Then, there are two different
types of GCN layer for aggregating both homogeneous and heterogeneous neighbors’
information for drug response prediction. However, even though it is a graph-based model,
the biochemical structures of cell line and drug are not reflected in the model.
While the recent studies described above have introduced graphs into the deeplearning models to leverage structural information and improve prediction accuracy, the
models lack interpretability of the predicted results. Several methods tried to delineate the
mechanisms governing the drug responses, highlighting the important genes or high-level
subsystems, such as biological pathways that can cause changes in cellular phenotype.
SWnet [ 36 ] explored the interactions between genetic profile and the chemical structure
of drugs using self-attention and identified genes with the strongest predictive power.
Reference [ 37 ] proposed a multi-layer perceptron model called pathDNN which incorporates a layer of pathway nodes and quantified the activity of each pathway to explain its


_Int. J. Mol. Sci._ **2022**, _23_, 13919 3 of 19


effect on the drug response. DrugCell [ 9 ] obtained binary encodings of mutational status
via a visible neural network guided by a hierarchy of cell subsystems and measured the
predictive performance of the subsystems. Although pathDNN and DrugCell attempted
to construct an explainable model with a hierarchical structure, biological pathways were
implemented as gene sets rather than gene networks, indicating that domain knowledge in
gene–gene interactions is not fully reflected in the models.
According to the existing studies that suggest deep-learning models for drug response
prediction, it is helpful to incorporate graph representation for both drug and cell line
profiles, which enables detailed descriptions of compound structures and gene networks.
Moreover, the gene network can be dissected as a set of biological pathways that include
gene–gene interactions for each specific biological mechanism, which can help enhance
both prediction accuracy and interpretability. However, current interpretable models for
drug response prediction simply describe the network as gene-pathway layers, leaving the
interaction information inside the biological pathways unused. Here, we propose a novel
anticancer drug-response prediction model named DRPreter (drug response predictor and
interpreter) with key features as follows:


1. _Knowledge-guided cell representation with graphs._ DRPreter constructs a cell line network
as a set of subgraphs that correspond to cancer-related pathways for the detailed
representation of the biological mechanism.
2. _Interpretability of drug mechanisms of action._ Using the transformer’s encoder, the
interactions between drugs and pathways are derived from the model, and putative
key pathways for the drug mechanism can be highlighted.
3. _Enhanced performance._ DRPreter outperforms state-of-the-art drug-response prediction
models, as demonstrated by comparative experiments on the GDSC drug-sensitivity
dataset.


**2. Results and Discussion**


In this study, we developed a regression model to predict the half maximal inhibitory
concentration (IC50), normalized to natural logarithms, which is a representative indicator
of drug sensitivity. The following is a description of the graph configuration for cell lines
and drugs and the graphical abstract of DRPreter (Figure 1).
The main steps of DRPreter are: cell-line graphical representation, drug graphical
representation, and drug response prediction module. As a first step in creating the cellline graph representation, a template graph was created with all genes involved in the
selected 34 cancer pathways as nodes and gene–gene interactions between the genes as
edges. After that, each cancer pathway was used as a predefined subgraph of the template
graph, and each pathway embedding was derived using graph attention networks (GAT).
As a next step in creating drug-graph representation, we transformed SMILES-format
drug structures into graphs with atoms as nodes and bonds as edges, and used a graph
isomorphism network (GIN) as a graph encoder to generate drug embeddings. With the
34 pathway embeddings and drug embeddings obtained as inputs to the transformerbased cell line and drug fusion module, the embeddings were updated by reflecting
inter-pathway relationships and pathway-drug relationships in the model learning process. Updated pathway embeddings were combined into graph-level embeddings for the
entire cell line through concatenation, and graph-level drug embedding was obtained
by combining embeddings before and after passing through the transformer encoder. In
conjunction with two graph level embeddings, the IC50 of a given cell-line–drug pair was
predicted using a multi-layer perceptron.


_Int. J. Mol. Sci._ **2022**, _23_, 13919 4 of 19


**DRPreter: Drug Response PREdictor and interpreTER**



**Cell line graph representation**



**Drug response prediction module**



**Cell line graph construction** **Pathway-based subgraph encoding module** **Knowledge-guided fusion module** **Prediction module**












































|Col1|Col2|Raw pathway Drug-aware<br>embedding pathway embedding<br>… …<br>Transformer<br>Raw drug Cell line–aware<br>embedding drug embedding<br>+<br>Residual connection|Col4|Cell line<br>graph-level<br>embedding<br>MLP<br>IC50<br>Drug<br>graph-level<br>embedding|
|---|---|---|---|---|
|||…<br><br>**Transformer**<br>Raw pathway<br>embedding<br>Drug-aware<br>pathway embedding<br>Raw drug<br>embedding<br>Cell line–aware<br>drug embedding<br>Residual connection<br>~~+~~<br>…|||
|**SMILES format**<br>**drug structure**<br>**Feature**<br>**Node**<br>Atom type, Aromatic, Chirality, Degree,<br>Formal charge, Hybridization,<br>Number of Hydrogen, Implicit valence,<br>Radical electrons, Ring<br>**Edge**<br>Bond type<br>**Drug graph**𝐺𝑑<br>B(C(CC(C)C)NC(=<br>O)C(CC1=CC=CC=<br>C1)NC(=O)C2=NC<br>=CN=C2)(O)O<br>GIN<br>Drug encoder|**SMILES format**<br>**drug structure**<br>**Feature**<br>**Node**<br>Atom type, Aromatic, Chirality, Degree,<br>Formal charge, Hybridization,<br>Number of Hydrogen, Implicit valence,<br>Radical electrons, Ring<br>**Edge**<br>Bond type<br>**Drug graph**𝐺𝑑<br>B(C(CC(C)C)NC(=<br>O)C(CC1=CC=CC=<br>C1)NC(=O)C2=NC<br>=CN=C2)(O)O<br>GIN<br>Drug encoder|**SMILES format**<br>**drug structure**<br>**Feature**<br>**Node**<br>Atom type, Aromatic, Chirality, Degree,<br>Formal charge, Hybridization,<br>Number of Hydrogen, Implicit valence,<br>Radical electrons, Ring<br>**Edge**<br>Bond type<br>**Drug graph**𝐺𝑑<br>B(C(CC(C)C)NC(=<br>O)C(CC1=CC=CC=<br>C1)NC(=O)C2=NC<br>=CN=C2)(O)O<br>GIN<br>Drug encoder|**SMILES format**<br>**drug structure**<br>**Feature**<br>**Node**<br>Atom type, Aromatic, Chirality, Degree,<br>Formal charge, Hybridization,<br>Number of Hydrogen, Implicit valence,<br>Radical electrons, Ring<br>**Edge**<br>Bond type<br>**Drug graph**𝐺𝑑<br>B(C(CC(C)C)NC(=<br>O)C(CC1=CC=CC=<br>C1)NC(=O)C2=NC<br>=CN=C2)(O)O<br>GIN<br>Drug encoder|**SMILES format**<br>**drug structure**<br>**Feature**<br>**Node**<br>Atom type, Aromatic, Chirality, Degree,<br>Formal charge, Hybridization,<br>Number of Hydrogen, Implicit valence,<br>Radical electrons, Ring<br>**Edge**<br>Bond type<br>**Drug graph**𝐺𝑑<br>B(C(CC(C)C)NC(=<br>O)C(CC1=CC=CC=<br>C1)NC(=O)C2=NC<br>=CN=C2)(O)O<br>GIN<br>Drug encoder|
|**SMILES format**<br>**drug structure**<br>**Feature**<br>**Node**<br>Atom type, Aromatic, Chirality, Degree,<br>Formal charge, Hybridization,<br>Number of Hydrogen, Implicit valence,<br>Radical electrons, Ring<br>**Edge**<br>Bond type<br>**Drug graph**𝐺𝑑<br>B(C(CC(C)C)NC(=<br>O)C(CC1=CC=CC=<br>C1)NC(=O)C2=NC<br>=CN=C2)(O)O<br>GIN<br>Drug encoder|||||



**Figure 1.** An overview of DRPreter. In the graph representation sections, embeddings of pathway
subgraphs and drug molecule were obtained using GNN. With the obtained pathway embeddings
and drug embeddings as inputs to the transformer-based cell-line and drug fusion module, the
embeddings were updated by reflecting inter-pathway relationships and pathway-drug relationships
in the model learning process.


_2.1. Performance Comparison_

2.1.1. Dataset


In the cell line template graph, the initial feature of each gene node was derived
from transcriptomic data of each cell line obtained from the CCLE database version of
21Q4 [ 38 [] (https://portals.broadinstitute.org/ccle, accessed on 3 December 2021). The](https://portals.broadinstitute.org/ccle)
gene expression data were TPM values of the protein-coding genes for DepMap cell lines,
which were inferred from RNA-seq data using the RSEM tool and were provided after
log2 transformation, using a pseudo-count of 1; _log_ 2 (TPM+1) [ 38 ]. We assigned edges of
the graph as only those interactions with high reliability scores and a combined score of
at least 990 among the STRING (v11.5) [ 39 ] protein–protein interactions. The edges of the
template graph and each subgraph were all STRING protein–protein interactions. Only
the genes corresponding to each cancer-related pathway were obtained in KEGG, and the
genes corresponding to each pathway were used as nodes in the subgraph. The STRING
interactions were used as the edges connecting them. Pathways for constructing subgraphs
were selected in the following manner. The non-processed pathways listed in categories 6.1
[and 6.2 of the KEGG pathway database (https://www.genome.jp/kegg/pathway.html,](https://www.genome.jp/kegg/pathway.html)
accessed on 16 April 2022) were categorized according to the cancer types. These pathways
include common subpathways related to cell signaling, the cell cycle, and apoptosis, which
are key in various types of cancer. Consequently, if the cancer pathways provided by
KEGG are used as they are, the overlap between the pathways will be excessive, and
the meaning of learning for each pathway diminishes. Additionally, KEGG provides
information on detailed pathways associated with each cancer type pathway. There were
a total of 84 detailed pathways categorized by function. Among these pathways, we
eliminated duplicate pathways, metabolic pathways, non-cancer disease pathways, viral
infection pathways, and pathways with fewer than 10 genes or gene–gene interaction edges.
Additionally, the focal adhesion pathway (hsa04510) was also eliminated because 91% of


_Int. J. Mol. Sci._ **2022**, _23_, 13919 5 of 19


the genes constituting this pathway were included in the remaining pathways. The finally
selected 34 cancer-related detailed pathway list can be found in Table A1. For drug graph
construction, we obtained SMILES strings from PubChem [40].
For the performance comparison experiment, we compared our method with state-ofthe-art GNN-based drug-response prediction models obtaining either cell-line or drug embedding using a homogenous biochemical structure-based graph: GraphDRP, TGDRP, and
TGSA. As the initial feature for each gene node, GraphDRP uses mutation (mut) and copy
number variation (cnv); and TGDRP and TGSA use mut, cnv, and gene expression (exp).
As GraphDRP represents cell lines as one-dimensional binary vectors, a one-dimensional
CNN is used to get their embeddings. Cell lines and drugs are represented in graph format
in TGDRP and TGSA, and the embeddings are obtained by an GNN. The cancer driver
genes from COSMIC were selected as the genes to represent the cell lines in all baseline
models [ 41 ]. The COSMIC database provides information about mutation-containing
genes involved with cancer, and how these mutations can cause cancer. We selected
[702 COSMIC Cancer Gene Census (https://cancer.sanger.ac.uk/cosmic/census?tier=all,](https://cancer.sanger.ac.uk/cosmic/census?tier=all)
accessed on 3 December 2021) genes; all three omics data types are provided in CCLE. We
used the genes equally for the baseline model execution. Moreover, the types of cell lines
and drugs used in this study were the same as in the TGDRP and TGSA. The data type
used by our model differed from that of every baseline model, and ours used the most
numerous omics types among them. To use only cell-line–drug pairs with three types of
omics data available, intensive filtering was done on cell lines and drugs. Since all omics
data had to be imported for baseline model execution, the same cell-line–drug pair was
used as in the most data-intensive models. Consequently, the performance test consisted
of 580 cancer cell lines that can obtain omics data from CCLE and 170 anticancer drugs
provided by GDSC2. The total number of possible cell-line–drug pairs was 82,833 with
log-normalized IC50 values.
In addition, as performance comparisons with deep learning, we added random forest
and support vector machine (SVM) as baseline models for comparative analysis. For a fair
comparison, we used the same features and preprocessed the data to feed into traditional
machine-learning methods. For cell line embedding, we concatenated all gene expression
vectors, resulting in a one-dimensional vector, and for drug embedding, since different
drugs have different atoms, we simply the sum of each atom embedding, resulting in a
one-dimensional vector. Finally, we concatenated cell line embedding and drug embedding
to one-dimensional embedding and fed it into the models. The nodes constituting the cellline graphs of the existing GNN-based drug-response prediction models were configured
according to the settings in each comparison paper.


2.1.2. Experimental Setups


In the regression experiments for predicting natural log-transformed IC50 values
based on drug and cancer cell line profiles, we used four standard evaluation metrics to
compare the results of different models by computing the statistical correlation and accuracy
of predicted and observed IC50 values. The metrics included the Pearson correlation
coefficient (PCC), Spearman correlation coefficient (SCC), root absolute error (MAE), and
mean-squared error (MSE). PCC measures the linear correlation of observed and predicted
IC50 values. SCC is a non-parametric measure for rank correlation between observed and
predicted IC50 values. MSE and MAE directly measure the difference between observed
and predicted IC50 values.


2.1.3. Rediscovered Responses of Known Pairs


All possible cell-line–drug pairs were randomly divided into training, validation, and
test datasets in an 8:1:1 ratio, and the experiments were conducted repeatedly on 10 random
seeds. For each model, the test performance was averaged over the seeds and is reported
as mean _±_ standard deviation. Comparing the results of different models was based on
four common evaluation indicators. The mean-squared error and mean absolute error


_Int. J. Mol. Sci._ **2022**, _23_, 13919 6 of 19


between the predicted IC50 and the true IC50; and the Pearson correlation coefficient and
Spearman correlation coefficient of each IC50 distribution were used as evaluation criteria.
Compared to the baseline model we selected above, we conducted an ablation study to
examine each part of DRPreter’s effectiveness (Table 1). Based on the results of the ablation
study, we assume that network information from biological pathways helped improve the
performance of our model. It has also been found that the ability to relate the cell line in
its pre-drug treatment state to the drug through the transformer has a significant effect on
performance improvement. In addition, DRPreter showed a performance improvement of
about 20% in MSE compared with the next best model (Table 2). We also conducted internal
validation using a 10-fold cross-validation experiment on the original dataset using CCLE
and GDSC2. The validation set was selected by five random seeds (Table 3).


**Table 1.** Model ablation studies with different settings.


**Structural**
**Settings of** **Data** **MSE (** _**↓**_ **)** **MAE (** _**↓**_ **)** **PCC (** _**↑**_ **)** **SCC (** _**↑**_ **)**
**DRPreter**


Template graph COSMIC [1] 0.8926 _±_ 0.0363 0.6909 _±_ 0.0146 0.9423 _±_ 0.0027 0.9196 _±_ 0.0034
Template graph Pathway [2] 0.8536 _±_ 0.0420 0.6759 _±_ 0.0161 0.9449 _±_ 0.0032 0.9224 _±_ 0.0035
Pathway Pathway [2] 0.8645 _±_ 0.0277 0.6791 _±_ 0.0113 0.9446 _±_ 0.0014 0.9233 _±_ 0.0008
TransformerPathway + Pathway [2] 0.8302 _±_ 0.0156 **0.6676** _**±**_ **0.0051** 0.9465 _±_ 0.0015 0.9242 _±_ 0.0015

Pathway +
Transformer + Pathway [2] **0.8251** _**±**_ **0.0122** 0.6682 _±_ 0.0047 **0.9467** _**±**_ **0.0013** **0.9248** _**±**_ **0.0014**
Similarity


1 COSMIC: 702 COSMIC genes. 2 Pathway: 2369 genes of 34 cancer-related pathways. The best performance is

shown in bold for each metric.


**Table 2.** Performance comparison with baseline models.


**Model** **Cell Encoder** **Data** **MSE (** _**↓**_ **)** **MAE (** _**↓**_ **)** **PCC (** _**↑**_ **)** **SCC (** _**↑**_ **)**


SVM [1]   - Pathway 8.5780 _±_ 2.0615 2.2976 _±_ 0.3005 0.5282 _±_ 0.0355 0.4471 _±_ 0.0476
RF [2]   - Pathway 1.6711 _±_ 0.0422 0.9608 _±_ 0.0100 0.8887 _±_ 0.0021 0.8497 _±_ 0.0034
GraphDRP 1D CNN COSMIC 1.0110 _±_ 0.0157 0.7618 _±_ 0.0083 0.9386 _±_ 0.0018 0.9151 _±_ 0.0021
TGDRP GNN COSMIC 0.9004 _±_ 0.0341 0.6933 _±_ 0.0148 0.9417 _±_ 0.0026 0.9188 _±_ 0.0040

TGSA GNN COSMIC 0.8955 _±_ 0.0536 0.6913 _±_ 0.0238 0.9425 _±_ 0.0043 0.9201 _±_ 0.0051

KnowledgeDRPreter guided Pathway **0.8251** _**±**_ **0.0122** **0.6682** _**±**_ **0.0047** **0.9467** _**±**_ **0.0013** **0.9248** _**±**_ **0.0014**
GNN


1 SVM: support vector machine. 2 RF: random forest. The best performance is shown in bold for each metric.


**Table 3.** Internal validation using 10-fold cross-validation on 5 random seeds.


**Comparison**
**Data** **MSE (** _**↓**_ **)** **MAE (** _**↓**_ **)** **PCC (** _**↑**_ **)** **SCC (** _**↑**_ **)**
**Models**


TGDRP COSMIC [1] 1.9398 _±_ 0.0231 1.0435 _±_ 0.0058 0.8665 _±_ 0.0026 0.8164 _±_ 0.0074


DRPreter Template COSMIC [1] 1.9665 _±_ 0.0323 1.0435 _±_ 0.0089 0.8685 _±_ 0.0018 0.8232 _±_ 0.0022
graph

DRPreter Template Pathway [2] 1.9276 _±_ 0.0495 1.0351 _±_ 0.0130 0.8711 _±_ 0.0034 0.8270 _±_ 0.0042
graph

DRPreter w/o
Trans [3] and Pathway [2] 1.8536 _±_ 0.0548 1.0085 _±_ 0.0123 **0.8820** _**±**_ **0.0049** **0.8445** _**±**_ **0.0094**
Similarity

DRPreter w/o
Pathway [2] **1.8317** _**±**_ **0.0276** **1.0076** _**±**_ **0.0067** 0.8778 _±_ 0.0018 0.8356 _±_ 0.0022
similarity


1 COSMIC: 702 COSMIC genes. 2 Pathway: 2369 genes of 34 cancer-related pathways. 3 Trans: transformer-based
cell-line–drug fusion module.


_Int. J. Mol. Sci._ **2022**, _23_, 13919 7 of 19


_2.2. Case Study_
2.2.1. Interpolation of Unknown Values


The method of missing values prediction has been widely used in drug-responseprediction studies [ 28, 30, 31, 33 ] to identify whether the model is capable of inductive
prediction. For evaluating the inductive predictability of our model, we trained with all the
known cell-line–drug pairs and predicted values without experimental results of pairs in
the GDSC2 database. There were a total of 98,600 pairs using 580 cancer cell lines and 170
drugs, but 15,767 cell lines were not covered by our data due to filtering because of a lack
of omics data or due to the absence of drug response experiments in GDSC. The model
with the highest performance was used to predict missing drug response values.
We illustrate the distributions of known IC50 values in GDSC2 and the predicted
values of our model (Figure 2). The box plots are grouped by drugs, and each box represents
the distribution of the IC50 values within a cell line. We displayed the drugs with the top 10
highest and top 10 lowest median IC50 value. After conducting Mann–Whitney Wilcoxon
test for each drug distribution, 18 drugs among the 20 selected drugs showed no significant
difference between the GDSC2 and predicted unknown IC50 value distribution. The result
implies the predicted missing IC50 values follow the measured value distribution.































**Figure 2.** Box plot of drug-specific IC50 distributions of cell lines. The distribution of GDSC2 data
(blue) compared with predicted missing IC50 values (orange). The 10 drugs with the highest median
IC50 values and the 10 drugs with the lowest median were selected. Among the 20 drugs, IC50 value
distributions of 18 drugs showed no significant differences through the Mann–Whitney Wilcoxon
Test. ns: not significant, *: 0.01 < _p_ -value < 0.05.


The total predicted missing values using our model can be found in Table S1 of the
Supplementary Data.
Not knowing the actual values for these missing pairs, we conducted literature
searches to assess our predictions. Bortezomib had the smallest overall IC50 distribution, and the most sensitive cell-line pair was LP-1 in our model. LP-1 is a cell line derived
from the peripheral blood of a multiple myeloma patient. Bortezomib is a proteasome
inhibitor that is widely used in patients with multiple myeloma [ 42, 43 ]. Rapamycin was not
included among the top 10 sensitive drugs in the known GDSC data but in our predicted
values, so we analyzed it further. In our study, rapamycin was most sensitive to the MV-4-11
cell line. The MV-4-11 cells are macrophages that were isolated from the blast cells of a
biphenotypic B myelomonocytic leukemia patient. Rapamycin can inhibit leukemic activity
in acute myeloid leukemia by mTOR inhibition through the blockade in G0/G1 phase of
the cell cycle [44].


_Int. J. Mol. Sci._ **2022**, _23_, 13919 8 of 19


Based on the biological processes at the cellular and molecular level of cancer cells and
drugs, DRPreter can make inductive predictions for cell lines and drugs when there are no
known responses and seems to have the potential to select candidates for drug treatment.


2.2.2. Gradient-Weighted Gene Nodes Interpretation


It is essential for drug-response prediction methods to capture significant biological
implications and to make accurate predictions. A gene-level analysis was performed first
to determine whether the model was taking into account genes that are known as drug
targets, involved in target pathways, or biomarkers of disease. We prioritized genes from an
input drug and cancer–cell-line pair by scoring each gene with a gradient-weighted extent
to check whether it is drug-target-related. The importance score of each gene node was
determined by GradCAM, which is a widely utilized technique to produce explanations of
model decisions [ 45 ], and we considered the score as the extent of its contribution. In our
model, GradCAM determined the influence of input gene nodes on the label by tracing
back the gradient backpropagation process of the model for predicting IC50 value. Table 4
shows the top five most significant genes of each cell-line–drug pair in the test dataset.


**Table 4.** Gradient-based gene importance analysis.


**ln(IC50)**
**Drug** **Cell Line** **Disease** **Top 5 Significant Genes**

**True** **Predicted**


Afatinib GMS-10 Glioblastoma _ACTR3B_, _PRR5_, _PRKCZ_, 0.5372 0.5324
_**ERBB2**_, _LTBR_

Vinblastine NCI-H1792 NSCLC _CYP7A1_, _GTF2H2_, _DVL2_, _−_ 5.9258 _−_ 5.27633
_RAB5B_, _**TP53**_

Docetaxel PANC0327 Pancreatic cancer _**CLDN18**_, _SOX17_, _FGF19_, _−_ 3.7668 _−_ 3.8204
_WNT7A_, _CDH5_

Rapamycin IGR1 Melanoma _TYRP1_, _DCT_, _TYR_, _FRZB_, _**CDK2**_ _−_ 1.6747 _−_ 1.7651



Lung squamous cell
Bortezomib EBC-1 carcinoma Derived from

metastatic site: Skin



_SHC4_, _TNR_, _IL17RA_, _**MAPK12**_, _−_ 5.7714 _−_ 6.0714
_SMURF1_



Genes in bold are direct targets of drugs, are involved in target pathways, or are biomarkers of disease.


As verified by literature searches, the bold genes in Table 4 are the target genes or genes
associated with the target pathway for each drug–cell-line pair. The targets were obtained
from DrugBank [ 46 ] and GDSC, and the genes corresponding to the target pathway were
obtained from GeneCards [ 47 ] and Harmonizome [ 48 ]. Afatinib is a irreversible ErbB family
blocker [ 49 ] that targets _EGFR_ and _ERBB2_, and its target pathway is the EGFR signaling
pathway. Our model found _ERBB2_ as a significant gene of the afatinib pair. Among the
other top five genes, _LTBR_ was found to be related with tumor treatment for its potential
in triggering apoptosis of tumor cells or anti-tumor immune response [ 50 ]. As with the
majority of cancers, _TP53_ is the most common mutated gene, showing a predominant clonal
expression in Non-Small-Cell Lung Cancer (NSCLC) [ 51 ]. Additionally, microtubule-active
drugs, including vinblastine, are known to induce apoptosis through inducing expression
of p53 protein [ 52 ]. It is known to be possible to use _CLDN18_ as an early-stage indicator
of pancreatic ductal carcinogenesis and to study _CLDN18_ ’s regulatory mechanisms for
uncovering key pathways such as the PKC pathway of pancreatic cancer [ 53 ]. _WNT7A_,
the fourth-ranked gene, shows relation with docetaxel for Wnt signaling, playing a role
in docetaxel resistance [ 54 ]. _CDK2_ corresponds to the mTOR signaling pathway, which
is the target pathway of rapamycin. Additionally, an in vivo experiment reported that
upregulation of TYRP1 and TYR proteins may explain the melanogenesis of rapamycintreated melanoma cells [ 55 ]. The use of bortezomib and paclitaxel suggests the potential
for rationally designed treatments for solid tumors with MAPK pathway activation [56].


_Int. J. Mol. Sci._ **2022**, _23_, 13919 9 of 19


2.2.3. Pathway-Level Interpretation Using the Transformer


We examined which pathways were stimulated in various cancer types that are
sensitive to drugs and some that are not, and whether our model could capture such
meaningful pathways. The self-attention score from the transformer-based structure
(Figure 3) was investigated for a drug that is sensitive only to specific cell lines. All
the GDSC data with known IC50 values were observed in the same way as Figure 2a,
and dasatinib was selected as having the widest IC50 distributions. The wide distribution
of its IC50 means that the drug exhibits the greatest differences in efficacy based on the
type of cell line. We compared the self-attention score of the transformer on MEG-01, the
cell line judged to be sensitive by having the smallest IC50 value, and BT-483, the most
insensitive cell line with the largest IC50 value, among the 548 cell line pairs with dasatinib
(Figure 4). The MEG-01 cell line was derived from the hematopoietic and lymphoid tissue
of a leukemia patient, and the BT-483 cell line was derived from the breast tissue of a
breast-cancer patient.


Drug-aware updated
pathway embeddings









Cell line-aware updated
drug embedding


Self-attention score matrix

between pathways and a drug


Drug


Pathways



(Binary Data type Token)


0 1



Pathways



Drug





Raw drug
embedding



Raw pathway

embeddings


**Figure 3.** A detailed structure of type-aware transformer encoder reflecting interactions and relationships between pathways and a drug. We extracted drug-pathway interaction information from the
modified encoder of the Transformer module and identified putative key pathways for the drug’s
mechanism of action using a matrix of self-attention scores between pathways and the drug.


The TGF- _β_ signaling pathway (hsa04350) was the pathway with the highest attention
score for the MEG-01 cell line, which is the most sensitive to dasatinib. The secondmost-important pathway, ubiquitin-mediated proteolysis (hsa04120), involves the covalent
binding of ubiquitin to the target protein and its degradation. It is known that ubiquitinmediated degradation can regulate the TGF- _β_ signaling pathway [ 57 ]. The TGF- _β_ signaling
pathway suppresses tumors in normal and premalignant cells, yet promotes oncogenesis in
advanced cancer cells, and its components are regulated by ubiquitin-modifying enzymes;
abnormalities of the enzymes can cause malfunctioning of the pathway, which can cause
cancer, tissue fibrosis, and metastasis [ 58 – 60 ]. In this regard, the ubiquitin-modifying
enzymes in the pathway and their counterparts are increasingly being explored as potential drug targets [ 59 ]. Dasatinib is a tyrosine kinase inhibitor that can be a treatment for
chronic myeloid leukemia [ 61 ]. Dasatinib functions by binding to the ATP site of the active
conformation of BCR-Abl [ 62 ]. As a signal-transduction inhibitor, dasatinib inhibits the


_Int. J. Mol. Sci._ **2022**, _23_, 13919 10 of 19


proliferation of tumor cells by inhibiting tyrosine kinase action, especially blocking transcriptional and promigratory responses to TGF- _β_ through inhibition of Smad signaling [ 63 ].
The ubiquitin pathway can regulate the basal level of Smads, and altered Smad proteins
can cause a malfunction in responding to the incoming signals due to their importance
in transducing TGF- _β_ signals [ 57 ]. From the ubiquitin to the TGF- _β_ pathway, our model
captures the mechanism of action of the drug.


**Figure 4.** Visualization of all-pairwise self-attention scores from the transformer. ( **a** ) Dasatinib and
leukemia cell line MEG-01 pair. ( **b** ) Dasatinib and breast cancer cell line BT-483. The figures show the
y-axis as the query of the transformer and the x-axis as the key. On each axis, there is a drug and 34
pathways which start with “hsa”, indicating KEGG pathway identifiers.


Moreover, the ECM–receptor interaction pathway (hsa04512) was found to be most
important in the breast-cancer cell line, which is the most insensitive to dasatinib. The
ECM–receptor interaction pathway has been shown to be possibly useful as a biomarker
for breast cancer [ 64 ], but it does not relate to dasatinib’s mechanism of action. Hence, our
model identified the pathways related to the drug’s mechanism of action for drug-sensitive
carcinoma and focused on the biomarker for carcinoma without drug efficacy.


**3. Materials and Methods**

_3.1. Graph Neural Networks_


A graph neural network (GNN) is a type of neural network that operates on graphstructured data. GNN uses the topology of the graph to learn the relationships between the
input features. It can perform more effectively than other representation learning methods
on input data with topological information. In this study, we represent a graph as _G_ = ( _V_,
_E_ ) where _V_ = _{_ _v_ 1, . . ., _v_ _n_ _}_ is the set of _n_ nodes and _E_ _⊆_ _V_ _×_ _V_ is the set of edges. The node
_v_ _i_ has node feature _x_ _i_ _∈_ R _[d]_, where _d_ is a dimension of the feature. The node feature matrix
of the graph can be represented as _X_ _∈_ R _[n]_ _[×]_ _[d]_, where _n_ is the number of nodes in the graph.
Adjacency matrix _A_ _∈_ R _[n]_ _[×]_ _[n]_ indicates the total connectivity of nodes in the graph, where
_A_ _i_, _j_ = 1 means nodes, _v_ _i_ and _v_ _j_ are linked, and _W_ [(] _[l]_ [)] represents the parameters of the _l_ -th
layer of the graph (Table 5).


_Int. J. Mol. Sci._ **2022**, _23_, 13919 11 of 19


**Table 5.** Notation of graph neural networks used in this paper.


**Notation** **Description**


_G_ A graph.
_V_ Set of nodes of a graph.
_v_ A node included in V.
_i_, _j_ Indexes of the nodes.
_l_ Index of the layer of a graph.
_v_ _i_ _i_ -th node in V.
_x_ _i_ Node feature of node _v_ _i_
_N_ ( _i_ ) Set of neighbor nodes of a node _v_ _i_
_E_ Set of edges of a graph.
_A_ Adjacency matrix between nodes.
_W_ [(] _[l]_ [)] Trainable parameter matrix of _l_ -th layer.
_X_ [(] _[l]_ [)] Node feature matrix of _l_ -th layer.
_σ_ Nonlinear activation function softmax.
_ϵ_ Learnable parameter.


In each GNN layer, a key mechanism, called message passing, updates the node
representation by using the node features of the previous layer and the topology of the
graph [ 65 ]. The message passing mechanism involves aggregating the information of
neighboring nodes and updating the hidden state of each node by combining the node
representation from the previous layer and the aggregated messages. For every node in
each layer, a transformed feature vector is generated capturing the structural information
of the k-hop neighbor nodes. The GNN can update the _i_ -th node representation in the
_l_ -th layer as in the following Equation [ 66, 67 ], where _N_ ( _i_ ) is the set of neighbor nodes
linked to the target node _i_ . For a given node, the AGGREGATE step applies a permutation
invariant function to its neighboring nodes to produce the aggregated node feature of
neighbors, and the COMBINE step delivers the aggregated node feature to the learnable
layer to produce updated node embedding by integrating the existing embedding and the
aggregated neighbor embedding.


_x_ _i_ [(] _[l]_ [)] = _COMBINE_ [(] _[l]_ [)] _x_ _i_ [(] _[l]_ _[−]_ [1] [)], _AGGREGATE_ [(] _[l]_ _[−]_ [1] [)] [�] _x_ [(] _j_ _[l]_ _[−]_ [1] [)] : _j_ _∈_ _N_ ( _i_ ) � [�] (1)
�


_3.2. Cell-Line Graph Representation_
3.2.1. Cell-Line Graph Construction


We used a biological template network to represent cell lines to simulate gene–gene
interactions in actual cells. In the cell-line graph **G** _c_, genes are represented by nodes and
edges represent the relationships between genes. This template graph contained 2369 genes
selected using the pathway selection method described in the next section.
It is known that drugs do not have a universal effect throughout all cellular components, but tend to have distinct effects on specific genes or pathway targets. In this
way, cancer cells undergo phenotypic changes as a result of drug molecules inhibiting
or activating their target pathways. Motivated by this point, instead of representing the
cell line as a homogeneous large-scale graph that contains the entire genes, we divided
the template network **G** _c_ into pathway subgraphs **G** _p_ according to the biological domain
knowledge inspired by [ 68 ] and learned graph embeddings from the selected subgraph
units. Finally, the divided cell-line graph **G** _′_ _c_ [was represented as a heterogeneous graph]
containing multiple subgraphs. We selected pathways that can be targeted by drugs, as
they are associated with cancer from the KEGG pathway database [ 69 ], and used these
pathways as pre-defined subgraphs of the cell-line template.
In the case of template graph **G** _c_, the _i_ -th selected pathway subgraph can be described

as **G** [(] _p_ _[i]_ [)] = ( **V** [(] _p_ _[i]_ [)] [,] **[ E]** [(] _p_ _[i]_ [)] [), where] **[ V]** [(] _p_ _[i]_ [)] refers to a set of nodes and **E** [(] _p_ _[i]_ [)] refers to a set of edges
of the pathway. Thus, the template graph **G** _c_ is extended as a union of disjoint graphs,


_Int. J. Mol. Sci._ **2022**, _23_, 13919 12 of 19


with overlaps between the pathways in the form of **G** _′_ _c_ [= {] **[G]** [(] _p_ [1] [)] [, ...,] **[ G]** [(] _p_ [34] [)] }. In the template
cell-line graph **G** _c_, gene sets included in 34 pathways were represented by 2369 nodes and
7954 edges. A divided template graph **G** _′_ _c_ [with pathways as subgraphs had 4646 nodes and]
12,004 edges after combining the data from all pathways. The types of constituting genes
remained the same, but the numbers of nodes and edges increased when the template
network was divided into subnetworks due to the overlap of functions.


3.2.2. Cell-Line Graph Encoder on Pathway Subgraphs


Transcriptomic features of nodes and biological network topology were captured
within each subgraph using a graph attention network (GAT) [ 70 ]. Using the self-attention
mechanism, GAT calculates a normalized attention score _α_ _ij_, indicating the importance of
the features of the neighbor nodes for a target node _i_, where _j_ _∈_ _N_ ( _i_ ) . A subsequent step in
the message passing process is for each node to reflect the importance of the neighboring
nodes’ information in accordance with the previously obtained attention scores. Details
and graphical overview of GAT can be found in the original paper [70].


_X_ [(] _[l]_ [)] = _σ_ Σ _j_ _∈_ _N_ ( _i_ ) _α_ _ij_ [(] _[l]_ _[−]_ [1] [)] _W_ [(] _[l]_ _[−]_ [1] [)] _X_ [(] _[l]_ _[−]_ [1] [)] (2)
� �


If template graph **G** _c_ is used as it is, edges connected to one gene include interactions
from multi pathways, which can be noise. Node representations were updated through
GAT on the cell-line graph constructed in the previous subsection. The cell-line graph
consists of pathway-based subgraphs; thus, the updated node representation can reflect
the intra-pathway gene–gene interaction information. To pool the cell-line graph-level
embedding, we initially used simple hierarchical permutation-invariant graph-pooling
strategies [ 71 – 73 ]. However, the graph-pooling strategies we employed resulted in slight
performance degradation. We assumed that this was due to the relatively large size of the
cell-line graph, and simply pooling the nodes into a vector of the same dimension may lose
the information of the nodes in a cell line. As a result, the embeddings of each node learned
through GAT were concatenated to form a graph-level embedding for each pathway.


_3.3. Drug Graph Representation_
3.3.1. Drug Graph Construction


We used a graph neural network to learn the drug representation by reflecting the
relationships between atoms connected by bonds and the overall molecular structural
information. A drug can be represented as a graph in which atoms are nodes and bonds are
edges. We used RDKit [ 74 ] to transform SMILES [ 29 ], a one-dimensional string format drug
structure, into a graph format that can reflect structural information of an actual drug. The
ten initial features of atomic nodes were imported from previous research [ 30, 33 ], which
predicted drug sensitivity from GNN-based embeddings of drug structures. The details of
atomic and bond features can be found in Table A2.


3.3.2. Drug Graph Encoder


We used the graph isomorphism network (GIN) [ 75 ] to learn the features of the atomic
nodes within the drug graph. GIN applies a neighborhood aggregation method similar
to the Weisfeiler–Lehman test [ 76 ] and updates the _i_ -th node feature of the _l_ -th layer as
follows.



_x_ _i_ [(] _[l]_ [)] = _MLP_ [(] _[l]_ [)] 1 + _ϵ_ [(] _[l]_ [)] [�] _·_ _x_ _i_ [(] _[l]_ _[−]_ [1] [)] + Σ _j_ _∈_ _N_ ( _i_ ) _x_ [(] _j_ _[l]_ _[−]_ [1] [)]
��



(3)
�



Details and graphical overview of GIN can be found in the original paper [ 75 ]. The
graph encoder was chosen following the results of GraphDRP, which involved a comparison of different types of graph neural networks—GIN, GAT, and GCN+GAT—in order to
analyze the effectiveness of each graph encoder in predicting drug response. In addition,
GIN was widely used for the embeddings of drug graphs in various drug-response predic

_Int. J. Mol. Sci._ **2022**, _23_, 13919 13 of 19


tion models [ 31, 33, 34, 77 ]. All embeddings of each atom node were updated through GIN,
then they were concatenated to create raw drug embeddings before the pathway affected
them.


_3.4. Drug Response Prediction Module_
3.4.1. Knowledge-Guided Cell-Line–Drug Fusion Module Using Transformer


The transformer model tracks relationships in sequential data, such as the words
in a sentence, to discover context and meaning from the components [ 78 ]. We used
a transformer-based module to reflect not only inter-pathway interactions but also the
interactions between the pathways and each drug, which would allow exploring the
pharmacological mechanisms of action at the pathway level during a therapeutic process
(Figure 3).
The embedding of 34 pathways and that of a drug obtained from graph representation
modules were updated in the transformer-based cell line and drug fusion module. The
encoder is based on the transformer encoder, and does not use positional encoding, as
the order of the embeddings does not matter. Instead, we conducted two experiments:
first with binary-data-type tokens indicating whether the embedding is a pathway or a
drug, and then without positional encoding or data-type tokens. All-pairwise self-attention
scores can be obtained between pathways and drug embeddings via multi-head attention.
As a result of the encoder-based transformer model, the raw pathway embeddings were
updated to drug-aware pathway embeddings according to the effects of the drugs, and the
raw drug embedding was updated based on the relationship with each pathway.
Our structure has a single encoder-based layer taking pathway embeddings

_X_ [(] _p_ _[l]_ [)] ( _l_ = 1, ..., 34) and a drug embedding _X_ _d_ derived from knowledge-guided GNNs
as input values. Inputs in a typical transformer’s encoder are constructed by adding positional encoding to embeddings of source sequences. Unlike translation, where an order
of words in a sentence is important, the pathway embeddings entering our encoder are
not affected by the order in which they are encoded, so a transformer’s encoder structure
other than positional encoding was used for this study. As an alternative, we added a
type-encoded token that indicates whether the embedding is a drug or a pathway. In an
element-wise manner, type-encoded binary tokens are added to the input feature matrix
with the same number of dimensions before input embeddings are fed into the module.
On the fusion pathway and drug embeddings, self-attention was performed several
times through multi-head attention, and the average of each trial was used as the final attention score. After the encoder completed its execution, the encoder produced drug-aware

updated pathway embeddings _X_ [(] _p_ _[l]_ [)] [’ and pathways’ transcriptome-aware updated drug]
embedding _X_ _d_ ’ reflecting interaction information. These drug-aware pathway embeddings
facilitate interpreting the medication’s mechanism of action, since they can reflect both the
drug–pathway interaction information and the interactions between the pathways. Drugs
have a large structural variation when compared to cell-line graphs which are composed of
the same genes and are structurally equal but have different node feature values. Therefore, it is possible that the variation of the drug embedding may be blurred because the
new drug embedding updated as a result of the transformer is affected by the cell line
embedding. Hence, we connect the raw drug embedding obtained through GNN prior to
the transformer with the updated drug embedding obtained after the transformer using
residual connection [ 79 ]. By residual connection, it is possible to preserve the original
drug structure information and utilize the cell-line–drug interaction information using
the updated drug embedding which recognizes the transcriptomic information of each
pathway. We concatenated the resulting 34 subgraph embeddings in order to prevent
information loss, thereby embedding the entire cell line.


3.4.2. Improving Predictive Performance Using a Similarity Graph


Based on the idea that similar drugs and similar cell lines exhibit interchangeable drug
response behaviors, some drug-response prediction models use prior knowledge of drug


_Int. J. Mol. Sci._ **2022**, _23_, 13919 14 of 19


and cell-line similarity to minimize differences between drugs and cell lines in the latent
space. Reference [ 22 ] applied regularization terms based on chemical-structure similarities
between drugs and similarities between cell lines based on gene expression profiles to
improve prediction accuracy and prevent overfitting.
We followed the similarity-based embedding updating strategy of [ 33 ]. From the
completed end-to-end model up to Section 3.3, embeddings of all 580 cell lines and 170
drugs were created. Then, we constructed two homogeneous graphs, each consisting of
cell lines and drug nodes, with the initial feature of each node set having the resulting
embeddings of the previous step. Using GraphSAGE [ 80 ], we updated the embeddings of
each homogeneous cell line and drug graph. After that, we updated embeddings of each
cell-line–drug pair from two homogeneous graphs. We concatenated two embeddings into
the one-dimensional vector and used a multi-layer perceptron to predict final IC50 values.


**4. Conclusions**


In this paper, we proposed an interpretable drug-response prediction model called
DRPreter which integrates biological and chemical-domain knowledge with cutting-edge
deep learning technologies to deliver outstanding predictive performance and interpretability. We introduced cancer-related pathways and constructed the cell line network as a set of
subgraphs to represent and interpret biological mechanisms in detail. We extracted drug–
pathway interaction information from the modified encoder of the transformer module
and obtained putative key pathways for the drug’s mechanism. Ablation studies verified
the effectiveness of each component of the model, and performance comparison experiments showed DRPreter has enhanced predictive power compared to the state-of-the-art
graph-based drug-response prediction models which obtain either the cell line or drug
embedding using a homogeneous biochemical structure-based graph.
To properly apply the drug response predicted by the model for clinical use or drug
discovery, it is essential to understand the process and mechanism from which it was
derived due to safety and reliability issues. Accordingly, we implemented gene and
pathway-level analysis via DRPreter, and it has been shown that DRPreter predicts drug
sensitivity based on known drug mechanisms of action and target-related factors. We
also identified the cell line that would act most sensitively for each drug in the absence
of experimented data through a case study and confirmed that it is widely used for each
drug currently in the clinical situation. By doing so, patients who have shown resistance
to a specific drug may be able to select a drug candidate group that would replace the
unsuitable drug. It will be remarkably efficient to have comprehensive public databases
of drug targets and predictive models that can interpret pharmacological mechanisms for
personalized medicine and drug discovery.


**Supplementary Materials:** [The following supporting information can be downloaded at: https:](https://www.mdpi.com/article/10.3390/ijms232213919/s1)
[//www.mdpi.com/article/10.3390/ijms232213919/s1.](https://www.mdpi.com/article/10.3390/ijms232213919/s1)


**Author Contributions:** Conceptualization, J.S., Y.P., S.K., and K.J.; methodology, J.S. and Y.P.; software, J.S. and Y.P.; validation, J.S. and Y.P.; investigation, J.S., D.B. and K.J.; data curation, J.S.;
writing—original draft preparation, J.S.; writing—review and editing, J.S., Y.P., D.B., S.K., and K.J.;
visualization, J.S. and D.B.; supervision, S.K. and K.J.; funding acquisition, S.K. and K.J. All authors
have read and agreed to the published version of the manuscript.


**Funding:** This research was supported by the Bio and Medical Technology Development Program
of the National Research Foundation (NRF) funded by the Ministry of Science and ICT (NRF2022M3E5F3085677); by a grant (number DY0002259501) from the Ministry of food and Drug Safety;
by an Institute of Information and communications Technology Planning and Evaluation (IITP) grant
funded by the Korea government (MSIT) (number 2021-0-01343, Artificial Intelligence Graduate
School Program (Seoul National University)); by a National Research Foundation of Korea (NRF)
grant funded by the Korea government (MSIT) (number NRF-2020R1G1A1003558); and by the
Collaborative Genome Program for Fostering New Post Genome Industry of the National Research
Foundation (NRF) funded by the Ministry of Science and ICT (MSIT) (NRF2014M3C9A3063541).


_Int. J. Mol. Sci._ **2022**, _23_, 13919 15 of 19


**Institutional Review Board Statement:** Not applicable.


**Informed Consent Statement:** Not applicable.


**Data Availability Statement:** The data and code used in this study are available at
[https://github.com/babaling/DRPreter, (accessed on 30 September 2022).](https://github.com/babaling/DRPreter)


**Acknowledgments:** We would like to show our gratitude to Sangseon Lee and Dohoon Lee (Bio and
Health Informatics Lab, Seoul National University) for providing insight and expertise that greatly
assisted the research.


**Conflicts of Interest:** The authors declare no conflict of interest.


**Abbreviations**


The following abbreviations are used in this manuscript:


mut Mutation of gene
exp Gene expression
cnv Copy number variation
CCLE Cancer Cell Line Encyclopedia
CNN Convolutional Neural Network

COSMIC Catalogue of Somatic Mutations in Cancer
GAT Graph Attention Network
GCN Graph Convolutional Network
GDSC Genomics of Drug Sensitivity in Cancer
GIN Graph Isomorphism Network
GNN Graph Neural Network
GradCAM Gradient-weighted Class Activation Mapping
IC50 half maximal inhibitory concentration
MLP Multi-Layer Perceptron
RSEM RNA-Seq by Expectation Maximization
TGF Transforming Growth Factor
TPM Transcripts Per Kilobase Million


**Appendix A**


The number of genes and edges in Table A1 is the result of filtering according to the
threshold of protein–protein interactions’ combined score.


**Table A1.** A list of cancer-related pathways used as subgraphs in the cell-line template graph.


**Pathway Name** **KEGG Identifier** **Number of Genes** **Number of Edges**


Ubiquitin mediated proteolysis hsa04120 142 534
TGF- _β_ signaling pathway hsa04350 94 228
Estrogen signaling pathway hsa04915 137 222
MAPK signaling pathway hsa04010 294 692
PPAR signaling pathway hsa03320 74 28
mTOR signaling pathway hsa04150 155 688
Regulation of actin cytoskeleton hsa04810 218 552
B cell receptor signaling pathway hsa04662 79 208
Cell adhesion molecules hsa04514 146 150
Chemokine signaling pathway hsa04062 190 514
Apoptosis hsa04210 136 424
Cytokine-cytokine receptor interaction hsa04060 293 588
Wnt signaling pathway hsa04310 167 384
p53 signaling pathway hsa04115 73 180


_Int. J. Mol. Sci._ **2022**, _23_, 13919 16 of 19


**Table A1.** _Cont._


**Pathway Name** **KEGG Identifier** **Number of Genes** **Number of Edges**


Ras signaling pathway hsa04014 232 600
Notch signaling pathway hsa04330 59 76
Calcium signaling pathway hsa04020 239 218
HIF-1 signaling pathway hsa04066 109 204
T cell receptor signaling pathway hsa04660 104 336
ErbB signaling pathway hsa04012 85 326
Cell cycle hsa04110 126 1076
Melanogenesis hsa04916 101 110
cAMP signaling pathway hsa04024 221 222
VEGF signaling pathway hsa04370 59 102
Hedgehog signaling pathway hsa04340 56 80
Adherens junction hsa04520 71 172
Basal transcription factors hsa03022 44 470
PI3K-Akt signaling pathway hsa04151 351 1030
JAK-STAT signaling pathway hsa04630 162 508
Hematopoietic cell lineage hsa04640 96 102
Toll-like receptor signaling pathway hsa04620 102 328
Homologous recombination hsa03440 41 140
ECM-receptor interaction hsa04512 88 120
NF- _κ_ B signaling pathway hsa04064 102 392


**Table A2.** Atomic and bond features of the drug graph.


**Feature** **Size** **Description**


[B, C, N, O, F, ...]
Atom type 43
(one-hot)



Node



Whether the atom is

Aromatic 1 in aromatic system
(binary)
Chirality 2 [R, S] (one-hot or null)

[0, 1, 2, 3, 4, 5, 6, 7, 8,
Degree 11
9, 10] (one-hot)

Formal charge 1 electric charge
(integer)

[ _sp_, _sp_ [2], _sp_ [3], _sp_ [3] _d_,
Hybridization 5 _sp_ [3] _d_ [2] ] (one-hot or
null)
Number of
5 [0, 1, 2, 3, 4] (one-hot)
Hydrogens

[0, 1, 2, 3, 4, 5, 6]
Implicit valence 7
(one-hot)

Number of radical
Radical electrons 1
electrons (integer)

Whether the atom is
Ring 1
in ring (binary)



Edge Bond type 4 [single, double, triple,
aromatic] (one-hot)


**References**


1. Kellogg, R.A.; Dunn, J.; Snyder, M.P. Personal omics for precision health. _Circ. Res._ **2018**, _122_ [, 1169–1171. [CrossRef] [PubMed]](http://doi.org/10.1161/CIRCRESAHA.117.310909)
2. Ahmed, Z. Practicing precision medicine with intelligently integrative clinical and multi-omics data analysis. _Hum. Genom._ **2020**,
_14_ [, 1–5. [CrossRef] [PubMed]](http://dx.doi.org/10.1186/s40246-020-00287-z)
3. Kalamara, A.; Tobalina, L.; Saez-Rodriguez, J. How to find the right drug for each patient? Advances and challenges in
pharmacogenomics. _Curr. Opin. Syst. Biol._ **2018**, _10_ [, 53–62. [CrossRef] [PubMed]](http://dx.doi.org/10.1016/j.coisb.2018.07.001)
4. Singh, D.B. The impact of pharmacogenomics in personalized medicine. _Curr. Appl. Pharm. Biotechnol._ **2019**, _171_, 369–394.


_Int. J. Mol. Sci._ **2022**, _23_, 13919 17 of 19


5. Cho, S.Y. Patient-derived xenografts as compatible models for precision oncology. _Lab. Anim. Res._ **2020**, _36_ [, 1–11. [CrossRef]](http://dx.doi.org/10.1186/s42826-020-00045-1)

[[PubMed]](http://www.ncbi.nlm.nih.gov/pubmed/32461927)
6. Singh, V.P.; Pratap, K.; Sinha, J.; Desiraju, K.; Bahal, D.; Kukreti, R. Critical evaluation of challenges and future use of animals in
experimentation for biomedical research. _Int. J. Immunopathol. Pharmacol._ **2016**, _29_ [, 551–561. [CrossRef]](http://dx.doi.org/10.1177/0394632016671728)
7. Rees, M.G.; Seashore-Ludlow, B.; Cheah, J.H.; Adams, D.J.; Price, E.V.; Gill, S.; Javaid, S.; Coletti, M.E.; Jones, V.L.; Bodycombe,
N.E.; et al. Correlating chemical sensitivity and basal gene expression reveals mechanism of action. _Nat. Chem. Biol._ **2016**,
_12_ [, 109–116. [CrossRef]](http://dx.doi.org/10.1038/nchembio.1986)
8. Seyhan, A.A. Lost in translation: The valley of death across preclinical and clinical divide–identification of problems and
overcoming obstacles. _Transl. Med. Commun._ **2019**, _4_ [, 1–19. [CrossRef]](http://dx.doi.org/10.1186/s41231-019-0050-7)
9. Kuenzi, B.M.; Park, J.; Fong, S.H.; Sanchez, K.S.; Lee, J.; Kreisberg, J.F.; Ma, J.; Ideker, T. Predicting drug response and synergy
using a deep learning model of human cancer cells. _Cancer Cell_ **2020**, _38_ [, 672–684. [CrossRef]](http://dx.doi.org/10.1016/j.ccell.2020.09.014)
10. Savage, N. Tapping into the drug discovery potential of AI. _Biopharma Deal_ **2021** [. [CrossRef]](http://dx.doi.org/10.1038/d43747-021-00045-7)
11. Shoemaker, R.H. The NCI60 human tumour cell line anticancer drug screen. _Nat. Rev. Cancer_ **2006**, _6_ [, 813–823. [CrossRef]](http://dx.doi.org/10.1038/nrc1951)
12. Barretina, J.; Caponigro, G.; Stransky, N.; Venkatesan, K.; Margolin, A.A.; Kim, S.; Wilson, C.J.; Lehár, J.; Kryukov, G.V.; Sonkin, D.;
et al. The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. _Nature_ **2012**, _483_, 603–607.

[[CrossRef]](http://dx.doi.org/10.1038/nature11003)
13. Yang, W.; Soares, J.; Greninger, P.; Edelman, E.J.; Lightfoot, H.; Forbes, S.; Bindal, N.; Beare, D.; Smith, J.A.; Thompson, I.R.; et al.
Genomics of Drug Sensitivity in Cancer (GDSC): A resource for therapeutic biomarker discovery in cancer cells. _Nucleic Acids Res._
**2012**, _41_ [, D955–D961. [CrossRef]](http://dx.doi.org/10.1093/nar/gks1111)
14. Basu, A.; Bodycombe, N.E.; Cheah, J.H.; Price, E.V.; Liu, K.; Schaefer, G.I.; Ebright, R.Y.; Stewart, M.L.; Ito, D.; Wang, S.; et al. An
interactive resource to identify cancer genetic and lineage dependencies targeted by small molecules. _Cell_ **2013**, _154_, 1151–1161.

[[CrossRef]](http://dx.doi.org/10.1016/j.cell.2013.08.003)
15. Seashore-Ludlow, B.; Rees, M.G.; Cheah, J.H.; Cokol, M.; Price, E.V.; Coletti, M.E.; Jones, V.; Bodycombe, N.E.; Soule, C.K.; Gould,
J.; et al. Harnessing Connectivity in a Large-Scale Small-Molecule Sensitivity DatasetHarnessing Connectivity in a Sensitivity
Dataset. _Cancer Discov._ **2015**, _5_ [, 1210–1223. [CrossRef]](http://dx.doi.org/10.1158/2159-8290.CD-15-0235)
16. Iorio, F.; Knijnenburg, T.A.; Vis, D.J.; Bignell, G.R.; Menden, M.P.; Schubert, M.; Aben, N.; Gonçalves, E.; Barthorpe, S.; Lightfoot,
H.; et al. A landscape of pharmacogenomic interactions in cancer. _Cell_ **2016**, _166_ [, 740–754. [CrossRef]](http://dx.doi.org/10.1016/j.cell.2016.06.017)
17. Güvenç Paltun, B.; Mamitsuka, H.; Kaski, S. Improving drug response prediction by integrating multiple data sources: Matrix
factorization, kernel and network-based approaches. _Brief. Bioinform._ **2021**, _22_ [, 346–359. [CrossRef]](http://dx.doi.org/10.1093/bib/bbz153)
18. Adam, G.; Rampášek, L.; Safikhani, Z.; Smirnov, P.; Haibe-Kains, B.; Goldenberg, A. Machine learning approaches to drug
response prediction: Challenges and recent progress. _NPJ Precis. Oncol._ **2020**, _4_ [, 1–10. [CrossRef]](http://dx.doi.org/10.1038/s41698-020-0122-1)
19. Firoozbakht, F.; Yousefi, B.; Schwikowski, B. An overview of machine learning methods for monotherapy drug response
prediction. _Brief. Bioinform._ **2022**, _23_ [, bbab408. [CrossRef]](http://dx.doi.org/10.1093/bib/bbab408)
20. Riddick, G.; Song, H.; Ahn, S.; Walling, J.; Borges-Rivera, D.; Zhang, W.; Fine, H.A. Predicting in vitro drug sensitivity using
Random Forests. _Bioinformatics_ **2011**, _27_ [, 220–224. [CrossRef]](http://dx.doi.org/10.1093/bioinformatics/btq628)
21. Dong, Z.; Zhang, N.; Li, C.; Wang, H.; Fang, Y.; Wang, J.; Zheng, X. Anticancer drug sensitivity prediction in cell lines from
baseline gene expression through recursive feature selection. _BMC Cancer_ **2015**, _15_ [, 1–12. [CrossRef] [PubMed]](http://dx.doi.org/10.1186/s12885-015-1492-6)
22. Wang, L.; Li, X.; Zhang, L.; Gao, Q. Improved anticancer drug response prediction in cell lines using matrix factorization with
similarity regularization. _BMC Cancer_ **2017**, _17_ [, 1–12. [CrossRef] [PubMed]](http://dx.doi.org/10.1186/s12885-017-3500-5)
23. Guan, N.N.; Zhao, Y.; Wang, C.C.; Li, J.Q.; Chen, X.; Piao, X. Anticancer drug response prediction in cell lines using weighted
graph regularized matrix factorization. _Mol. Ther. Nucleic Acids_ **2019**, _17_ [, 164–174. [CrossRef] [PubMed]](http://dx.doi.org/10.1016/j.omtn.2019.05.017)
24. Baptista, D.; Ferreira, P.G.; Rocha, M. Deep learning for drug response prediction in cancer. _Brief. Bioinform._ **2021**, _22_, 360–379.

[[CrossRef] [PubMed]](http://dx.doi.org/10.1093/bib/bbz171)
25. Sakellaropoulos, T.; Vougas, K.; Narang, S.; Koinis, F.; Kotsinas, A.; Polyzos, A.; Moss, T.J.; Piha-Paul, S.; Zhou, H.; Kardala, E.;
et al. A deep learning framework for predicting response to therapy in cancer. _Cell Rep._ **2019**, _29_ [, 3367–3373. [CrossRef]](http://dx.doi.org/10.1016/j.celrep.2019.11.017)
26. Chiu, Y.C.; Chen, H.I.H.; Zhang, T.; Zhang, S.; Gorthi, A.; Wang, L.J.; Huang, Y.; Chen, Y. Predicting drug response of tumors
from integrated genomic profiles by deep neural networks. _BMC Med. Genom._ **2019**, _12_, 143–155.
27. Sharifi-Noghabi, H.; Zolotareva, O.; Collins, C.C.; Ester, M. MOLI: Multi-omics late integration with deep neural networks for
drug response prediction. _Bioinformatics_ **2019**, _35_ [, i501–i509. [CrossRef]](http://dx.doi.org/10.1093/bioinformatics/btz318)
28. Liu, P.; Li, H.; Li, S.; Leung, K.S. Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional
network. _BMC Bioinform._ **2019**, _20_ [, 1–14. [CrossRef]](http://dx.doi.org/10.1186/s12859-019-2910-6)
29. Weininger, D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. _J._
_Chem. Inf. Comput. Sci._ **1988**, _28_ [, 31–36. [CrossRef]](http://dx.doi.org/10.1021/ci00057a005)
30. Liu, Q.; Hu, Z.; Jiang, R.; Zhou, M. DeepCDR: A hybrid graph convolutional network for predicting cancer drug response.
_Bioinformatics_ **2020**, _36_ [, i911–i918. [CrossRef]](http://dx.doi.org/10.1093/bioinformatics/btaa822)
31. Nguyen, T.; Nguyen, G.T.; Nguyen, T.; Le, D.H. Graph convolutional networks for drug response prediction. _IEEE/ACM Trans._
_Comput. Biol. Bioinform._ **2021**, _19_ [, 146–154. [CrossRef]](http://dx.doi.org/10.1109/TCBB.2021.3060430)
32. Kim, S.; Bae, S.; Piao, Y.; Jo, K. Graph convolutional network for drug response prediction using gene expression data. _Mathematics_
**2021**, _9_ [, 772. [CrossRef]](http://dx.doi.org/10.3390/math9070772)


_Int. J. Mol. Sci._ **2022**, _23_, 13919 18 of 19


33. Zhu, Y.; Ouyang, Z.; Chen, W.; Feng, R.; Chen, D.Z.; Cao, J.; Wu, J. TGSA: Protein–protein association-based twin graph neural
networks for drug response prediction with similarity augmentation. _Bioinformatics_ **2022**, _38_ [, 461–468. [CrossRef]](http://dx.doi.org/10.1093/bioinformatics/btab650)
34. Feng, R.; Xie, Y.; Lai, M.; Chen, D.Z.; Cao, J.; Wu, J. AGMI: Attention-Guided Multi-omics Integration for Drug Response
Prediction with Graph Neural Networks. In Proceedings of the 2021 IEEE International Conference on Bioinformatics and
Biomedicine (BIBM), Houston, TX, USA, 9–12 December 2021; pp. 1295–1298.
35. Peng, W.; Liu, H.; Dai, W.; Yu, N.; Wang, J. Predicting cancer drug response using parallel heterogeneous graph convolutional
networks with neighborhood interactions. _Bioinformatics_ **2022**, _38_ [, 4546–4553. [CrossRef]](http://dx.doi.org/10.1093/bioinformatics/btac574)
36. Zuo, Z.; Wang, P.; Chen, X.; Tian, L.; Ge, H.; Qian, D. SWnet: A deep learning model for drug response prediction from cancer
genomic signatures and compound chemical structures. _BMC Bioinform._ **2021**, _22_ [, 1–16. [CrossRef]](http://dx.doi.org/10.1186/s12859-021-04352-9)
37. Deng, L.; Cai, Y.; Zhang, W.; Yang, W.; Gao, B.; Liu, H. Pathway-guided deep neural network toward interpretable and predictive
modeling of drug sensitivity. _J. Chem. Inf. Model._ **2020**, _60_ [, 4497–4505. [CrossRef]](http://dx.doi.org/10.1021/acs.jcim.0c00331)
38. DepMap Broad. [DepMap 21Q4 Public. Figshare. Dataset, 2021. Available online: https://portals.broadinstitute.org/ccle](https://portals.broadinstitute.org/ccle)
(accessed on 3 December 2021).
39. Szklarczyk, D.; Gable, A.L.; Lyon, D.; Junge, A.; Wyder, S.; Huerta-Cepas, J.; Simonovic, M.; Doncheva, N.T.; Morris, J.H.;
Bork, P.; et al. STRING v11: Protein–Protein association networks with increased coverage, supporting functional discovery in
genome-wide experimental datasets. _Nucleic Acids Res._ **2019**, _47_ [, D607–D613. [CrossRef]](http://dx.doi.org/10.1093/nar/gky1131)
40. Wang, Y.; Xiao, J.; Suzek, T.O.; Zhang, J.; Wang, J.; Bryant, S.H. PubChem: A public information system for analyzing bioactivities
of small molecules. _Nucleic Acids Res._ **2009**, _37_ [, W623–W633. [CrossRef]](http://dx.doi.org/10.1093/nar/gkp456)
41. Sondka, Z.; Bamford, S.; Cole, C.G.; Ward, S.A.; Dunham, I.; Forbes, S.A. The COSMIC Cancer Gene Census: Describing genetic
dysfunction across all human cancers. _Nat. Rev. Cancer_ **2018**, _18_ [, 696–705. [CrossRef]](http://dx.doi.org/10.1038/s41568-018-0060-1)
42. Field-Smith, A.; Morgan, G.J.; Davies, F.E. Bortezomib (Velcade ™ ) in the treatment of multiple myeloma. _Ther. Clin. Risk Manag._
**2006**, _2_ [, 271. [CrossRef]](http://dx.doi.org/10.2147/tcrm.2006.2.3.271)
43. Kouroukis, T.; Baldassarre, F.; Haynes, A.; Imrie, K.; Reece, D.; Cheung, M. Bortezomib in multiple myeloma: Systematic review
and clinical considerations. _Curr. Oncol._ **2014**, _21_ [, 573–603. [CrossRef] [PubMed]](http://dx.doi.org/10.3747/co.21.1798)
44. Récher, C.; Beyne-Rauzy, O.; Demur, C.; Chicanne, G.; Dos Santos, C.; Mas, V.M.D.; Benzaquen, D.; Laurent, G.; Huguet, F.;
Payrastre, B. Antileukemic activity of rapamycin in acute myeloid leukemia. _Blood_ **2005**, _105_ [, 2527–2534. [CrossRef] [PubMed]](http://dx.doi.org/10.1182/blood-2004-06-2494)
45. Selvaraju, R.R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; Batra, D. Grad-cam: Visual explanations from deep networks
via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision, Venice, Italy, 22–29
October 2017; pp. 618–626.
46. Wishart, D.S.; Knox, C.; Guo, A.C.; Cheng, D.; Shrivastava, S.; Tzur, D.; Gautam, B.; Hassanali, M. DrugBank: A knowledgebase
for drugs, drug actions and drug targets. _Nucleic Acids Res._ **2008**, _36_ [, D901–D906. [CrossRef] [PubMed]](http://dx.doi.org/10.1093/nar/gkm958)
47. Safran, M.; Dalah, I.; Alexander, J.; Rosen, N.; Iny Stein, T.; Shmoish, M.; Nativ, N.; Bahir, I.; Doniger, T.; Krug, H.; et al. GeneCards
Version 3: The human gene integrator. _Database_ **2010**, _2010_ [, baq020. [CrossRef] [PubMed]](http://dx.doi.org/10.1093/database/baq020)
48. Rouillard, A.D.; Gundersen, G.W.; Fernandez, N.F.; Wang, Z.; Monteiro, C.D.; McDermott, M.G.; Ma’ayan, A. The harmonizome:
A collection of processed datasets gathered to serve and mine knowledge about genes and proteins. _Database_ **2016**, _2016_, baw100.

[[CrossRef]](http://dx.doi.org/10.1093/database/baw100)
49. Ioannou, N.; Dalgleish, A.; Seddon, A.; Mackintosh, D.; Guertler, U.; Solca, F.; Modjtahedi, H. Anti-tumour activity of afatinib, an
irreversible ErbB family blocker, in human pancreatic tumour cells. _Br. J. Cancer_ **2011**, _105_ [, 1554–1562. [CrossRef]](http://dx.doi.org/10.1038/bjc.2011.396)
50. Fernandes, M.T.; Dejardin, E.; dos Santos, N.R. Context-dependent roles for lymphotoxin- _β_ receptor signaling in cancer
development. _Biochim. Biophys. Acta BBA Rev. Cancer_ **2016**, _1865_ [, 204–219. [CrossRef]](http://dx.doi.org/10.1016/j.bbcan.2016.02.005)
51. Canale, M.; Andrikou, K.; Priano, I.; Cravero, P.; Pasini, L.; Urbini, M.; Delmonte, A.; Crinò, L.; Bronte, G.; Ulivi, P. The Role of
TP53 Mutations in EGFR-Mutated Non-Small-Cell Lung Cancer: Clinical Significance and Implications for Therapy. _Cancers_ **2022**,
_14_ [, 1143. [CrossRef]](http://dx.doi.org/10.3390/cancers14051143)
52. Tishler, R.B.; Lamppu, D.M.; Park, S.; Price, B.D. Microtubule-active drugs taxol, vinblastine, and nocodazole increase the levels
of transcriptionally active p53. _Cancer Res._ **1995**, _55_, 6021–6025.
53. Tanaka, M.; Shibahara, J.; Fukushima, N.; Shinozaki, A.; Umeda, M.; Ishikawa, S.; Kokudo, N.; Fukayama, M. Claudin-18 is an
early-stage marker of pancreatic carcinogenesis. _J. Histochem. Cytochem._ **2011**, _59_ [, 942–952. [CrossRef]](http://dx.doi.org/10.1369/0022155411420569)
54. Stewart, D.J. Wnt signaling pathway in non–small cell lung cancer. _J. Natl. Cancer Inst._ **2014**, _106_ [, djt356. [CrossRef]](http://dx.doi.org/10.1093/jnci/djt356)
55. Hah, Y.S.; Cho, H.Y.; Lim, T.Y.; Park, D.H.; Kim, H.M.; Yoon, J.; Kim, J.G.; Kim, C.Y.; Yoon, T.J. Induction of melanogenesis by
rapamycin in human MNT-1 melanoma cells. _Ann. Dermatol._ **2012**, _24_ [, 151–157. [CrossRef]](http://dx.doi.org/10.5021/ad.2012.24.2.151)
56. Mehnert, J.M.; Tan, A.R.; Moss, R.; Poplin, E.; Stein, M.N.; Sovak, M.; Levinson, K.; Lin, H.; Kane, M.; Gounder, M.; et al.
Rationally Designed Treatment for Solid Tumors with MAPK Pathway Activation: A Phase I Study of Paclitaxel and Bortezomib
Using an Adaptive Dose-Finding ApproachPaclitaxel and Bortezomib for Tumors with MAPK Activation. _Mol. Cancer Ther._ **2011**,
_10_ [, 1509–1519. [CrossRef]](http://dx.doi.org/10.1158/1535-7163.MCT-10-0944)
57. Izzi, L.; Attisano, L. Regulation of the TGF _β_ signalling pathway by ubiquitin-mediated degradation. _Oncogene_ **2004**, _23_, 2071–2078.

[[CrossRef]](http://dx.doi.org/10.1038/sj.onc.1207412)
58. Huang, F.; Chen, Y.G. Regulation of TGF- _β_ receptor activity. _Cell Biosci._ **2012**, _2_ [, 1–10. [CrossRef]](http://dx.doi.org/10.1186/2045-3701-2-9)
59. Iyengar, P.V. Regulation of Ubiquitin Enzymes in the TGF- _β_ Pathway. _Int. J. Mol. Sci._ **2017**, _18_ [, 877. [CrossRef]](http://dx.doi.org/10.3390/ijms18040877)


_Int. J. Mol. Sci._ **2022**, _23_, 13919 19 of 19


60. Seoane, J.; Gomis, R.R. TGF- _β_ family signaling in tumor suppression and cancer progression. _Cold Spring Harb. Perspect. Biol._
**2017**, _9_ [, a022277. [CrossRef]](http://dx.doi.org/10.1101/cshperspect.a022277)
61. Keskin, D.; Sadri, S.; Eskazan, A.E. Dasatinib for the treatment of chronic myeloid leukemia: Patient selection and special
considerations. _Drug Des. Dev. Ther._ **2016**, _10_ [, 3355. [CrossRef]](http://dx.doi.org/10.2147/DDDT.S85050)
62. Sun, H.; Kapuria, V.; Peterson, L.F.; Fang, D.; Bornmann, W.G.; Bartholomeusz, G.; Talpaz, M.; Donato, N.J. Bcr-Abl ubiquitination
and Usp9x inhibition block kinase signaling and promote CML cell apoptosis. _Blood J. Am. Soc. Hematol.y_ **2011**, _117_, 3151–3162.

[[CrossRef]](http://dx.doi.org/10.1182/blood-2010-03-276477)
63. Bartscht, T.; Rosien, B.; Rades, D.; Kaufmann, R.; Biersack, H.; Lehnert, H.; Gieseler, F.; Ungefroren, H. Dasatinib blocks
transcriptional and promigratory responses to transforming growth factor-beta in pancreatic adenocarcinoma cells through
inhibition of Smad signalling: Implications for in vivo mode of action. _Mol. Cancer_ **2015**, _14_ [, 1–12. [CrossRef]](http://dx.doi.org/10.1186/s12943-015-0468-0)
64. Bao, Y.; Wang, L.; Shi, L.; Yun, F.; Liu, X.; Chen, Y.; Chen, C.; Ren, Y.; Jia, Y. Transcriptome profiling revealed multiple genes and
ECM-receptor interaction pathways that may be associated with breast cancer. _Cell. Mol. Biol. Lett._ **2019**, _24_ [, 1–20. [CrossRef]](http://dx.doi.org/10.1186/s11658-019-0162-0)

[[PubMed]](http://www.ncbi.nlm.nih.gov/pubmed/31182966)
65. Gilmer, J.; Schoenholz, S.S.; Riley, P.F.; Vinyals, O.; Dahl, G.E. Neural message passing for quantum chemistry. In Proceedings of
the International Conference on Machine Learning (PMLR), Sydney, Australia, 6–11 August 2017; pp. 1263–1272.
66. Li, M.M.; Huang, K.; Zitnik, M. Graph Representation Learning in Biomedicine. _arXiv_ **2021**, arXiv:2104.04883.
67. Dai, E.; Zhao, T.; Zhu, H.; Xu, J.; Guo, Z.; Liu, H.; Tang, J.; Wang, S. A Comprehensive Survey on Trustworthy Graph Neural
Networks: Privacy, Robustness, Fairness, and Explainability. _arXiv_ **2022**, arXiv:2204.08570.
68. Lee, S.; Lim, S.; Lee, T.; Sung, I.; Kim, S. Cancer subtype classification and modeling by pathway attention and propagation.
_Bioinformatics_ **2020**, _36_ [, 3818–3824. [CrossRef] [PubMed]](http://dx.doi.org/10.1093/bioinformatics/btaa203)
69. Kanehisa, M.; Goto, S. KEGG: Kyoto encyclopedia of genes and genomes. _Nucleic Acids Res._ **2000**, _28_ [, 27–30. [CrossRef]](http://dx.doi.org/10.1093/nar/28.1.27)
70. Veliˇckovi´c, P.; Cucurull, G.; Casanova, A.; Romero, A.; Lio, P.; Bengio, Y. Graph attention networks. _arXiv_ **2017**, arXiv:1710.10903.
71. Zhang, M.; Cui, Z.; Neumann, M.; Chen, Y. An end-to-end deep learning architecture for graph classification. In Proceedings of
the AAAI conference on Artificial Intelligence, New Orleans, LA, USA, 2 February 2018; Volume 32.
72. Gao, H.; Ji, S. Graph u-nets. In Proceedings of the International Conference On Machine Learning (PMLR), Long Beach, CA,
USA, 9–15 June 2019; pp. 2083–2092.
73. Lee, J.; Lee, I.; Kang, J. Self-attention graph pooling. In Proceedings of the International Conference on Machine Learning (PMLR),
Long Beach, CA, USA, 9–15 June 2019; pp. 3734–3743.
74. Landrum, G. RDKit: A Software Suite for Cheminformatics, Computational Chemistry, and Predictive Modeling. 2013. Available
[online: http://rdkit.sourceforge.net (accessed on 16 September 2021).](http://rdkit.sourceforge.net)
75. Xu, K.; Hu, W.; Leskovec, J.; Jegelka, S. How powerful are graph neural networks? _arXiv_ **2018**, arXiv:1810.00826.
76. Weisfeiler, B.; Leman, A. The reduction of a graph to canonical form and the algebra which appears therein. _NTI Ser._ **1968**,
_2_, 12–16.
77. Zheng, K.; Zhao, H.; Zhao, Q.; Wang, B.; Gao, X.; Wang, J. NASMDR: A framework for miRNA-drug resistance prediction using
efficient neural architecture search and graph isomorphism networks. _Brief. Bioinform._ **2022**, _20_ [, bbac338. [CrossRef]](http://dx.doi.org/10.1093/bib/bbac338)
78. Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A.N.; Kaiser, Ł.; Polosukhin, I. Attention is all you need.
_Adv. Neural Inf. Process. Syst._ **2017**, _30_, 6000–6010.
79. He, K.; Zhang, X.; Ren, S.; Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, Las Vegas, NV, USA, 26 June–1 July 2016; pp. 770–778.
80. Hamilton, W.; Ying, Z.; Leskovec, J. Inductive representation learning on large graphs. _Adv. Neural Inf. Process. Syst._ **2017**, _30_,
1025–1035.


