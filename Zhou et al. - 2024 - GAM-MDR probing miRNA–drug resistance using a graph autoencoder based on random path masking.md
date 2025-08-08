_Briefings in Functional Genomics_, 2024, **23**, 475–483


**https://doi.org/10.1093/bfgp/elae005**
Advance access publication date 22 February 2024

**Protocol Article**

# **GAM-MDR: probing miRNA–drug resistance using a** **graph autoencoder based on random path masking**


Zhecheng Zhou, Zhenya Du, Xin Jiang, Linlin Zhuo, Yixin Xu, Xiangzheng Fu, Mingzhe Liu and Quan Zou


Corresponding author: Linlin Zhuo, School of Data Science and Artificial Intelligence, Wenzhou University of Technology, Wenzhou, 325000, China.
[E-mail: zhuoninnin@163.com; Xiangzheng Fu, College of Computer Science and Electronic Engineering, Hunan University, Changsha, 410012, China.](mailto:zhuoninnin@163.com)
[E-mail: fxz326@hnu.edu.cn; Quan Zou, Institute of Fundamental and Frontier Sciences, University of Electronic Science and Technology of China, Chengdu,](mailto:fxz326@hnu.edu.cn)
[611730, China. E-mail: zouquan@nclab.net](mailto:zouquan@nclab.net)

- Zhecheng Zhou and Zhenya Du contributed equally.


Abstract


MicroRNAs (miRNAs) are found ubiquitously in biological cells and play a pivotal role in regulating the expression of numerous target
genes. Therapies centered around miRNAs are emerging as a promising strategy for disease treatment, aiming to intervene in disease
progression by modulating abnormal miRNA expressions. The accurate prediction of miRNA–drug resistance (MDR) is crucial for the
success of miRNA therapies. Computational models based on deep learning have demonstrated exceptional performance in predicting
potential MDRs. However, their effectiveness can be compromised by errors in the data acquisition process, leading to inaccurate
node representations. To address this challenge, we introduce the GAM-MDR model, which combines the graph autoencoder (GAE)
with random path masking techniques to precisely predict potential MDRs. The reliability and effectiveness of the GAM-MDR model
are mainly reflected in two aspects. Firstly, it efficiently extracts the representations of miRNA and drug nodes in the miRNA–drug
network. Secondly, our designed random path masking strategy efficiently reconstructs critical paths in the network, thereby reducing
the adverse impact of noisy data. To our knowledge, this is the first time that a random path masking strategy has been integrated
into a GAE to infer MDRs. Our method was subjected to multiple validations on public datasets and yielded promising results. We are
optimistic that our model could offer valuable insights for miRNA therapeutic strategies and deepen the understanding of the regulatory
mechanisms of miRNAs. Our data and code are publicly available at GitHub:https://github.com/ZZCrazy00/GAM-MDR.


_Keywords_ : miRNA–drug resistance (MDR); graph autoencoder (GAE); random path masking; accurate node representations



INTRODUCTION


MicroRNAs (miRNAs) represent a class of small, non-coding RNA
molecules that play a critical role in gene expression regulation

[1]. miRNA–drug resistance (MDR) often referred to as miRNA–
drug association in literature [2]. MiRNAs regulate complex biological processes through their impact on the expression of numerous genes. Consequently, their role in drug resistance could unveil
novel targets for disease treatment [3]. Abnormal miRNA expression can interact with various drugs, exacerbating diseases such
as cancer, cardiovascular, neurological and immune system disorders. Furthermore, miRNA alterations may predict individual drug
responses, offering crucial insights for personalized medicine [4].
The study of MDR is vital for comprehending drug mechanisms,
side effects, repositioning and advancing personalized medicine

[5]. Accurately predicting potential MDR is one crucial direction,
as it significantly improves miRNA-based therapeutic strategies
and related studies [6]. The discovery of more potential MDRs
can assist researchers in accurately and swiftly understanding



the mechanisms of miRNA resistance, thereby reducing the development cycle of miRNA therapeutic strategies. It is evident that
there is an urgent need for a prediction model for MDR that is
both efficient and accurate.


Traditional methods for studying MDR often require extensive consumption of manpower and resources. These methods
primarily include various biochemical experiments. For example,
potential resistance-associated miRNAs can be screened by comparing miRNAs in cell or tissue samples with verified resistant
miRNAs, using techniques such as gene chips [7], RNA sequencing

[8] and real-time quantitative PCR [9]. Additionally, Sarkar _et al._

[10] applied miRNA mimics and miRNA inhibitors in their experiments to observe miRNA drug sensitivity. MiRNAs can regulate
gene expression by binding to target genes. Predicting target
genes of differentially expressed miRNAs can help understand
the mechanisms of miRNA resistance. Tools such as TargetScan
and miRanda serve similar predictive functions [11, 12]. Gong _et al._

[13] verified the target genes of miRNAs through luciferase gene



**Zhecheng Zhou** pursued his studies at Wenzhou University of Science and Technology.
**Zhenya Du** was affiliated with Guangzhou Xinhua University.
**Xin Jiang** was affiliated with Wenzhou University of Science and Technology, focusing primarily on bioinformatics research.
**Linlin Zhuo** was affiliated with Wenzhou University of Science and Technology, focusing primarily on bioinformatics research.
**Yixin Xu** was affiliated with West China of Pharmacy Sichuan University.
**Xiangzheng Fu** was affiliated with Hunan University.
**Mingzhe Liu** was affiliated with Wenzhou University of Science and Technology, focusing primarily on bioinformatics research.
**Quan Zou** was affiliated with University of Electronic and Technology of China.
**Received:** November 7, 2023. **Revised:** January 15, 2024. **Accepted:** January 31, 2024
© The Author(s) 2024. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


476 | _Briefings in Functional Genomics_, 2024, Vol. 23, No. 4


experiments or protein expression level detection. These experimental verification methods can accurately predict MDR but are
time and resource-intensive, which has spurred the development
of computational approaches [14–16].

In recent years, computational methods predominantly based
on deep learning have been widely applied to MDR prediction

[17, 18]. Guo _et al._ [19] gathered SMILES and miRNA sequence data
of drugs, derived node representations and employed the random
forest approach to deduce unknown MDRs. Xu _et al._ [20] combined
various networks, including miRNA–drug and miRNA–disease, to
predict potential MDRs. Guan _et al._ [21] utilized a convolutional
neural network (CNN) to extract miRNA and drug features, along
with the topological feature of the miRNA–drug network. These
features were then integrated, and a Multilayer Perceptron (MLP)
was used for predicting unknown MDRs. Deepthi and Jereesh [22]
merged miRNA and drug similarity data, utilized CNN for node
representation extraction, and applied support vector machines
to identify unknown MDRs. Furthermore, Zheng _et al._ [23] refined
the MLMDA model, employed an autoencoder for miRNA and drug
node feature extraction and utilized an RF classifier to predict
potential MDRs [2]. Moreover, Zheng _et al._ [24] further modified
the iCDA-CGR model [24] and incorporated multi-path networks
for predicting unknown MDRs [2]. While these deep learning
methods efficiently predict unknown MDRs, their performance is
limited and lacks comprehensive consideration of miRNA–drug
topological network information.

The task of predicting MDR can be viewed as predicting missing
edges in a miRNA–drug bipartite graph, which is well suited
to Graph Neural Networks (GNNs). This has spurred the emergence of many GNN-based models. For instance, Huang _et al._
effectively aggregated and extracted features of miRNA and drug
nodes based on Graph Convolutional Neural Networks, converting
MDR prediction into link prediction on the miRNA–drug graph.
They integrated various types of information, such as miRNA
expression profiles and drug substructure fingerprints, to predict
potential MDR end-to-end [25]. Furthermore, Zheng _et al._ proposed an MDR prediction model based on Neural Architecture
Search (NAS) and GNN technology. Unlike common GNNs, this
model does not use miRNA–drug interaction graph information
but rather exploits node attribute information to predict MDR.
In addition, the model designs a novel sequence representation method, further enhancing the model’s predictive performance [2]. Inspired by contrastive learning, Wei _et al._ developed
a model based on Graph Collaborative Filtering and Multi-View
Contrastive Learning. This model is the first to apply multi-view
contrastive learning to MDR prediction, a strategy that alleviates
the impact of noisy data and sparse neighborhoods on model
performance [26].

The aforementioned methods have continually improved in
predicting molecular associations such as MDR, achieving commendable performance. However, several pressing issues need
to be addressed. First, both similarity network-based and GNNbased methods have not fully extracted information from the
miRNA nodes (or drug nodes). Second, they have not given enough
consideration to the handling of noisy data, which in turn limits
the performance of these methods. To alleviate these issues, we
develop an MDR prediction model based on graph autoencoder
(GAE) [27, 28] and random path masking strategy. The GAE combines the advantages of GNNs and autoencoders, integrating the
topology of the miRNA–drug graph and node sequence information to extract robust node representations. Moreover, we design
a path masking strategy grounded in random walk principles
to selectively obscure certain pathways within the miRNA–drug



graph, a maneuver followed by self-supervised training. This tactic significantly minimizes the perturbation caused by noisy data
within the graph, concurrently enhancing the representation of
nodes. The model then trains or predicts the miRNA–drug pairs
based on these high-quality node representations. In summary,
our contributions can be encapsulated as follows.


(i) We develop an MDR prediction model based on GAE and
random path masking, which achieved satisfactory results.
(ii) We advance a harmonized GAE-based framework, interweaving the topological information of the miRNA–drug
graph with the initial attribute data of miRNA nodes (or
drug nodes) to enrich node representation. Moreover, this
framework can utilize various GNNs as encoders, increasing
its scalability.
(iii) We design a path masking strategy grounded in random
walk principles and incorporated it into GAE. This strategy
proves effective in attenuating the influence of noise within
the miRNA–drug network, markedly elevating the model’s
performance.
(iv) We assemble multiple experiments utilizing publicly
available datasets, thereby substantiating the effectiveness,
robustness and scalability of our proposed model.


METHOD


We propose a novel MDR prediction model, known as GAM-MDR,
grounded in the principles of GAE [28] and random path masking
strategy, with the objective of mitigating the influence of noisy
data and addressing insufficient feature extraction within the
graph. The GAM-MDR model is a harmonized framework based
on GAE [28], primarily comprising a GNN-based encoder and a
MLP-based decoder. Its encoder can opt for classic GNNs such as
GraphSAGE [29], GCN [30], GIN [31], GAT [32] and GAT2 [33], among
others. Furthermore, we have architected a path masking strategy,
anchored in random walks, to enhance the model’s proficiency in
handling noisy data. The ensuing sections of this article will delve
into the introduce of the model’s constituent modules.


**Model architecture**


As shown in Figure 1, the architecture of the GAM-MDR model
mainly consists of three modules: (A) construction of the initial
miRNA–drug association graph, (B) masking part of the miRNA–
drug association graph based on random walks and (C) employing an autoencoder to reconstruct the miRNA–drug association
graph, extract miRNA node (or drug node) representations, and
carry out prediction. Initially, the miRNA–drug association graph
is constructed. The Graph Isomorphism Network (GIN) is used
to extract feature vectors from the drug molecular structure
graphs, and the k-mer algorithm is employed to extract feature
vectors from the miRNA sequences, serving as the initial node
representations in the graph. In Figure 1, different colors are used
to distinguish miRNA and drug nodes. This part will be described
in detail in the first part of the experiment section. Subsequently,
a path masking strategy based on random walks is designed to
mask part of the paths in the input miRNA–drug association
graph. Then, the masked graph data is encoded using a GNN, and
all the edges are decoded using an MLP. After multiple iterations
of training, the miRNA–drug graph is reconstructed, obtaining the
representation of each node in the graph. Finally, the Hadamard
product is used to calculate the score of the miRNA–drug pair to
predict whether there is an association.


_Zhou_ et al. | 477


Figure 1: The GAM-MDR model’s architecture is composed of three primary modules. ( **A** ) Data Collection and Feature Extraction: Drug chemical
structures and miRNA sequence data are sourced from the PubChem and ncDR databases, respectively. The GIN encoder is utilized for extracting drug
features, while the pse-one tool is employed to derive k-mer features from the miRNA sequences. ( **B** ) Network Construction and Masking: This step
involves building an MDR network using established MDR data, along with initial features of miRNAs and drugs. The process includes selecting initial
nodes for the random walk based on the Bernoulli distribution and applying a random walk strategy to obscure certain paths in the network. ( **C** )
Network Reconstruction and Completion: A GAE is used to reconstruct the MDR network and to fill in missing MDR data.



**Path-based masking strategy**


The novel path masking strategy based on random walks that
we designed in this study mainly includes two steps: firstly,



some nodes on the miRNA–drug graph are selected as starting
points based on a Bernoulli distribution. Secondly, paths are
extracted based on these starting nodes and are then masked


478 | _Briefings in Functional Genomics_, 2024, Vol. 23, No. 4


using a random walk method. In this approach, the masked
edges are still treated as positive samples during training.
However, their integration in the message passing process is
not considered. In each training iteration, edges along certain
paths are selectively masked again, employing the random walk
method. This approach ensures a more robust and effective
training process. The model can learn this masked information
through self-supervised training, which improves the learning
capability of the model. The detailed computation strategy for
masking is as follows:


_ε_ _mask_ = _RandomWalk(_ _**R**_, _l_ _walk_ _)_, (1)


_**R**_ ∼ _Bernoulli(p)_, (2)


where _l_ _walk_ represents the path length of the random walk and also
an adjustable parameter. _**R**_ represents the set of nodes sampled
from the graph according to the Bernoulli distribution, and _p_
represents the sampling rate between 0 and 1. The edges sampled
by the masking strategy based on random walks are masked, with
the starting points being the nodes in the sampled node set.


**GNN encoder**


GNNs can uncover the underlying structure of graph data. In this
study, two GNN encoders are employed. One is used to extract
representation information from the chemical structure of drugs,
and the other serves as an encoder in the GAE to model the

observed MDR. For drug molecule representation, GIN is chosen as
the encoder. Under the framework of GAE, several classical GNNs,
such as GraphSAGE, GCN, GIN, GAT and GAT2, can be selected.
GraphSAGE efficiently leverages node feature information but
may underperform in inductive settings. GCN is effective for semisupervised learning but struggles with large graphs. GIN excels in
representational power, often at increased computational costs.
GAT provides attention-based weights for node importance but
can be computationally intensive. GAT2 extends GAT with more
flexible attention mechanisms, potentially increasing complexity.
Taking GCN as an example, it applies convolution-like operations
to propagate, aggregate and update features of miRNA–drug graph
nodes. The computation for the (l+1)th layer is as follows:


_H_ _[(][l]_ [+][1] _[)]_ = _σ(AH_ [�] _[(][l][)]_ _W_ _[(][l][)]_ _)_, (3)


where _σ_ represents the Activation function, _H_ _[(][l][)]_ represents the
characteristic matrix of nodes in layer l and _W_ _[(][l][)]_ represents the
learnable weight matrix in layer l. Specifically, _H_ [0] = _X_ represents
the initial feature matrix of the input miRNA (or drug) node. _A_ � represents the normalized Adjacency matrix, calculated as
follows:



�
_A_ = � _D_ [−] 2 [1]




[1] ��

2 _AD_ [−] 2 [1]



2, (4)



**Edge decoder**


We design an MLP-based decoder to decode edges in miRNA–drug
graphs. Based on the node features obtained by the GNN encoder,
the model calculates the Hadamard product of the miRNA–drug
pair; then, it is input to the MLP, and the score of the miRNA–
drug pair is calculated using the sigmoid function. Calculated as
follows:


_h(z_ _m_, _z_ _d_ _)_ = _Sigmoid(MLP(z_ _m_ ◦ _z_ _d_ _))_ . (5)


Among them, _z_ _m_ represents the feature vector of miRNA, _z_ _d_
represents the feature vector of the drug, _h(z_ _m_, _z_ _d_ _)_ represents the
calculated score of miRNA–drug pair, which is in the range of

[0,1], indicating the strength of MDR. ◦ means Hadamard product
operation.


**Model training**


During training, the predicted score is compared with the true
label, and the BCE loss function is used to calculate the loss value


_Edge_ _loss_ = _(y_ − 1 _)_ ∗ _log(_ 1 − _p)_ − _y_ ∗ _log(p)_, (6)


where _y_ represents the true label of the miRNA–drug pair, and _p_
represents the score of the predicted miRNA–drug pair. During
prediction, we set a threshold of 0.5. If the miRNA–drug pair
score is greater than 0.5, then the pair is predicted to be MDR;
if the score is less than 0.5, then the pair is predicted to be

non-MDR.


RESULTS


This chapter comprises two main sections: Experimental Setup
and Experimental Results. The Experimental Setup outlines the
dataset and evaluation strategy employed in this study, while
the Experimental Results present the performance of various
comparative models, the outcomes of parameter experiments and
relevant experimental analyses.


**Experimental setup**
_Dataset_


To evaluate the performance of the model, we conduct multiple
sets of experiments on public datasets, and 3163 MDR were
obtained from the ncDR database (http://www.jianglab.cn/ncDR/),
involving 101 drugs and 701 miRNA molecules. In the experiments, 3163 experimentally validated MDR were used as positive
samples and randomly sampled from the remaining miRNA–drug
pairs as negative samples. Based on this, an initial miRNA–drug
association map can be constructed. The acquisition of initial
features of drugs and miRNAs is described next.


_Initial drug feature_


For drugs, the chemical structure data are obtained from the
PubChem database (https://pubchem.ncbi.nlm.nih.gov), and GIN
is used to extract representations as the initial input features of
drug nodes. GIN is a GNN based on an unsupervised learning
strategy. The chemical structure of a drug can be viewed as a
graph. GIN can be used to fully extract features based on the
atomic link relationship. The formula is as follows:



where _A_ represents the Adjacency matrix, _A_ [�] represents the Adjacency matrix with self ring and [�] _D_ represents the degree matrix
of _A_ [�] . In each layer, GCN encoder adds and averages the representations of neighbor nodes, performs linear transformation based
on weight matrix and then uses Activation function to perform
nonlinear transformation. Setting up a multi-layer GCN encoder
can aggregate node features at longer distances and capture
longer node dependencies. In addition to GCN, we also evaluated
the performance of various GNN encoders in the experiment, such
as GraphSAGE, GIN, GAT, GAT2, etc. Then, the characteristics of
miRNA and drug nodes can be obtained.



_h_ _v_ _[(][k][)]_ = _MLP_ _[(][k][)]_ _(_ 1 + _ϵ_ _[(][k][)]_ _)_ - _h_ _v_ _[(][k][)]_ [+] � _h_ _u_ _[(][k]_ [−][1] _[)]_
� _u_ ∈ _N(v)_



. (7)
�


Among them, _h_ _[(][k][)]_ represents the characteristics of node _v_ in the
kth layer, and _N(v)_ represents the set of neighbors of node _v_ .
_ϵ_ _[(][k][)]_ represents the learnable parameters of the kth layer, which
is used to adjust the weight of the atom’s own node features
and neighbor node features. _MLP_ _[(][k][)]_ represents the fully connected
neural network of the kth layer, which is used to aggregate the
characteristics of the atom itself and its neighbor nodes. After
multi-layer aggregation update, the final drug node is represented
as a 64-dimensional feature vector.


_Initial miRNA feature_


For miRNA, the nucleotide sequence of miRNA is obtained from
the ncDR database, and the k-mer algorithm is used to extract
and characterize it as the initial input feature of the miRNA
molecule. First, the original miRNA sequence is divided into a
set of subsequences containing k bases, k is set to 3; then the
frequency of each subsequence is calculated, and these frequencies are normalized; finally, based on the projection matrix,
it is mapped to a 64-dimensional eigenvectors of. Therefore,
each miRNA can be represented as a 64-dimensional feature

vector.


_Evaluation indicators_


In order to evaluate the performance of the model, two comprehensive indicators, AUC and AUPR, were used in the study. In
addition,we also selected several common indicators such as ACC,
PRE, SPE, SEN, F1 and MCC, which are calculated as follows:


_TP_ + _TN_
_ACC_ = (8)
_TP_ + _TN_ + _FP_ + _FN_ [,]


_TP_ _TN_
_PRE_ = _SPE_ = (9)
_TP_ + _FP_ [,] _TN_ + _FP_ [,]



_TP_
_SEN_ = _F_ 1 = [2][ ×] _[ PRE]_ [ ×] _[ SEN]_, (10)
_TP_ + _FN_ [,] _PRE_ + _SEN_



_Zhou_ et al. | 479


Figure 2: Comparison of the GAM-MDR model with other models.


randomly divided into 5 (10) parts, and one of them was used as
the test set in turn, and a 5 (10)-fold crossover experiment was
carried out.


The experimental results in Figure 2 demonstrate that the
performance of the GAM-MDR and NASMDR models is better
than that of the iCDA-CGR and MLMDA models. This shows that

the classification model of deep learning may be better than the
traditional machine learning classification model. In addition,
the NASMDR model also designs an excellent representation
method, which can more fully extract node features. This shows
that robust node representation is very important for association
prediction. The GAM-MDR model has achieved the best AUC
performance in the 5 (10)-fold crossover experiments, and the
variance is also smaller than other models. This may be due to the
fact that the GAE can fully exploit the characteristics of the nodes
themselves and the topological information of the miRNA–drug
graph to extract more robust node representations. In addition,
the model adopts a path masking strategy based on random
walks, which alleviates the influence of noisy data and further
improves the performance of the model.


_Comparison of different GNN encoders_


To evaluate the impact of different GNN encoders on model
performance, we select different GNNs as the encoders of the GAE
framework. Moreover, these models using different encoders are
compared with NASMDR, and the results are shown in Table 1.
While keeping the rest of the model parameters unchanged, we
set the number of encoder layers to 3 and the number of decoder
layers to 4. Additionally, we established the masking rate _p_ at
0.3. In addition, the _AUC_ and _AUPR_ curves with different encoder
models are also plotted, as shown in Figure 3.

From the overall analysis of the data in Table 1 and Figure 3,
when the model uses GCN, GIN, GraphSAGE, GAT and GAT2 as
encoders, it can achieve better performance. This shows that the
performance of the model is relatively stable and less affected
by different encoders. Compared with the current state-of-theart NASDMR model, the AUC indicators of the GAM-MDR model
with different encoders are all improved by more than 3%. This
shows that the GAM-MDR model based on the GAE framework

can effectively identify potential MDR. Additionally, We employed
fully connection (FC) layers for extracting node representations
and noted a marked deterioration in the model’s performance.
This observation underscores the pivotal role that topological
information of the MDR network plays in substantially enhancing
node representations.



_TP_ × _TN_ − _FP_ × _FN_
_MCC_ =, (11)
~~√~~ _(TP_ + _FP)(TP_ + _FN)(TN_ + _FN)(TN_ + _FN)_


where _TP_ represents the number of positive MDR predicted correctly; _TN_ represents the number of negative MDR predicted
correctly; _FP_ represents the number of positive MDR predicted
incorrectly; _FN_ represents the number of negative MDR predicted
incorrectly.


**Experimental results**
_Comparison with other models_


We selected the three models of MLMDA [23], iCDA-CGR [24]
and NASMDR [2] as the comparison models, using AUC as the
evaluation index, and compared the performance with the proposed GAM-MDR model on the MDR dataset. Among them, the
MLMDA model integrates a variety of similarity networks, uses
autoencoder to extract features and predicts MDR based on random forests. The iCDA-CGR model integrates miRNA (or drug)
node features and association network information and predicts
MDR based on the SVM algorithm. While these two methods
leverage deep learning technology to autonomously extract node
features, they overlook the structural intricacies of the MDR network. The NASMDR model proposes a new node representation
method based on NAS to predict potential MDR. In contrast, this
model incorporates the network’s topological data. However, it
still falls short in acknowledging the potential influence of data
noise on the model’s performance, an aspect crucial for accurate
predictions and analysis. In the experiment, the dataset was


480 | _Briefings in Functional Genomics_, 2024, Vol. 23, No. 4


**Table 1.** Comparison of GAM-MDR model with different encoders and NASDMR model


**Models** **AUC** **AUPR** **ACC** **PRE** **SEN** **F1**


NASMDR 94.68±0.78% 94.77±1.10% 88.32±1.19% 88.59±1.51% 87.98±1.13% 88.28±1.18%

ours-FC 94.79±0.21% 94.22±0.23% 88.60±0.19% 86.39±0.14% 90.39±0.27% 88.34±0.16%

ours-SAGE 98.18±0.07% 96.79±0.15% 94.62±0.33% 91.96±0.52% **97.78** ± **0.14%** 94.79±0.30%

ours-GCN **98.62** ± **0.04%** 97.95±0.18% **95.97** ± **0.22%** **94.76** ± **0.31%** 97.31±0.37% **96.02** ± **0.22%**

ours-GIN 98.60±1.52% **98.16** ± **0.90%** 94.94±3.20% 92.77±1.21% 97.47±7.95% 95.06±3.95%

ours-GAT 98.30±0.09% 96.97±0.59% 95.09±0.27% 94.39±0.50% 95.89±0.20% 95.13±0.24%

ours-GAT2 98.45±0.05% 96.93±0.20% 95.25±0.55% 93.73±1.50% 96.99±0.85% 95.33±0.47%


Figure 3: AUC and AUPR curves of the GAM-MDR model with different encoders.


_Parameter experiment_


In this section, we construct multiple sets of parametric experiments to evaluate the adaptability and stability of the model.
In the model, the following adjustable parameters are mainly
involved: the number of encoders and decoders, the mask ratio
and the length of the random walk. Next, we will analyze the
performance of the model from these aspects.


_Impact of number of encoder and decoder_



In this subsection, we explore the impact of the number of
encoders and decoders on model performance. In the proposed
model, the number of encoders and decoders can be set manually.
In the experiment, AUC was selected as the evaluation index,
keeping other conditions the same, the model uses encoders and
decoders with different layers and the results are shown in the
heat map in Figure 4.

It can be seen from the heat map that when the number of
layers of the encoder is set to 3 and the number of layers of
the decoder is set to 2, the model will obtain the highest AUC
performance; after that, as the number of encoders and decoders
increases, the model, the AUC performance tends to be stable.
When both encoder and decoder layers are set to 1, the model
achieves the lowest AUC performance. The experimental results
show that the number of encoders and decoders will affect the

performance of the model, in general, the larger the number, the
better the model performance. This shows that it is convenient
to stack encoders and decoders to improve the predictive performance of the model.


_Impact of mask ratio p_


In this subsection, we explore the impact of mask ratio on model
performance. We devise a path-based masking strategy to mask
part of the paths in the miRNA–drug graph. By setting different



Figure 4: The impact of the number of encoders and decoders on model
performance.


mask ratio _p_, the start point of the mask path is sampled based on
the Bernoulli distribution. In the experiment, keeping the random
walk length as 3 and other conditions unchanged, 20 rounds of
tests were carried out for each _P_ -value, and the experimental
results are shown in Figure 5.

The experimental results in Figure 5 show that the performance of the model using the random path masking strategy is
better than that of the model without the random path masking
strategy. The mask ratio _p_ has a more obvious impact on the performance of the model. When _p_ is in the range of [0.1, 0.3], as the
mask ratio increases, the performance of the model continues to
improve, and the best AUC performance is achieved when _p_ =0.3;
when _p >_ 0.4, as the mask ratio changes large, the performance of


Figure 5: The impact of different mask ratio _p_ on the performance of the
model, the fluctuation range of the performance is marked in the figure.
The blue curve represents the model performance with the random
path masking strategy, and the red curve represents the model
performance without the random path masking strategy.


the model gradually decreases. This proves that self-supervised
training based on a small mask ratio may effectively alleviate the
impact of noisy data; self-supervised training based on a large
mask ratio can alleviate the impact of noisy data to a certain
extent, but at the same time, it will lose useful information.
These results indicate that choosing a smaller mask ratio can
significantly improve the performance of the model.


_Impact of walk length_


In this subsection, we explore the impact of the path length
of the random walk on the performance of the model. In the
designed random path masking strategy, the masking rate _p_ is
set to 0.3. After the starting point of sampling is determined, the
masked path is determined based on the random walk method.
In the experiment, 2, 3 and 4 were selected as the path length
of the random walk, and then, the mask path was determined.
And 20 rounds of tests were carried out for each length, and the
corresponding box figure was drawn based on the experimental
results.


Based on the experimental results in Figure 6, the AUC, ACC,
SEN and F1 performance indicators of the model are generally
stable with little fluctuation; the PRE, SPE and MCC indicators of
the model are the lowest when the length is 4. In addition, when
setting the length to 3, all indicators achieved the best results.
This shows that the path length of the random walk does affect
the prediction performance of the model, and masking shorter
paths often improves the performance of the self-supervised
learning model. Conversely, shading longer paths may degrade
the performance of the model because more useful information
is lost. Therefore, paths with shorter shadows can be chosen to
improve the performance of the model.


_Impact of feature dimension_


We also investigated how the feature dimensions of miRNA and
drugs affect the model’s performance. To do this, we set the
feature dimensions for miRNA and drugs at three different levels:
64, 128 and 256. The outcomes of this investigation are detailed in
Table 2, providing insights into the relationship between feature
dimension sizes and model efficacy.

The observations suggest that varying the sizes of feature
dimensions has a negligible effect on the performance of our



_Zhou_ et al. | 481


Figure 6: The impact of random walk path length on model
performance. The blue, green and red box plots represent the
performance indicators of the model when the path lengths are 2, 3 and
4, respectively.


model, thereby demonstrating its robustness. Consequently, this
allows for greater flexibility and a broader range of options in
parameter settings, enhancing the usability and adaptability of
our model.


_Impact of hyperparameters_


Drawing from empirical principles, we set a threshold of 0.5
to distinguish between positive and negative samples. We also
examined the impact of 0.4 and 0.6 thresholds on model performance. In Table 3, upon comparison, the differences in results
were found to be insignificant. We have conducted corresponding
experiments to explore the impact of feature splicing technologies
on model performance. It is evident that the model’s performance
diminishes when employing feature concatenate technology. This
observation reaffirms the suitability of the Hadamard product
over feature splicing for more effectively calculating the interaction scores between miRNA and drugs.


_Case study_


To thoroughly investigate the model’s performance in real-world
scenarios, we undertake case studies. Our focus was on two drugs,
Verapamil and Doxorubicin (Adriamycin), and we aimed to predict
which miRNAs might potentially contribute to drug resistance
against them.

Verapamil, a calcium channel blocker drug predominantly utilized to combat cardiovascular ailments such as hypertension,
angina pectoris and specific arrhythmias, operates by blocking
calcium ion channels on the cellular membrane. This reduces the

influx of calcium ions into heart and vascular smooth muscle

cells, consequently attenuating blood vessel dilation and cardiac contractility, ultimately leading to decreased blood pressure and cardiac load. Analyzing, which miRNAs might engender
resistance to Verapamil, bears significant value, offering crucial
insights for future treatment strategies. Doxorubicin, also known
as Adriamycin, is a chemotherapy drug extensively employed in
cancer treatment. Emerging studies suggest that certain miRNAs
may serve as potential targets of Adriamycin, with the drug
potentially modulating the growth and metastasis of cancer cells
by influencing miRNA expression. However, these studies are in
their nascent stages, and unearthing additional potential miRNA
targets could potentially furnish valuable guidance for cancer
treatment approaches.


482 | _Briefings in Functional Genomics_, 2024, Vol. 23, No. 4


**Table 2.** Results of the GAM-MDR model using different feature dimensions


**Dimension** **AUC** **ACC** **PRE** **SEN** **SPE** **F1**


64 98.62% 95.97% 94.76% 97.31% 94.47% 96.02%

128 98.81% 95.73% 96.20% 95.29% 95.25% 95.74%

256 98.40% 95.33% 96.67% 94.14% 93.98% 95.39%


**Table 3.** Results of the GAM-MDR model using different hyperparameters


**Hyperparameters** **AUC** **ACC** **PRE** **SEN** **SPE** **F1**


**threshold**

0.4 98.81% 95.01% 97.78% 92.65% 92.24% 95.15%

0.5 98.62% 95.97% 94.76% 97.31% 94.47% 96.02%

0.6 98.13% 94.46% 96.67% 94.14% 93.98% 95.39%


**splicing strategies**
Hadamard product 98.62% 95.97% 94.76% 97.31% 94.47% 96.02%

concatenate 96.37% 92.87% 93.51% 92.34% 92.24% 92.92%



**Table 4.** Top 20 miRNAs with resistance scores to Verapamil


**miRNA** **ncDR** **miRNA** **ncDR**


hsa-let-7g definited hsa-mir-216b definited
hsa-let-7i definited hsa-mir-219a-1 definited

hsa-mir-100 definited hsa-mir-221 definited

hsa-mir-101-1 definited hsa-mir-25 definited

hsa-mir-10a definited hsa-mir-26a-1 definited

hsa-mir-10b definited hsa-mir-509-1 definited

hsa-mir-151a definited hsa-mir-514a-1 definited

hsa-mir-153-1 definited hsa-mir-148a undefinited

hsa-mir-154 definited hsa-mir-181a-2 undefinited

hsa-mir-155 definited hsa-mir-181b-1 undefinited


**Table 5.** Top 20 miRNAs with resistance scores to Doxorubicin


**miRNA** **ncDR** **miRNA** **ncDR**


hsa-mir-10a definited hsa-mir-206 definited

hsa-mir-181b-1 definited hsa-mir-20a definited

hsa-mir-193a definited hsa-mir-20b definited

hsa-mir-21 definited hsa-mir-210 definited

hsa-mir-27a definited hsa-mir-16-1 definited

hsa-let-7a-1 definited hsa-mir-17 definited

hsa-let-7b definited hsa-mir-181a-2 definited

hsa-let-7d definited hsa-mir-181c definited

hsa-mir-203a definited hsa-mir-181d definited

hsa-mir-205 definited hsa-mir-10a undefinited


During model training, the drugs Verapamil and Doxorubicin
(Adriamycin), along with their associated miRNAs, were excluded
from the dataset. In the model inference phase, these two drugs
were added to the test set, and the drug resistance scores for
the miRNAs against these two drugs were calculated and ranked,
respectively. For the purpose of this study, the top 20 miRNAs
with the highest drug resistance scores for these two drugs were
selected and added to Tables 4 and 5. The experimental results
demonstrate that out of the top 20 miRNAs predicted by the model
for drug resistance to Verapamil and Doxorubicin (Adriamycin),
17 and 19 miRNAs, respectively, have been validated in actual
databases. These findings suggest that the proposed GAM-MDR
model indeed has the ability to select high-confidence miRNA
targets. This can serve as a reference to some extent for future
treatment strategies.



CONCLUSION


Investigating MDR carries profound implications for disease treatment. Computational methodologies for unearthing potential
MDR have emerged as pivotal channels for refining existing
treatment strategies and pinpointing novel therapeutic targets.
However, contemporary machine learning and deep learning
strategies for pinpointing potential MDR often encounter obstacles due to data noise and inadequate information extraction,
resulting in less-than-optimal prediction outcomes. In response
to these challenges, we devise the GAM-MDR model predicated
on a GAE and a random walk path masking approach, with
the objective of amplifying the efficiency of MDR predictions.
The GAM-MDR models allow for the amalgamation of initial
node feature and the topological information from the miRNA–
drug graph, fully extracting the representations of miRNA (or
drug) nodes. The designed random walk path masking strategy
can effectively curtail the impact of noise and bolster the
model’s performance. Multiple experiment results affirmed
the high efficiency and stability of the GAM-MDR model. The
outcomes from the parameter experiments also attest to the
model’s robustness, as parameters can be conveniently adjusted,
thereby enabling swift application for the exploration of potential
MDR. The GAM-MDR model exhibits its ability to discover
potential MDR, possibly illuminating deeper relationships
between miRNA and drugs, and uncovering a broader array of
potential targets. We envisage the proposed GAM-MDR model
delivering valuable insights for the development of new treatment
paradigms or the design of novel drugs in the foreseeable
future.


**Key Points**


  - We propose a novel MDR predictor utilizing a graph
autoencoder combined with a random path masking

strategy.

  - Graph autoencoders are introduced to accurately extract
node representations on miRNA-drug graph.

  - The implementation of a random path masking strategy
enhances node representation.


REFERENCES


1. Cai Y, Xiaomin Y, Songnian H, Jun Y. A brief review on the mechanisms of mirna regulation. _Genom Proteom Bioinform_ 2009; **7** (4):

147–54.

2. Zheng K, Zhao H, Zhao Q, _et al._ Nasmdr: a framework for
mirna-drug resistance prediction using efficient neural architecture search and graph isomorphism networks. _Brief Bioinform_
2022; **23** (5):bbac338.
3. Afonso-Grunz F, Müller S. Principles of mirna–mrna interactions: beyond sequence complementarity. _Cell Mol Life Sci_

2015; **72** :3127–41.

4. Mørk S, Pletscher-Frankild S, Caro AP, _et al._ Protein-driven inference of mirna–disease associations. _Bioinformatics_ 2014; **30** (3):

392–7.

5. Lindow M, Kauppinen S. Discovering the first microrna-targeted
drug. _Journal of Cell Biology_ . NY, USA: The Rockefeller University

Press, 2012.

6. Ishida M, Selaru FM. Mirna-based therapeutic strategies. _Curr_
_Pathobiol Rep_ 2013; **1** :63–70.
7. Johnston M. Gene chips: array of hope for understanding gene
regulation. _Curr Biol_ 1998; **8** (5):R171–4.
8. Ozsolak F, Milos PM. Rna sequencing: advances, challenges and
opportunities. _Nat Rev Genet_ 2011; **12** (2):87–98.
9. Heid CA, Stevens J, Livak KJ, _et al._ Real time quantitative pcr.
_Genome Res_ 1996; **6** (10):986–94.
10. Sarkar FH, Li Y, Wang Z, _et al._ Implication of micrornas in drug
resistance for designing novel cancer therapy. _Drug Resist Updat_
2010; **13** (3):57–66.
11. Shi Y, Yang F, Wei S, Gang X. Identification of key genes affecting
results of hyperthermia in osteosarcoma based on integrative
chip-seq/targetscan analysis. _Med Sci Monit_ 2017; **23** :2042–8.
12. Peterson SM, Thompson JA, Ufkin ML, _et al._ Common features of
microrna target prediction tools. _Front Genet_ 2014; **5** :23.
13. Gong J, Tong Y, Zhang H-M, _et al._ Genome-wide identification of
snps in microrna genes and the snp effects on microrna target
binding and biogenesis. _Hum Mutat_ 2012; **33** (1):254–63.
14. Khan ZU, Pi D, Yao S, _et al._ pienpred: a bi-layered discriminative
model for enhancers and their subtypes via novel cascade multilevel subset feature selection algorithm. _Front Comp Sci_ 2021; **15** :

1–11.

15. Wang Y, You Z, Li L, Chen Z. A survey of current trends in
computational predictions of protein-protein interactions. _Front_
_Comp Sci_ 2020; **14** :1–12.
16. Sori WJ, Feng J, Godana AW, _et al._ Dfd-net: lung cancer detection
from denoised ct scan image using deep learning. _Front Comp Sci_

2021; **15** :1–13.

17. Xie W-B, Yan H, Zhao X-M. Emdl: extracting mirna-drug interactions from literature. _IEEE/ACM Trans Comput Biol Bioinform_
2017; **16** (5):1722–8.
18. Chen H, Zhang Z, Peng W. Mirddcr: a mirna-based method to
comprehensively infer drug-disease causal relationships. _Sci Rep_
2017; **7** (1):15921.



_Zhou_ et al. | 483


19. Guo Z-H, You Z-H, Li L-P, _et al._ Inferring drug-mirna associations
by integrating drug smiles and mirna sequence information.
In: De-Shuang Huang, Kang-Hyun Jo (eds). _Intelligent Computing_
_Theories and Application: 16th International Conference, ICIC 2020_,
Bari, Italy, October 2–5, 2020, 2020, Proceedings, Part II 16,
p. 279–89. Bari, Italy: Springer.
20. Zhou X, Dai E, Song Q, _et al._ In silico drug repositioning based on
drug-mirna associations. _Brief Bioinform_ 2020; **21** (2):498–510.
21. Guan Y-J, Chang-Qing Y, Qiao Y, _et al._ Mfidma: a multiple
information integration model for the prediction of drug–mirna
associations. _Biology_ 2022; **12** (1):41.
22. Deepthi K, Jereesh AS. An ensemble approach based on
multi-source information to predict drug-mirna associations
via convolutional neural networks. _IEEE_ _Access_ 2021; **9** :

38331–41.

23. Zheng K, You Z-H, Wang L, _et al._ Mlmda: a machine learning
approach to predict and validate microrna–disease associations
by integrating of heterogenous information sources. _J Transl Med_
2019; **17** (1):1–14.
24. Zheng K, You Z-H, Li J-Q, _et al._ Icda-cgr: identification of circrnadisease associations based on chaos game representation. _PLoS_
_Comput Biol_ 2020; **16** (5):e1007872.
25. Huang Y-A, Pengwei H, Chan KCC, You Z-H. Graph convolution
for predicting associations between mirna and drug resistance.
_Bioinformatics_ 2020; **36** (3):851–8.
26. Wei J, Zhuo L, Zhou Z, _et al._ Gcfmcl: predicting mirna-drug
sensitivity using graph collaborative filtering and multi-view
contrastive learning. _Brief Bioinform_ 2023; **24** :bbad247.
27. Kipf TN, Welling M. Variational graph auto-encoders arXiv
preprint arXiv:1611.07308. 2016.
28. Li J, Wu R, Sun W, _et al._ What’s behind the mask: Understanding masked graph modeling for graph autoencoders. In:
Ambuj K. Singh, Yizhou Sun, Leman Akoglu, Dimitrios Gunopulos, Xifeng Yan, Ravi Kumar, Fatma Ozcan, Jieping Ye (eds).
_Proceedings of the 29th ACM SIGKDD Conference on Knowledge_
_Discovery and Data Mining_ . Long Beach, CA, USA: ACM, 2023,

p. 1268–79.
29. Hamilton W, Ying Z, Leskovec J. Inductive representation
learning on large graphs. _Adv Neural Inf Process Syst_ 2017; **30** :

1024–34.

30. Kipf TN, Welling M. Semi-supervised classification with graph
convolutional networks. _5th International Conference on Learning_
_Representations_ . Toulon, Franceurl: OpenReview.net, 2017.
31. Xu K, Hu W, Leskovec J, Jegelka S. How powerful are graph neural
networks? _7th International Conference on Learning Representations_ .
New Orleans, LA, USA: OpenReview.net, 2018.
32. Velickovi´c P, Cucurull G, Casanova A,ˇ _et al._ Graph attention networks. _CoRR_ . 2017; **abs/1710.10903** [. http://arxiv.org/](http://arxiv.org/abs/1710.10903)
[abs/1710.10903.](http://arxiv.org/abs/1710.10903)

33. Brody S, Alon U, Yahav E. How attentive are graph attention
networks? _The Tenth International Conference on Learning Represen-_
_tations_ . Virtual Event: OpenReview.net, 2021.


