[Artificial Intelligence In Medicine 144 (2023) 102640](https://doi.org/10.1016/j.artmed.2023.102640)


Contents lists available at ScienceDirect
# Artificial Intelligence In Medicine


[journal homepage: www.elsevier.com/locate/artmed](https://www.elsevier.com/locate/artmed)


Research paper

## DDI-GCN: Drug-drug interaction prediction via explainable graph convolutional networks


Yi Zhong [b] [,] [1], Houbing Zheng [c] [,] [1], Xiaoming Chen [b], Yu Zhao [b], Tingfang Gao [a], Huiqun Dong [a],
Heng Luo [b] [,] [d] [,] [*], Zuquan Weng [a] [,] [b] [,] [c] [,] [* ]


a _College of Biological Science and Engineering, Fuzhou University, Fujian Province, China_
b _The Center for Big Data Research in Burns and Trauma, College of Computer and Data Science/College of Software, Fuzhou University, Fujian Province, China_
cd _Department of Plastic Surgery, the First Affiliated Hospital of Fujian Medical University, Fuzhou, China MetaNovas Biotech Inc., Foster City, CA, USA_



A R T I C L E I N F O


_Keywords:_
Drug-drug interactions prediction
Drug safety
Graph convolutional networks

Chemical structures

Web server


**1. Introduction**



A B S T R A C T


Drug-drug interactions (DDI) may lead to unexpected side effects, which is a growing concern in both academia
and industry. Many DDIs have been reported, but the underlying mechanisms are not well understood. Predicting
and understanding DDIs can help researchers to improve drug safety and protect patient health. Here, we
introduce DDI-GCN, a method that utilizes graph convolutional networks (GCN) to predict DDIs based on
chemical structures. We demonstrate that this method achieves state-of-the-art prediction performance on the
independent hold-out set. It can also provide visualization of structural features associated with DDIs, which can
help us to study the underlying mechanisms. To make it easy and accessible to use, we developed a web server for
[DDI-GCN, which is freely available at http://wengzq-lab.cn/ddi/.](http://wengzq-lab.cn/ddi/)



Combinatorial drug treatment is common in the clinic for treating
patients with complex diseases. However, it also increases the chance of
drug-drug interactions (DDI). As a major type of adverse drug events,
DDIs may lead to patient hospitalization or even drug withdrawal from
market [1–3]. Therefore, it is important to identify DDIs and understand
possible mechanisms to better improve drug safety and protect patient
health. Nevertheless, large scale of DDI screening in the wet-lab is both
costly and impractical to enumerate all types of drug combinations.
Recently, machine learning approaches have been developed to
identify potential DDIs by utilizing chemical, biological and phenotypic
data of drugs. For example, similarity-based machine learning models
can calculate pairwise similarities of chemical structures, targets,
pathways, side effects and other data types to infer DDIs based on the
existing ones [4–9]. Network-based models may extract features from
drug and protein networks to predict DDIs [10–12]. While many of these
methods were shown to identify DDIs with reliable performance, the
majority required comprehensive information such as drug targets and



side effect information, which are not always available, especially for
drug candidates in the early development stage.
To address this problem, chemical structure-based models were
proposed to infer DDIs based on drug structural information [6,13,14].
In these models, structural representations such as simplified molecular
input line entry specification (SMILES), molecular fingerprints and
molecular graph representations were used as features. SMILES is a textbased linear representation for molecular structures, which can be
treated as a string and converted into one-hot encoding data as inputs

[14–16]. However, one molecule can be represented by more than one
SMILES which may create challenging tasks for the model to learn

[17,18]. Additionally, it is hard for the models to understand the asso­
ciation between the SMILES text and the actual structure of a molecule
due to the complex grammar [19,20]. As an alternative, molecular fin­
gerprints encode the topological properties of a molecule as features

[21,22]. Several molecular fingerprints such as extended connectivity
23
fingerprints (ECFP) [ ], molecular access systems keys fingerprints
(MACCS) [24] and PubChem fingerprints [25] were used as descriptors
of molecular structures to characterize structural similarities [6,7,26]




 - Corresponding authors at: The Center for Big Data Research in Burns and Trauma, College of Computer and Data Science/College of Software, Fuzhou University,
Fujian Province, China.
_E-mail addresses:_ [hengluo88@gmail.com (H. Luo), wengzq@fzu.edu.cn (Z. Weng).](mailto:hengluo88@gmail.com)
1 These authors contributed equally to this work.


[https://doi.org/10.1016/j.artmed.2023.102640](https://doi.org/10.1016/j.artmed.2023.102640)
Received 27 June 2022; Received in revised form 21 March 2023; Accepted 20 August 2023

Available online 21 August 2023
0933-3657/© 2023 Elsevier B.V. All rights reserved.


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_



and predict biological activities [27,28]. However, the molecular fin­
gerprints are abstraction of molecular structures and do not retain all the
structural details. As an improvement, recent studies leveraged the
molecular graph representations, which converted a molecule into a
graph in which the atoms became nodes and bonds became edges

[21,22]. Graph convolutional networks (GCN) were then utilized to
generate the graphic features by aggregating information for each node
from their direct neighbors as well as indirect neighbors several hops
away [29,30]. Since the molecular graph captured more details on
structural information, it was able to obtain better performance and
provide meaningful interpretations on the structural level [13,31–34].
Since DDIs are directly or indirectly associated with the chemical
structures of the individual drugs, in this study, we introduce DDI-GCN,
a deep learning framework to predict DDIs and DDI mechanisms via
GCNs based on molecule structures. It can also highlight substructures
on the drug molecules that may be associated with the DDIs using the
attentive neural networks. In order to make it easy to use, we developed



a webserver based on the model for DDI exploration, which is freely
[available at http://wengzq-lab.cn/ddi/.](http://wengzq-lab.cn/ddi/)


**2. Methods**


_2.1. Problem definition_


We formulate the DDI prediction as a classification problem to
determine the interaction labels between two molecules. In our setting, a
molecule is represented by a graph _G_ = ( _V, E_ ), where _V_ and _E_ are the sets
of nodes and edges, respectively. Given a DDI dataset _D_ =
{( _G_ _A_ _, G_ _B_ ) _i_ _, y_ _i_ } _qi_ =1 [of ] _[q ]_ [drug pairs and a set of ] _[p ]_ [interaction labels ] _[C]_ [ =]

{ _c_ _i_ } _[p]_ ⅈ=1 [, where (] _[G]_ _[A]_ _[,][ G]_ _[B]_ [)] _[i ]_ [is a pair of molecular graphs with the corre­]
sponding ground-truth label _y_ _i_ ∈ _C_ . The goal of the DDI prediction task is
to learn a function that can accurately map any molecule pair to a DDI
label _c_ _i_ _._



**Fig. 1.** The architecture of DDI-GCN. (A) The GCNs aggregate features of a node along with its neighbor nodes and bonds as the new node features. (B) The neural
network of a bi-GRU layer and an attention layer (BiGRU-Att) determines the receptive field and outputs the weighted node feature vector for each node. (C) The coattention network learns the interactive information from a given molecular pair and assigns attention weights to all nodes within a molecule via attention
mechanism. (D) The overall architecture of DDI-GCN.


2


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_



_2.2. Graph convolutional networks_


The molecular graphs were fed into GCNs to extract graph features
(Fig. 1A). The GCNs generated a matrix _D_ ∈ _ℝ_ _[V]_ [×] _[H]_ of a given molecule
graph, where _V_ and _H_ denoted the number of nodes and the dimension
of node features, respectively [31].
For each node _v_ in _V_, we initialized its feature _f_ _v_ and updated it as
follows:



_Z_ _v_ into scalar values, and a softmax function yielded the attention scores


_L_

_S_ _v_ = { _s_ [(] _v_ _[l]_ [)] } 0 [for all GCN layers. The weighted sums were calculated by ]

multiplying _S_ _v_ and the hidden states _Z_ _v_ as the final node feature
_f_ _v_ _[final]_ ∈ _ℝ_ [1][×][2] _[K]_ .


_2.4. Co-attention networks_


Co-attention networks have been widely used for visual question
answering (VQA), as they link the most relevant parts from the visual
and textual features for visual reasoning [38,39].
Inspired from Ma et al. [40], we utilized the co-attention networks
(Fig. 1C) to find the most important structural features within a mo­
lecular pair that could be linked to the DDIs. Given two molecules a and
b, Eq. (7) above helped to yield node feature matrices _F_ _a_ ∈ _ℝ_ _[N]_ [×][2] _[K ]_ and
_F_ _b_ ∈ _ℝ_ _[N]_ [×][2] _[K ]_ for molecules a and b, respectively. _N_ is the number of nodes

and 2 _K_ is the output dimension of the final node feature _f_ _v_ _[final]_ . To avoid a
situation that similar substructures in both molecule a and molecule b

may be given high scores [41], we transformed _F_ _a_ and _F_ _b_ as:


_D_ _a_ = _F_ _a_ _W_ _a_ _, W_ _a_ ∈ _ℝ_ [2] _[K]_ [×] _[H]_ (8)


_D_ _b_ = _F_ _b_ _W_ _b_ _, W_ _b_ ∈ _ℝ_ [2] _[K]_ [×] _[H]_ (9)


where _W_ _a_ and _W_ _b_ are learnable weight matrices, and _H_ equals the
dimension of node features described in Section 2.3. Let { _d_ _[a]_ _n_ [}][ and {] _[d]_ _[b]_ _n_ [}][ be ]
the columns of _D_ _[T]_ _a_ [and ] _[D]_ _[T]_ _b_ [, respectively. Then the base vector ] _[m]_ [0] [ ∈] _[ℝ]_ _[H]_ [×][1 ]

for the mutual information of the given molecular pair was calculated as
follows:


_m_ 0 = d _a_ ʘ d _b_ (10)


where



_, l_ = 0 _,_ 1 _,_ … _, L_ (1)

)



) _M_ [(] _[l]_ [)]



_f_ _v_ [(] _[l]_ [+][1][)] = _σ_



(



_f_ _v_ [(] _[l]_ [)] _[W]_ [(] _[l]_ [)] [ +] ∑



_concatenate_ ( _f_ _v_ [(] _i_ _[l]_ [)] _[,][ f]_ _e_ [ (] _i_ _[l]_ [)]



_i_ ∈ _N_ ( _v_ )



In the formula, _l_ represents the layer of GCNs which ranges from 0 to
_L_ . _L_ denotes the total number of layers for the GCNs, and it was set to 8 in
our model. _N_ ( _v_ ) represents the neighbor nodes of node _v_ . In each layer _l_,
the GCNs aggregated all the neighbor node features _f_ _v_ [(] _i_ _[l]_ [)] [∈] _[ℝ]_ [1][×] _[H ]_ [and edge ]

features _f_ _e_ [(] _i_ _[l]_ [)] [∈] _[ℝ]_ [1][×] _[E ]_ [of node ] _[v ]_ [and added them to the embeddings of node ]

_v_ ( _E_ denoted the dimension of edge features). _W_ [(] _[l]_ [)] and _M_ [(] _[l]_ [)] are the
learnable weight matrices associated with node _v_ and its neighbors _v_ _i_,
respectively. _σ_ () is the rectified linear unit (ReLU) activation function.
Note that _f_ _v_ [(][0][)] = _σ_ ( _f_ _v_ _W_ [(][0][)] [ )] when _l_ = 0, where _f_ _v_ is the initial features of

node _v_ .


_2.3. Receptive field awareness via attentive sequential models_


By default, GCNs used fixed-size subgraph structures (or receptive
fields) to extract node features in each layer. However, DDIs may be
associated with various sizes of substructures for different molecules.
Thus, the determination of different receptive fields is helpful for the
extraction of meaningful substructures by GCNs [35,36]. Inspired by Xu
et al. [37], we utilized bidirectional gated recurrent units with attention
mechanism (BiGRU-Att) (Fig. 1B), to determine the receptive field for
each node in the molecular graph. The bi-GRU layer enhanced node



} _L_



}
)


}
)



features { _f_ _v_ [(] _[l]_ [)]



0 [of various sized receptive fields. The attention mecha­]



d _a_ = _tanh_



1

_N_



∑{ _d_ _n_ _[a]_



∑



nism determined the meaningful receptive field for each node via



(



_n_



assigning an attention score _s_ [(] _v_ _[l]_ [)] for each GCN layer _l_
(∑ _L_



_s_ [(] _v_ _[l]_ [)] [=][ 1],
)



(



∑{ _d_ _n_ _[b]_


_n_



(11)


(12)



d _b_ = _tanh_



1

_N_



where each _s_ [(] _v_ _[l]_ [)] [represents the importance of learned feature of node ] _[v ]_ [on ]
the _l_ -th layer. The layer with the highest attention score became the
receptive field of node _v_ and the weighted average of layer features

_v_ .
yields the final feature of node
The details are formulated as follows:



ʘ is the element-wise product and _n_ ∈{1 _,_ 2 _,_ … _, N_ }. Then we calcu­
lated the attended feature vectors ( _D_ [*] _a_ [and ] _[D]_ [*] _b_ [) for a molecule pair by a ]
soft attention. Using molecule _a_ as an example, its attended feature
vector _D_ [*] _a_ [was calculated as: ]



̅→ ̅̅→ [)]
_z_ [(] _v_ _[l]_ [)] = _GRU_ _fwd_ ( _f_ _v_ [(] _[l]_ [)] _[,][ z]_ [(] _v_ _[l]_ [−] [1][)] (2)



←̅ ←̅̅ [)]
_z_ [(] _v_ _[l]_ [)] = _GRU_ _bwd_ ( _f_ _v_ [(] _[l]_ [)] _[,][ z]_ [(] _v_ _[l]_ [+][1][)] (3)



**̅** → ← **̅**
_z_ [(] _v_ _[l]_ [)] [=] _[ Concatenate]_ ( _z_ [(] _v_ _[l]_ [)] _, z_ [(] _v_ _[l]_ [)] ) (4)



_Z_ _v_ = ( _z_ [0] _v_ _[z]_ [1] _v_ [⋯] _[z]_ _[l]_ _v_ ) (5)



_S_ _v_ = _softmax_ ( _Θ_ _att_ _Z_ _v_ _[T]_



) (6)



_f_ _v_ _[final]_ = _S_ _v_ _Z_ _v_ (7)



_h_ _a_ = _tanh_ ( _W_ _q_ _D_ _[T]_ _a_ ) ʘ _tanh_ ( _W_ _m_ _m_ _0_ ) _, W_ _q_ ∈ _ℝ_ _[U]_ [×] _[H]_ _, W_ _m_ ∈ _ℝ_ _[U]_ [×] _[H]_ (13)


_∂_ _a_ = _softmax_ ( _W_ _h_ _h_ _a_ ) _, W_ _h_ ∈ _ℝ_ [1][×] _[U]_ (14)


_D_ [*] _a_ [=] _[ ∂]_ _[a]_ _[D]_ _[a]_ (15)


In the formulas, _U_ is the hyperparameter in co-attention and _W_ _q_, _W_ _m_
and _W_ _h_ are learnable weight matrices.


_2.5. Fully connected layers_


The co-attention network enables us to obtain two informative vec­
tors _D_ [*] _a_ [and ] _[D]_ [*] _b_ [that convey molecule graph information and substructure ]

interaction features for molecules a and b. We merged the _D_ [*] _a_ [and ] _[D]_ [*] _b_
using either concatenation or element-addition and fed them into
several fully connected layers (Fcl) for DDI prediction. We defined the
concatenation for _D_ [*] _a_ [and ] _[D]_ [*] _b_ [using two fully connected layers as follows: ]



Node features { _f_ _v_ [(] _[l]_ [)]



} _L_ 0 [were input of the bi-GRU layer to generate the ]



̅→
forward-GRU and backward-GRU hidden states _z_ [(] _v_ _[l]_ [)] ∈ _ℝ_ [1][×] _[K ]_ and


←̅ ̅→ ←̅
_z_ [(] _v_ _[l]_ [)] ∈ _ℝ_ [1][×] _[K]_, where _K_ represents the output dimension. _z_ [(] _v_ _[l]_ [)] and _z_ [(] _v_ _[l]_ [)] were
concatenated to obtain the hidden state _z_ [(] _v_ _[l]_ [)] [∈] _[ℝ]_ [1][×][2] _[K]_ [, which was a more ]
informative vector that captured the dependent relationship between
GCN layers (time steps) for each node _v_ . _Z_ _v_ ∈ _ℝ_ _[L]_ [×][2] _[K ]_ contained all node _v_
features from GCN layers. _Θ_ _att_ ∈ _ℝ_ [1][×][2] _[K]_ was a weight matrix that mapped



) ) ) ) (16)



( _Fcl_ 1 ( _Concatenate_ ( _D_ [*] _a_ _[,][ D]_ [*] _b_



̂
_y_ = _σ_ _p_



( _Fcl_ 2 ( _σ_ _r_



where _Fcl_ _i_ ( _x_ ) = _W_ _i_ _x_ + _b_ _i_, _i_ ∈{1 _,_ 2} _, W_ _i_ are the trainable matrices, _W_ 1 ∈
_ℝ_ [2] _[H]_ [×][C] _[i ]_ and _W_ 2 ∈ _ℝ_ [C] _[i]_ [×] _[p]_ . C _i_ is the dimension of hidden vector and _p_ is the



3


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_



number of DDI labels. _b_ i ∈ _ℝ_ is the bias parameter. The activation

̂
function _σ_ _r_ (⋅) is ReLU. _y_ ∈ _ℝ_ [1][×] _[p ]_ is the output of the softmax function
_σ_ _p_ (⋅). Given the dataset {( _G_ _A_ _, G_ _B_ ) _i_ _, y_ _i_ } _qi_ =1 [, the training objective is to ]

minimize the following cross-entropy loss:


_p_
_Loss_ = − ∑ _i_ =1 _[y]_ _[i]_ [⋅] _[log]_ [(][̂] _[y]_ _[i]_ [)] (17)


where ̂ _y_ _i_ is the _i_ -th scalar value in ̂ _y_ .


_2.6. Model overview_


The neural network architecture for DDI-GCN was illustrated in

Fig. 1D. DDI-GCN took inputs of two drug SMILES representations and
transformed them into molecule graphs. Then the parallel GCN-BiGRUAtt module updated features for each node within each molecule graph
and output graph node feature matrices. The co-attention module
computed interactive information of the molecular pair and output
attended vectors for each graph. Then a merge function merged all
vectors and fed them into fully connected networks to make final DDI
prediction.


**3. Experiments**


_3.1. Datasets_


DrugBank is a comprehensive resource for drugs which contains
DDIs and relevant descriptions extracted from the drug labels and the
literature [42]. In this study, we collected all drugs that have DDI in­
formation along with their chemical structures represented by SMILES
from DrugBank Version 5.1.3. Based on the labels, two objectives were
set in this study: (1) to predict whether a drug pair can cause DDI (Task
1) and (2) if the DDI exists, what the biological mechanism or conse­
quence is (Task 2). For the binary classification task (Task 1), all the
collected DDIs were labeled as the positive set and an equally sized
negative set was generated by either the positive-unlabeled learning
(PULearn) (Supplementary information) [43] or random sampling
method. For the multi-class classification task (Task 2), we identified
106 types of DDI biological mechanisms or consequences (Supplemen­
tary Table S1) with over 100 samples in each type. As a result, we
collected 1,948,436 drug pairs for Task 1 and 96,7202 DDIs with labeled
DDI types for Task 2, containing 2755 and 2689 drugs, respectively.


_3.2. Molecular featurization_


For each molecule, we used the RDKit library [44] to transform its
SMILES code to a molecule graph [31,45]. Each node was represented as
a 51-dimentional vector based on seven types of atomic features and
each edge was represented as a 10-dimentional vector based on four
types of bond features (Supplementary Table S2).


_3.3. Evaluation metrics_


In this study, DDI-GCN is used to predict DDI between two molecules
(Task 1) as well as their possible biological consequences (Task 2). For
each task, we generally focus on three scenarios in DDI prediction [46]:
(1) Scenario one ( _S_ 1 ) is to predict DDIs among seen drugs (drugs from
the training set), (2) Scenario two ( _S_ 2 ) is to predict DDIs between seen
drugs and unseen drugs, and (3) Scenario three ( _S_ 3 ) is to predict DDIs
among unseen drugs. Accordingly, we split the datasets by DDI (for _S_ 1 )
and by drug (for _S_ 2 and _S_ 3 ). For DDI-based splitting, we randomly held
out 10 % of the DDIs as the independent test set. For the remaining 90 %,
we trained our models and performed hyperparameter optimization
through 5-fold cross-validations. For drug-based splitting, we held out
drugs that were exclusively categorized as “Experimental” or “Investi­
gational” in DrugBank version 5.1.3 as unseen drugs, and the rest of



drugs (generally labeled as “Approved”) were used as seen drugs. The
DDIs between seen drugs and unseen drugs were used as independent
test set in _S_ 2 while DDIs among unseen drugs were the independent test
set in _S_ 3 . The splitting results are in Supplementary Table S3.
For both tasks, to assess the model performance, we calculated ac­
curacy (ACC), the area under receiver operating characteristic curve
(AUROC), and the area under precision-recall curve (AUPR) for Task 1.
We calculated ACC, micro metrics for AUROC and AUPR for Task 2
referring Deng et al. [46].


_3.4. Experimental setup_


Since DDI-GCN has two parallel GCN-BiGRU-Att networks, the re­
sults may change depending on the input order of drug pairs [14]. In
Task 1, we would like DDI-GCN to output identical results regardless of
the input order. Thus, we set the two GCN-BiGRU-Att networks to share
the same weight parameters and used element-wise addition to merge _D_ [*] _a_
and _D_ [*] _b_ [. In Task 2, the biological consequence of a DDI can be different ]
based on the input order of two drug molecules. For example, given an


outcome that “ e can be decreased when
the metabolism of Enfluran

combined with Ademetionine”, the meaning is altered if we exchange
“Enflurane” with “Ademetionine”. As a consequence, we kept the se­
mantic order of drug molecules to feed models, allowed same weight
parameters for the two GCN-BiGRU-Att networks and concatenated _D_ [*] _a_
and _D_ [*] _b_ [in Task 2. ]
Since the number of GCN layers was an important hyperparameter,
we evaluated the prediction performance on scenario one in Tasks 1 and
2 for a set of GCN layer counts (2, 4, 8, 12 and 16) and attached results in
Supplementary Table S4. Based on the test, we set up all models with 8
GCN layers with 128 dimensions. We trained with Xavier initializer,
Adam optimizer and a learning rate of 0.0001. _U_ in the co-attention
network was set to 65. We used four fully connected layers to output
the DDI prediction. The first three layers were 100 hidden neurons and
ReLU activation functions without dropout, and the final layer consist of
a softmax activation function and 2 neurons for Task 1 and 106 neurons

for Task 2. We deployed the early-stopping strategy to automatically
stop the training if no decrease of validation loss was observed for 10
epochs [46]. The details of hyperparameter were in Supplementary
Table S5.


_3.5. Substructure analysis for model interpretation_


The model was able to highlight substructures within the drug pairs
which may be associated with the DDIs. In a Visual Question Answering
(VQA) task, a co-attention network is used to associate image regions
with words within the question for a given answer [39,40]. Likewise, in
this study, we evaluated DDI-relevant substructures via attention scores
from the co-attention module. Referring to Dey et al. [47], we also
mathematically evaluate the associations between substructures and
DDIs by generating a confusion matrix (Table 1) and calculating the
odds ratio and _p_ -value using chi-squared test.


_3.6. Comparisons_


We compare the performance of DDI-GCN with following methods as
references:


**Table 1**

The confusion matrix to evaluate the association between substructure A and

DDI type X. In the table, _a_, _b_, _c_ and _d_ are integers calculated from the data.


Substructure A+ Substructure A−


DDI type X+ a b
DDI type X− c d



4


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_


**Table 2**
The performance of different models on DDI binary classification using PULearn-sampled negative samples. (1) Scenario one ( _S_ 1 ) is to predict DDIs among seen drugs
(drugs from the training set), (2) Scenario two ( _S_ 2 ) is to predict DDIs between seen drugs and unseen drugs, and (3) Scenario three ( _S_ 3 ) is to predict DDIs among unseen
drugs. The bold values are the best performance models for each metric.


Models _S_ _1_ _S_ _2_ _S_ _3_


ACC AUROC AUPR ACC AUROC AUPR ACC AUROC AUPR


Baseline LR 0.849 0.912 0.933 0.818 0.914 0.913 0.792 0.908 0.867

RF 0.921 0.967 0.975 0.891 0.952 0.951 **0.851** 0.932 0.904

DeepDDI 0.962 0.993 0.994 0.888 **0.959** 0.944 0.839 0.926 0.861

MR-GNN 0.978 0.998 0.998 0.890 0.952 0.934 0.818 **0.955** 0.871

SSI-DDI 0.980 0.997 0.998 0.892 0.951 0.934 0.828 **0.955** 0.873

Ablation No BiGRU-Att 0.990 **0.999** **0.999** 0.890 0.934 0.950 0.814 0.927 0.891

No Co-attention 0.989 **0.999** **0.999** 0.887 0.932 0.950 0.826 0.940 0.901

**DDI-GCN** **0.994** **0.999** **0.999** **0.901** 0.951 **0.961** 0.834 0.932 **0.906**



1. **Logistic regression (LR)** : a LR model with L2 regularization using
chemical structural similarity profiles (SSP) [26] of drug pairs to
predict DDIs.
2. **Random forest (RF)** : a RF model with 100 estimators using SSPs of
drug pairs to predict DDIs.
3. **DeepDDI** [26]: a deep learning model that computes SSPs of drugs
and utilized principal component analysis (PCA) to reduce features.
The reduced similarity profiles of drug pairs were fed into a deep
neural network to predict DDIs. We modified output units of the last
layer of DeepDDI for Tasks 1 and 2.
4. **MR-GNN** [48]: a multi-resolution end-to-end graph neural network
to predict DDIs.
5. **SSI-DDI** [41]: a knowledge-driven deep learning model to learn
substructure–substructure interaction for DDI prediction. SSI-DDI
transformed drugs and their relations of biological consequences
from known DDIs into a network for inference. It is a binary pre­
diction model.


**4. Results**


_4.1. Task 1: binary classification_


The performance of DDI-GCN and other models on the independent
test sets of three scenarios using PULearn-sampled negative samples
(PU-Learn negatives) was summarized in Table 2. From scenario one
( _S_ _1_ ) to scenario three ( _S_ _3_ ), as the prediction task became more difficult,
the metrices of all models gradually decrease. DDI-GCN outperformed
all other models on _S_ _1_ with ACC = 0.994, AUROC = 0.999 and AUPR =
0.999. In scenario two ( _S_ _2_ ), DDI-GCN exceeded all models in terms of
ACC (0.901) and AUPR (0.961), while the differences of AUROC values
were neglectable towards the cutting-edge models (AUROC = 0.951 for
DDI-GCN, AUROC = 0.959 for DeepDDI, AUROC = 0.952 for MR-GNN
and AUROC = 0.951 for SSI-DDI). In _S_ _3_, DDI-GCN obtained ACC =
0.834, AUROC = 0.932 and AUPR = 0.906. The RF model achieved the
highest ACC = 0.851 among all models, while MR-GNN and SSI-DDI had
the highest AUROC = 0.955. As DDI-GCN outperforms all models in
AUPR, there may be a trade-off among ACC, AUROC and AUPR in



difficult scenarios.
As a comparison, we also generated random negative samples
(random negatives) and evaluated the model performance using the
same metrics (Table 3). The results showed an obvious performance
decrease of all models in all scenarios when switching PU-Learn nega­
tives to random negatives. Nonetheless, DDI-GCN achieved the best
performance in most metrics.
In addition, we conducted ablation experiments of removing BiGRUAtt or the co-attention module to test the importance of these modules
(Tables 2 and 3). The results showed that combining BiGRU-Att and coattention modules helped DDI-GCN to perform the best.


_4.2. Task 2: categorical classification_


The models were tested on the independent test set for DDI cate­
gorical classification and the results were shown in Table 4. In this task,
DDI-GCN outperformed all the other models (including ablation models)
in all scenarios. The performance is ACC = 0.960, AUROC = 0.999 and
AUPR = 0.990 in _S_ _1_, ACC = 0.598, AUROC = 0.982 and AUPR = 0.619
in _S_ _2_ and ACC = 0.332, AUROC = 0.934 and AUPR = 0.273 in _S_ _3_ . In
terms of ablation studies, a performance boost was observed when both
BiGRU-Att and co-attention modules were both incorporated. Overall,
DDI-GCN had an improved performance of DDI prediction compared to
existing models.


_4.3. DDI type prediction and interpretation_


To test DDI-GCN on other datasets, we harvested a separate DDI
[dataset from a drug label database called Kangzhou Big Data (http](https://data.yaozh.com/)
[s://data.yaozh.com/). We excluded any DDIs that exists in our](https://data.yaozh.com/)
training set (DrugBank v5.1.3) and obtained 1316 DDIs from 533 drugs
(Supplementary Table S6). We utilized DDI-GCN to predict DDI proba­
bility and type for each drug pair in the dataset and compared the top ten
predictions to their labels in Kangzhou Big Data and DrugBank v5.1.8
(Table 5).
From the results, seven drug pairs (bold in Table 5) and their DDI
types predicted by DDI-GCN aligned with the labels in Kangzhou Big



**Table 3**
The performance of different models on DDI binary classification using randomly generated negative samples. The bold values are the best performance models for
each metric.


Models _S_ _1_ _S_ _2_ _S_ _3_


ACC AUROC AUPR ACC AUROC AUPR ACC AUROC AUPR


Baseline LR 0.617 0.663 0.646 0.517 0.645 0.506 0.413 0.621 0.319

RF 0.764 0.839 0.830 0.607 0.734 0.601 0.475 0.639 0.335

DeepDDI 0.920 0.976 0.975 0.722 0.808 0.690 0.600 0.668 0.363

MR-GNN 0.954 0.992 0.993 0.716 **0.809** **0.694** 0.568 **0.799** 0.377

SSI-DDI 0.961 0.993 0.994 0.710 0.753 0.686 0.558 0.645 0.364

Ablation No BiGRU-Att 0.951 0.992 0.993 0.715 0.798 0.674 0.602 0.787 0.369

No Co-attention 0.959 0.994 0.995 0.706 0.793 0.663 0.570 0.783 0.364

**DDI-GCN** **0.975** **0.997** **0.997** **0.740** 0.798 0.690 **0.607** 0.785 **0.388**


5


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_


**Table 4**
The performance of different models on DDI categorical classification. The bold values are the best performance models for each metric.


Models _S_ _1_ _S_ _2_ _S_ _3_


ACC AUROC AUPR ACC AUROC AUPR ACC AUROC AUPR


Baseline LR 0.362 0.969 0.321 0.294 0.956 0.238 0.188 0.929 0.136

RF 0.615 0.986 0.640 0.405 0.959 0.398 0.218 0.902 0.172

DeepDDI 0.897 0.995 0.750 0.542 0.957 0.404 0.292 0.872 0.216

MR-GNN 0.944 **0.999** 0.983 0.524 0.976 0.517 0.297 0.927 0.222

Ablation No BiGRU-Att 0.912 **0.999** 0.968 0.498 0.975 0.500 0.270 0.924 0.208

No Co-attention 0.947 **0.999** 0.985 0.513 0.977 0.515 0.279 0.926 0.214

**DDI-GCN** **0.960** **0.999** **0.990** **0.598** **0.982** **0.619** **0.332** **0.934** **0.273**



**Table 5**

Top ten predicted DDIs along with DDI types.


Drug A Drug B Predicted DDI type in
DDI type DrugBank

v5.1.8



DDI type in
Kangzhou



**Streptomycin** **Cefalotin** 19 2 a
**Gentamicin** **Cefalotin** 19 2 a

**Kanamycin** **Cefalotin** 19 2 a
**Tobramycin** **Cefalotin** 19 2 a

**Sisomicin** **Cefalotin** 19 2 a

**Isepamicin** **Cefalotin** 19 2 a
Eperisone Methocarbamol 3 b c

**Amikacin** **Cefalotin** 19 2 a

Bisacodyl Dihydrocodeine 6 6 e
Ambroxol Dextromethorphan 1 N/A d


N/A: Not available.

1: The metabolism of Drug A can be decreased when combined with Drug B.
2: Drug A may decrease the excretion rate of Drug B which could result in a
higher serum level.
3: The risk or severity of adverse effects can be increased when Drug A is
combined with Drug B.
6: The therapeutic efficacy of Drug A can be decreased when used in combina­
tion with Drug B.
19: The risk or severity of nephrotoxicity can be increased when Drug A is
combined with Drug B.
a: Local or systemic combination of this product with cefalotin may increase
nephrotoxicity.
b: The risk or severity of visual accommodation disturbances can be increased
when Drug A is combined with Drug B.
c: Methocarbamol combined with eperisone has been reported to have ocular
dysregulation.
d: It should be avoided to take it together with central antitussive drugs (such as
dextromethorphan, etc.) to avoid airway blockage caused by diluted sputum.
e: Cancer patients who take opioid analgesics have poor tolerance to Bisacodyl,
which may cause abdominal pain, diarrhea and fecal incontinence.


Data. All of them were predicted to have increased nephrotoxicity with
cefalotin which matched with Kangzhou. Another combination of bisa­
codyl and dihydrocodeine was predicted to have therapeutic efficacy
decrease, which was validated in DrugBank v5.1.8. For this drug pair,
Kangzhou labeled potential adverse effect consequences, which indi­
cated a drug combination may induce more than one biological conse­
quence [26]. Though some drug pairs induce multiple outcomes and
DDI-GCN may not identify all of them, the overall prediction still had
high concordance with DrugBank v5.1.8 and Kangzhou labels.
Next, we visualized the attention weights for the seven highlighted
drug pairs in Table 5, hoping to identify structural clues that link to this
DDI type (Fig. 2). Cefalotin is a cephalosporin antibiotic with a nucleus
substructure shown in Fig. 2B (left) [49] and its paired drugs are ami­
noglycoside antibiotics (AGs) that contain a 2-deoxystreptamine
(Fig. 2B, right) or a similar structure such as streptamine [50]. As
Fig. 2A illustrated, DDI-GCN paid more attention to the C-3 position of
the cephalosporin nucleus in cefalotin (Fig. 2B, left, green dashed cir­
cles). For AGs, DDI-GCN mostly highlighted the C-1, C-2 and C-3 posi­
tions (Fig. 2B, right, green dashed circles). DDI-GCN tended to visualize



the substructure made of C-1, C-2 and C-3 when none of hydrogens on
the amino-groups were substituted by other groups in 2-deoxystrept­
amine, while other cases only had one or two positions in C-1, C-2 and
C-3. Despite the small differences, the visualization suggested the sub­
structure pair of C-3 in cephalosporin nucleus and C-1, C-2 and C-3 in
AGs might be the important indicator of the DDIs between cephalo­
sporin and AGs (increased severity of nephrotoxicity). To validate the
hypothesis, we collected 49 cephalosporins and 14 AGs that are avail­
able and predicted all drug pairs. The confusion matrix of the sub­
structure pair of interest was calculated according to Table 1. The results
showed an odds ratio of 4.08 and a significant _p_ -value of 1.05 × 10 [−] [12 ]

using chi-squared test. Besides, research showed the C-3 position in the
cephalosporin nucleus is the key factor to the pharmacodynamics of
cephalosporin [51,52] and 2-deoxystreptamine plays a fundamental role
for the biological activity of the aminoglycosides [53], which validated
the findings of DDI-GCN. However, further biological experiments are
needed to evaluate the connection of the substructure pair and the DDI
mechanism.

Additional visualizations of DDI-GCN attention weights of quino­
lones and metal drugs are available in Supplementary information. We
believe the DDI prediction with structural interpretation may provide
helpful insights to understand the DDI mechanisms and guide the vali­
dation experiments in the wet lab.


_4.4. DDI semantic order_


DeepDDI is one of the early deep learning frameworks to output DDIs
as human-readable sentences [26]. However, many of the early models
ignored the semantic order of drugs in the sentences, thus cannot
directly determine “the perpetrator” and “the victim” in a DDI phar­
macological effect. In this work, we took the semantic order of drug
molecules to feed the model and trained the model to learn semantic

information (see Section 3.4). To test whether DDI-GCN is sensitive to
the semantic order, we reserved the semantic orders of drugs in the in­
dependent test set and computed the same evaluation metrics using the
pretrained model based on the scenario one in Task 2. Compared with
the original evaluation, the results (Fig. 3A) turned out that both ACC
and AUPR decreased dramatically (from 0.960 to 0.492 and from 0.990
to 0.463, respectively). Although about 49.2 % of drug pairs with
reverse order were predicted with the same DDI types, the prediction
probabilities of 70 % of these drug pairs were lower (Fig. 3B). Based on
this observation, given an unordered drug pair, DDI-GCN can determine
the semantic order by selecting the higher prediction probability.
Here, we provided a case study of a pharmacokinetic (PK) DDI to
demonstrate the importance of prediction with semantic order. PK-DDIs
occur when the perpetrator drug inhibits or induces the enzymes or
transporters that are responsible for the disposition (metabolism, elim­
ination, etc.) of a co-mediated agent (the victim drug) [54,55]. We
considered two critical PK-DDI types: “The metabolism of Molecule 1
can be decreased when combined with Molecule 2” (DDI type 1) and
“The metabolism of Molecule 1 can be increased when combined with

Molecule 2” (DDI type 4) [56]. The prediction challenge is to identify the
victim and the perpetrator for a given drug pair without order. 3245



6


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_


**Fig. 2.** (A) Attention visualization for the interactions between cefalotin and other seven aminoglycoside antibiotics. (B) The cephalosporin nucleus (left) and 2-deox­
ystreptamine structure in aminoglycoside antibiotics (right).


**Fig. 3.** (A) Evaluation results of independent test set with reversed semantic order compared with the original independent test set. (B) Among the 49.2 % drug pairs
with reverse order predicted with the unchanged DDI types, 70 % had lower prediction probabilities than their original order (S _>_ R). The remaining 5 % and 25 %
were equal to (S = R) and higher (S _<_ R) than their original order, respectively. Note that we denoted the reverse order as R and the original order as S (same).



**Table 6**
DDIs with confirmed types, drug victims and drug perpetrators.


The victim The perpetrator DDI type


Almotriptan Montelukast 1
Finasteride Hydrocortisone 1
Prochlorperazine Methylprednisolone 4
Bicalutamide Hydrocortisone 4

Finasteride Fluocinonide 4

Cephalexin Fluocinonide 4



unseen DDIs without mechanism descriptions from DDInter [57] were
harvested (Supplementary Table S7). We predicted the DDI types for
them using the pretrained model from scenario one in Task 2 for both
forward and reversed order. The input order with the higher predicted
probability was selected and the top 20 predictions of DDI type 1 and
DDI type 4 were validated in DrugBank v5.1.8. As a result, six DDI
predictions, including DDI types and drug orders, were confirmed
(Table 6). For example, almotriptan is a substrate for CYP2C8 [58,59]
and montelukast is a selective inhibitor for CYP2C8 [59]. The meta­
bolism of almotriptan was confirmed to be decreased by montelukast
when co-medicated. Another case, fluocinonide, a CYP3A5 inducer, can



7


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_



increase the metabolism of finasteride (a CYP3A5 substrate) according
to DrugBank. Though DDI pharmacological effects inferred by DDI-GCN
may not be studied clinically, they can still serve as references (or alerts)
before drugs are co-prescribed to patients.


**5. Discussion**


In this study, we developed a deep learning model that can extract
the structural features of drug pairs to predict DDIs. The model utilized
three modules including GCN, BiGRU-Att and co-attention. The com­
bined GCN and BiGRU-Att modules helped to adaptively learn the in­
formation of most useful neighborhood range for each node in a
molecular graph, which can help our model go much deeper than other
graph neural networks (GNN) such as SSI-DDI and capture more useful
structural information [37]. We trained DDI models for both binary and
categorical classifications (Task 1) and prediction of their possible bio­
logical mechanisms (Task 2). For each task, we adopted DDI-based and
drug-based data splitting strategies with three scenarios. According to
the evaluation results, DDI-GCN achieved competitive performance
compared to other models in both Tasks 1 and 2. Additionally, we found
that PULearn negatives can provide better prediction performance in
difficult prediction scenarios compared to random negatives.
We further used the case studies of cephalosporin-AG DDIs and
quinolone-metal drug DDIs (Supplementary information) to demon­
strate the capability of DDI-GCN prediction and structural interpreta­
tion, which may provide valuable clues for experimental researchers to
understand DDI mechanisms and structural associations. Besides,
different from DeepDDI, the output of DDI-GCN can be directly inter­
preted as human readable sentences with semantic orders.
Integration with additional features may improve the DDI prediction
performance [10,46]. To test this and demonstrate the scalability of
DDI-GCN in integrating additional information, we incorporated phys­
icochemical properties for DDI categorical classification in Task 2 (see
more details in Section 3 of the Supplementary information). The results
indicated that adding physicochemical properties improved the perfor­
mance in the complex DDI prediction task. Furthermore, if data are
available, it is possible to include patient factors, such as age, gender,
weight, and genetic background, to help with more accurate and
personalized DDI prediction.
However, further improvements are needed for DDI-GCN in the
future. First, the model architecture can be explored and improved. As
regular GCNs may suffer from complexity and redundant node infor­
mation in the process of propagation [60,61], the utilization of a GNN
architecture such as SparseShift-GCN [61] that can reduce feature
redundancy and calculation complexity can be a solution. Second, our
current molecular graph only considered the 2D structural properties of
a molecule, while a molecule may have more complicated forms such as
isomers and intramolecular hydrogens [62]. Third, we believe DDI-GCN
may provide clinicians with valuable information and help them to trade
off the benefits of polypharmacy and the adverse outcomes of DDIs when
making decisions. However, there may be gaps between in silico pre­
diction and clinical application settings. It is our future work to test DDIGCN in the clinical practice, incorporate individual patient data and
improve the model for clinical applications.


**6. Conclusion**


We developed DDI-GCN based on deep neural networks for DDI
predictions. It takes inputs of molecular structures and predicts both DDI
potential and DDI biological consequences with better performance
compared to other models. It can also highlight the substructures that
are associated with the DDI for better understanding and exploration of
the DDI mechanisms. Additionally, we developed a webserver based on
the model for DDI exploration, which is freely available at [http://](http://wengzq-lab.cn/ddi/)
[wengzq-lab.cn/ddi/. We believe DDI-GCN is a useful method to iden­](http://wengzq-lab.cn/ddi/)
tify and understand DDIs, which may have an implication to better



improve drug safety.


**Declaration of competing interest**


There is no conflict of interests regarding the publication of this
article.


**Data availability**


Supplementary data are available in supplementary files or on
[https://github.com/LabWeng/DDI-GCN.](https://github.com/LabWeng/DDI-GCN)


**Acknowledgment**


This work was supported by National Natural Science Foundation of
China (No. 81971837), Leading Project Foundation of Science and
Technology, Fujian Province (2022Y0015), Joint Funds for the Inno­
vation of Science and Technology, Fujian Province (2021Y9155).


**Appendix A. Supplementary data**


[Supplementary data to this article can be found online at https://doi.](https://doi.org/10.1016/j.artmed.2023.102640)
[org/10.1016/j.artmed.2023.102640.](https://doi.org/10.1016/j.artmed.2023.102640)


**References**


[[1] Mesgarpour B, Gouya G, Herkner H, et al. A population-based analysis of the risk of](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0005)
[drug interaction between clarithromycin and statins for hospitalisation or death.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0005)
[Lipids Health Dis 2015;14:1–6.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0005)

[[2] Heelon MW, Meade LB. Methadone withdrawal when starting an antiretroviral](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0010)
[regimen including nevirapine. Pharmacotherapy 1999;19:471–2.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0010)

[[3] Moura CS, Acurcio FA, Belo NO. Drug-drug interactions associated with length of](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0015)
[stay and cost of hospitalization. J Pharm Pharm Sci 2009;12:266–72.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0015)

[[4] Zhang W, Chen Y, Liu F, et al. Predicting potential drug-drug interactions by](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0020)
[integrating chemical, biological, phenotypic and network data. BMC Bioinf 2017;](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0020)
[18:1–12.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0020)

[[5] Takeda T, Hao M, Cheng T, et al. Predicting drug–drug interactions through drug](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0025)
[structural similarities and interaction networks incorporating pharmacokinetics](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0025)
[and pharmacodynamics knowledge. J Chem 2017;9:1–9.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0025)

[[6] Vilar S, Harpaz R, Uriarte E, et al. Drug—drug interaction through molecular](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0030)
[structure similarity analysis. J Am Med Inform Assoc 2012;19:1066–74.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0030)

[[7] Vilar S, Uriarte E, Santana L, et al. Similarity-based modeling in large-scale](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0035)
[prediction of drug-drug interactions. Nat Protoc 2014;9:2147.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0035)

[[8] Rohani N, Eslahchi C. Drug-drug interaction predicting by neural network using](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0040)
[integrated similarity. Sci Rep 2019;9:1–11.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0040)

[[9] Dere S, Ayvaz S. Prediction of drug–drug interactions by using profile fingerprint](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0045)
[vectors and protein similarities. Healthc Inf Res 2020;26:42–9.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0045)

[[10] Cheng F, Zhao Z. Machine learning-based prediction of drug–drug interactions by](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0050)
[integrating drug phenotypic, therapeutic, chemical, and genomic properties. J Am](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0050)
[Med Inform Assoc 2014;21:e278–86.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0050)

[[11] Park K, Kim D, Ha S, et al. Predicting pharmacodynamic drug-drug interactions](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0055)
[through signaling propagation interference on protein-protein interaction](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0055)
[networks. PLoS One 2015;10:e0140816.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0055)

[[12] Huang J, Niu C, Green CD, et al. Systematic prediction of pharmacodynamic drug-](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0060)
[drug interactions through protein-protein-interaction network. PLoS Comput Biol](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0060)
[2013;9:e1002998.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0060)

[[13] Chen X, Liu X, Wu J. Drug-drug interaction prediction with graph representation](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0065)
[learning. In: 2019 IEEE international conference on bioinformatics and](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0065)
[biomedicine (BIBM). IEEE; 2019. p. 354–61.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0065)

[[14] Kwon S, Yoon S. End-to-end representation learning for chemical-chemical](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0070)
[interaction prediction. IEEE/ACM Trans Comput Biol Bioinform 2018;16:1436–47.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0070)

[[15] Ertl P, Lewis R, Martin E, et al. In silico generation of novel, drug-like chemical](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0075)
[matter using the LSTM neural network [arXiv preprint arXiv:1712.07449]. 2017.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0075)

[[16] Bjerrum EJ, Threlfall R. Molecular generation with recurrent neural networks](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0080)
[(RNNs) [arXiv preprint arXiv:1705.04612]. 2017.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0080)

[[17] Elton DC, Boukouvalas Z, Fuge MD, et al. Deep learning for molecular design—a](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0085)
[review of the state of the art. Mol Syst Des Eng 2019;4:828–49.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0085)

[[18] O’Boyle NM, Dalke A. DeepSMILES: an adaptation of SMILES for use in machine-](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0090)
[learning of chemical structures [ChemRxiv]. 2018.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0090)

[[19] Xue D, Zhang H, Xiao D, et al. X-MOL: large-scale pre-training for molecular](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0095)
[understanding and diverse molecular analysis [bioRxiv]. 2021.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0095)

[[20] Xue D, Gong Y, Yang Z, et al. Advances and challenges in deep generative models](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0100)
[for de novo molecule generation. Wiley Interdiscip Rev: Comput Mol Sci 2019;9:](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0100)
[e1395.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0100)

[[21] Kwon Y, Lee D, Choi Y-S, et al. Compressed graph representation for scalable](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0105)
[molecular graph generation. J Chem 2020;12:1–8.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0105)

[[22] David L, Thakkar A, Mercado R, et al. Molecular representations in AI-driven drug](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0110)
[discovery: a review and practical guide. J Chem 2020;12:1–22.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0110)



8


_Y. Zhong et al._ _Artificial Intelligence In Medicine 144 (2023) 102640_




[[23] Rogers D, Hahn M. Extended-connectivity fingerprints. J Chem Inf Model 2010;50:](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0115)
[742–54.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0115)

[[24] Durant JL, Leland BA, Henry DR, et al. Reoptimization of MDL keys for use in drug](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0120)
[discovery. J Chem Inf Comput Sci 2002;42:1273–80.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0120)

[[25] Wang Y, Bryant SH, Cheng T, et al. Pubchem bioassay: 2017 update. Nucleic Acids](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0125)
[Res 2017;45:D955–63.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0125)

[[26] Ryu JY, Kim HU, Lee SY. Deep learning improves prediction of drug–drug and](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0130)
[drug–food interactions. Proc Natl Acad Sci 2018;115:E4304–11.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0130)

[[27] Jensen BF, Vind C, Padkjær SB, et al. In silico prediction of cytochrome P450 2D6](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0135)
[and 3A4 inhibition using Gaussian kernel weighted k-nearest neighbor and](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0135)
[extended connectivity fingerprints, including structural fragment analysis of](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0135)
[inhibitors versus noninhibitors. J Med Chem 2007;50:501–11.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0135)

[[28] Ayed M, Lim H, Xie L. Biological representation of chemicals using latent target](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0140)
[interaction profile. BMC Bioinf 2019;20:1–10.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0140)

[[29] Nikolentzos G, Dasoulas G, Vazirgiannis M. k-hop graph neural networks. Neural](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0145)
[Netw 2020;130:195–205.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0145)

[[30] Wang W, Yang X, Wu C, et al. CGINet: graph convolutional network-based model](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0150)
[for identifying chemical-gene interaction in an integrated multi-relational graph.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0150)
[BMC Bioinf 2020;21:1–17.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0150)

[[31] Duvenaud D, Maclaurin D, Aguilera-Iparraguirre J, et al. Convolutional networks](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0155)
[on graphs for learning molecular fingerprints [arXiv preprint arXiv:1509.09292].](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0155)
[2015.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0155)

[[32] Torng W, Altman RB. Graph convolutional neural networks for predicting drug-](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0160)
[target interactions. J Chem Inf Model 2019;59:4131–49.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0160)

[[33] Gao KY, Fokoue A, Luo H, et al. Interpretable drug target prediction using deep](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0165)
[neural representation. In: IJCAI; 2018. p. 3371–7.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0165)

[[34] Coley CW, Jin W, Rogers L, et al. A graph-convolutional neural network model for](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0170)
[the prediction of chemical reactivity. Chem Sci 2019;10:370–7.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0170)

[[35] Xu N, Wang P, Chen L, et al. MR-GNN: multi-resolution and dual graph neural](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0175)
[network for predicting structured entity interactions. In: Twenty-Eighth](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0175)
[International Joint Conference on Artificial Intelligence IJCAI-19; 2019.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0175)

[[36] Kazi A, Shekarforoush S, Krishna SA, et al. InceptionGCN: receptive field aware](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0180)
[graph convolutional network for disease prediction. In: International conference on](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0180)
[information processing in medical imaging. Springer; 2019. p. 73–85.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0180)

[[37] Xu K, Li C, Tian Y, et al. Representation learning on graphs with jumping](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0185)
[knowledge networks. In: International conference on machine learning. PMLR;](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0185)
[2018. p. 5453–62.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0185)

[[38] Yu Z, Yu J, Cui Y, et al. Deep modular co-attention networks for visual question](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0190)
[answering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0190)
[Pattern Recognition; 2019. p. 6281–90.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0190)

[[39] Lu J, Yang J, Batra D, et al. Hierarchical question-image co-attention for visual](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0195)
[question answering. In: Proceedings of the 30th International Conference on](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0195)
[Neural Information Processing Systems. Barcelona, Spain: Curran Associates Inc.;](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0195)
[2016. p. 289–97.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0195)

[[40] Ma C, Shen C, Dick A, et al. Visual question answering with memory-augmented](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0200)
[networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0200)
[Recognition; 2018. p. 6975–84.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0200)

[[41] Nyamabo AK, Yu H, Shi J-Y. SSI–DDI: substructure–substructure interactions for](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0205)
[drug–drug interaction prediction. Brief Bioinform 2021;22(6):bbab133.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0205)




[[42] Wishart DS, Feunang YD, Guo AC, et al. DrugBank 5.0: a major update to the](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0210)
[DrugBank database for 2018. Nucleic Acids Res 2018;46:D1074–82.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0210)

[[43] Zheng Y, Peng H, Zhang X, et al. DDI-PULearn: a positive-unlabeled learning](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0215)
[method for large-scale prediction of drug-drug interactions. BMC Bioinf 2019;20:](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0215)
[1–12.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0215)

[[44] Landrum G. RDKit: a software suite for cheminformatics, computational chemistry,](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0220)
[and predictive modeling. Academic Press; 2013.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0220)

[[45] Jiang D, Hsieh C-Y, Wu Z, et al. InteractionGraphNet: a novel and efficient deep](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0225)
[graph representation learning framework for accurate protein–ligand interaction](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0225)
[predictions. J Med Chem 2021;64:18209–32.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0225)

[[46] Deng Y, Xu X, Qiu Y, et al. A multimodal deep learning framework for predicting](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0230)
[drug–drug interaction events. Bioinformatics 2020;36:4316–22.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0230)

[[47] Dey S, Luo H, Fokoue A, et al. Predicting adverse drug reactions through](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0235)
[interpretable deep learning framework. BMC Bioinf 2018;19:1–13.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0235)

[[48] Xu N, Wang P, Chen L, et al. Mr-gnn: multi-resolution and dual graph neural](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0240)
[network for predicting structured entity interactions. In: International Joint](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0240)
[Conferences on Artificial Intelligence; 2019. p. 3968–74.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0240)

[[49] Mackenzie A. Studies on the biosynthetic pathways of clavulanic acid and](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0245)
cephamycin C in _[Streptomyces clavuligerus](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0245)_ . 2007.

[[50] Krause KM, Serio AW, Kane TR, et al. Aminoglycosides: an overview. Cold Spring](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0250)
[Harb Perspect Med 2016;6:a027029.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0250)

[[51] Bryskier A. New concepts in the field of cephalosporins: C-3](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0255) ′ quaternary
[ammonium cephems (Group IV). Clin Microbiol Infect 1997;3:s1–6.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0255)

[[52] Rahman MS, Koh Y-S. A novel antibiotic agent, cefiderocol, for multidrug-resistant](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0260)
[Gram-negative bacteria. J Bacteriol Virol 2020;50:218–26.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0260)

[[53] Busscher GF, Rutjes FP, Van Delft FL. 2-Deoxystreptamine: central scaffold of](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0265)
[aminoglycoside antibiotics. Chem Rev 2005;105:775–92.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0265)

[[54] Lu C, Di L. In vitro and in vivo methods to assess pharmacokinetic drug–drug](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0270)
[interactions in drug discovery and development. Biopharm Drug Dispos 2020;41:](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0270)
[3–31.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0270)

[[55] Levˆeque D, Lemachatti J, Nivoix Y, et al. Mechanisms of pharmacokinetic drug-](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0275)
[drug interactions. Rev Med Interne 2009;31:170–9.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0275)

[[56] Bibi Z. Role of cytochrome P450 in drug interactions. Nutr Metab 2008;5:1–10.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0280)

[[57] Xiong G, Yang Z, Yi J, et al. DDInter: an online drug–drug interaction database](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0285)
[towards improving clinical decision-making and patient safety. Nucleic Acids Res](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0285)
[2022;50:D1200–7.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0285)

[[58] Salva M, Jansat JM, Martinez-Tobed A, et al. Identification of the human liver](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0290)
[enzymes involved in the metabolism of the antimigraine agent almotriptan. Drug](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0290)
[Metab Dispos 2003;31:404–11.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0290)

[[59] Walsky RL, Obach RS, Gaman EA, et al. Selective inhibition of human cytochrome](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0295)
[P4502C8 by montelukast. Drug Metab Dispos 2005;33:413–8.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0295)

[[60] Gao Z, Shi J, Wang J. GQ-GCN: group quadratic graph convolutional network for](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0300)
[classification of histopathological images. In: International conference on medical](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0300)
[image computing and computer-assisted intervention. Springer; 2021. p. 121–31.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0300)

[[61] Zang Y, Yang D, Liu T, et al. SparseShift-GCN: high precision skeleton-based action](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0305)
[recognition. Pattern Recogn Lett 2022;153:136–43.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0305)

[[62] Xiong Z, Wang D, Liu X, et al. Pushing the boundaries of molecular representation](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0310)
[for drug discovery with the graph attention mechanism. J Med Chem 2019;63:](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0310)
[8749–60.](http://refhub.elsevier.com/S0933-3657(23)00154-9/rf0310)



9


