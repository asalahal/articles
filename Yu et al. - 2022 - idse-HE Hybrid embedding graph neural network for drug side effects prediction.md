[Journal of Biomedical Informatics 131 (2022) 104098](https://doi.org/10.1016/j.jbi.2022.104098)


Contents lists available at ScienceDirect

# Journal of Biomedical Informatics


[journal homepage: www.elsevier.com/locate/yjbin](https://www.elsevier.com/locate/yjbin)

## idse-HE: Hybrid embedding graph neural network for drug side effects prediction


Liyi Yu, Meiling Cheng, Wangren Qiu [*], Xuan Xiao [*], Weizhong Lin


_School of Information Engineering, Jingdezhen Ceramic Institute, Jingdezhen 333403, China_



A R T I C L E I N F O


_Keywords:_
Drug side effect
Graph neural network
Drug molecular structure

Matrix reconstruction


**1. Introduction**



A B S T R A C T


In drug development, unexpected side effects are the main reason for the failure of candidate drug trials.
Discovering potential side effects of drugs in silico can improve the success rate of drug screening. However, most
previous works extracted and utilized an effective representation of drugs from a single perspective. These
methods merely considered the topological information of drug in the biological entity network, or combined the
association information (e.g. knowledge graph KG) between drug and other biomarkers, or only used the
chemical structure or sequence information of drug.
Consequently, to jointly learn drug features from both the macroscopic biological network and the microscopic
drug molecules. We propose a hybrid embedding graph neural network model named idse-HE, which integrates
graph embedding module and node embedding module. idse-HE can fuse the drug chemical structure infor­
mation, the drug substructure sequence information and the drug network topology information. Our model
deems the final representation of drugs and side effects as two implicit factors to reconstruct the original matrix
and predicts the potential side effects of drugs. In the robustness experiment, idse-HE shows stable performance
in all indicators. We reproduce the baselines under the same conditions, and the experimental results indicate
that idse-HE is superior to other advanced methods. Finally, we also collect evidence to confirm several real drug
side effect pairs in the predicted results, which were previously regarded as negative samples. More detailed
information, scientific researchers can access the user-friendly web-server of idse-HE at _[http://bioinfo.jcu.edu.](http://bioinfo.jcu.edu.cn/idse-HE)_
_[cn/idse-HE](http://bioinfo.jcu.edu.cn/idse-HE)_ . In this server, users can obtain the original data and source code, and will be guided to reproduce
the model results.



Due to adverse reactions and side effects in the clinical trials of

pharmaceutical compounds, candidates were rejected for approval, and
the rate of drug wear increased significantly [1]. Therefore, side effects
are also considered a major burden in the modern drug discovery pro­
cess. In addition, side effects caused by the combined use of two or more
drugs are another serious drug safety issue. Generally, when multiple
types of drugs interact in the human body, the chemical reaction of
drugs will cause therapeutic effects or undesired reactions, and lifethreatening in extreme cases [2,3]. The time for drug premarket trials
is limited, lack of breadth for trial subjects, and many potential side
effects are difficult to detect in the early stages of drug development [4].
Moreover, it is extremely time-consuming and costly to explore as many
side effects as possible during the experimental phase.
Compared to traditional wet experiments, drug discovery in silico


 - Corresponding author.
_E-mail addresses:_ [qiuone@163.com (W. Qiu), jdzxiaoxuan@163.com (X. Xiao).](mailto:qiuone@163.com)



methods, mostly based on machine learning, can speed up the screening
process and reduce costs [5]. For example, the biological relationship
between drugs and side effects is understood as a network graph, in
which diffusion techniques are used to disseminate the information of
each node [6–9]. DeepWalk [10] applies the random walk strategy to
obtain the node topology sequence and then generates the node
embedding by a skip-gram method. Homoplastically, node2vec [11]
adopts two walking strategies, breadth-first search and depth-first
search, which satisfies the similar characteristics of neighboring nodes
and considers the isomorphism of analogous nodes. These methods all
rely on the biological network structure to extract node features. In
addition, the network can be expressed as an adjacency matrix of node
relations. From the perspective of the recommendation system, pre­
dicting the potential relationships between drugs and side effects are
actually utilizing matrix factorization (MF) to compute hidden factors
for matrix completion, and reconstruct a new matrix that fits the original



[https://doi.org/10.1016/j.jbi.2022.104098](https://doi.org/10.1016/j.jbi.2022.104098)
Received 23 December 2021; Received in revised form 29 April 2022; Accepted 24 May 2022

Available online 28 May 2022
[1532-0464/© 2022 The Authors. Published by Elsevier Inc. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-](http://creativecommons.org/licenses/by-nc-nd/4.0/)
[nc-nd/4.0/).](http://creativecommons.org/licenses/by-nc-nd/4.0/)


_L. Yu et al._ _Journal of Biomedical Informatics 131 (2022) 104098_


**Fig. 1.** The entire workflow of idse-HE. (a) The adjacency matrix represents the drugs and side effects association network. (b) idse-HE constructs a bipartite graph to
predict possible edge connections between drugs and side effects. In terms of representation, idse-HE calculates molecular structure embeddings and molecular
fingerprints based on drug smiles, and side effects are represented as unique random vectors. (c) In the bipartite graph, each node performs an aggregation (average
pooling) operation on its neighbors to obtain neighbor information. Then each node updates its feature representation by splicing its current representation and
neighbor information. (d) Multiply the drug feature matrix and the side effect feature matrix to reconstruct the adjacency matrix. idse-HE updates the model pa­
rameters based on the cross-entropy between the new matrix and the original matrix.



matrix to calculate the probability of node links [12]. However, the
feature learned from a single network are biased. In the context of drug
prediction, many related studies have introduced diverse databases,
including drug chemical structures, target proteins, Anatomical Thera­
peutic Chemistry (ATC) codes, genes, diseases, etc [13,14]. These data
not only comprehensively characterize drugs, but also indirectly
describe the similarity between drugs. In the molecular structure of
drugs, chemical information (e.g. atoms, chemical bonds, and sub­
structures) can also be used as drug features, which are extracted from
drug SMILES [15] by tools such as openbabel [16], CDK [17], RDKit

[18]. In biomedical literature mining, biomedical ontologies (genes,
proteins, drugs, etc.) have abundant semantic information. Natural
Language Processing (NLP)-based models use this information to create
word embeddings that can accurately express the relations between
ontologies [19].
In a realistic scenario, the drug side effect pairs confirmed by the
experiment are rare, most of them are unverified samples [20]. How­
ever, there may still be real positive samples in unlabeled samples.
Generally, similar drugs are more likely to share the same side effects,
and vary widely drugs are less likely to link the common side effects.
Based on this hypothesis, reasonable negative samples are screened out
of unlabeled samples using drug similarity measurement methods

[13,21,22].
In recent years, deep learning technology has demonstrated a greater
learning capability than traditional machine learning. Dey et al. [23]
built a deep learning framework with an attention mechanism to iden­
tify the relations between drug molecular substructures and specific side
effects, and test whether the substructures help predict the adverse ef­
fects of new drugs. Neural Collaborative Filtering (NCF) [24] cancels the
simple linear inner product operation of the traditional MF method, and
designs a neural network-based collaborative filtering method to cap­
ture the complex structure in the network data. Knowledge Graph (KG)
is a rich biomedical information library constructed from multi-source
data sets, which can effectively represent entity relations. In the
figure, nodes represent different entities, such as drugs, target proteins,
diseases, genes, side effects, and pathways [25,26]. The KG embedding
method (such as ComplEx [27]) is employed to extract the dense vector



that characterizes the complicated relations of the nodes in the graph
and feed into the downstream prediction model [28]. Recently, Graph
Neural Network (GNN) [29] has also been increasingly applied to bio­
logical network problems, including node classification, link prediction

[30] and so on.
Considering the unicity of the above drug feature methods, we plan
to comprehensively adopt the internal molecular structure and the
external biological network structure of the drug. We propose a hybrid
GNN framework composed of graph embedding and node embedding,
idse-HE, to identify potential drug side effects. The hybrid embedding
strategy can learn the graph-level representation of drug molecules
(graph embedding) and update the node-level information (node
embedding) with neighbor aggregation in the drug side effect network.
Experimental results under the same test conditions demonstrate that
idse-HE achieves the best performance compared with other baseline
prediction models, supporting the original intention of model design
based on a hybrid embedding structure. Our work mainly revolves
around the following aspects:


 - We intend to organize the two GNN models of graph embedding and
node embedding in series to learn drug features. This structure
design combines chemical information in drug molecular structure
and entity relation information in the biological network to more
exhaustively characterize drugs.

 - We turn the drug side effect prediction task into a matrix completion
process. The learned feature matrices of the drug and the side effect
are deem as the implicit factors and are projected into the lowdimensional space through linear layer for matrix completion. The
reconstruction matrix is equivalent to the predicted relevance scores
between drugs and side effects.

 - We compare the performance of idse-HE with other state-of-art
methods and variants of our model in potential side effects predic­
tion. Furthermore, we also verify and analyze some of the predicted
results.



2


_L. Yu et al._ _Journal of Biomedical Informatics 131 (2022) 104098_



**2. Methods**


_2.1. Problem formulation_



Discovering the potential side effects of drugs is equivalent to link
prediction of a bipartite graph [21,31,32]. A simple example of a
bipartite graph is shown in Fig. 1. We formulate the bipartite graph _G_ =
( _D, S, E_ ), drug node set _D_ = { _d_ _i_ |1 ≤ _i_ ≤ _N_ _i_ } and side effect node set _S_ =
{ _s_ _j_ |1 ≤ _j_ ≤ _N_ _j_ } are linked by edge set _E_ = {( _d_ _i_ _, s_ _j_ )| _d_ _i_ ∈ _D, s_ _j_ ∈ _S_ }⊂ _D_ × _S_,

where _N_ _d_, _N_ _s_ is the total number of drugs and side effects respectively. In
addition, _G_ can be represented by a _N_ _d_ × _N_ _s_ adjacency matrix _Y_ =
{ _y_ _ij_ }, where _y_ _ij_ = 1 if edge ( _d_ _i_ _, s_ _j_ ) exist, otherwise _y_ _ij_ = 0. idse-HE



} are linked by edge set _E_ = {( _d_ _i_ _, s_ _j_



}, where _y_ _ij_ = 1 if edge ( _d_ _i_ _, s_ _j_



) exist, otherwise _y_ _ij_ = 0. idse-HE



hydrogens, formal charge, radical electrons, hybridization, aromatic,
hydrogens) and four chemical bond attributes (i.e. type, conjugated,
ring, stereo) to represent the information of nodes and edges in the
molecular graph. For representation learning, we design the framework
of MPNN (e.g. message-passing phase) + set2set (e.g. readout phase) to
extract the physicochemical information of drugs and export 617dimensional molecular features. The set2set module can address two

crucial problems in transitioning from node embedding sequence to
graph embedding. One is the uncertain sequence order. Taking advan­
tage of the sequence invariance of the LSTM [40] hidden state to the
input sequence, we can eliminate the influence of node order on the
output result. The other is the variable sequence length. For the uncer­
tainty of the number of atoms in the drug molecule, the pointer network

[41] can handle inputs with varying lengths. Thus, the drug molecular
structure embedding can be formulated as:


_C_ _m_ = [ _m_ 1 _, m_ 2 _, m_ 3 _,_ ⋯ _m_ 617 ] (2)


_2.5. Node embedding module_


For the initial feature, the drug representation involves the embed­
ding of the chemical structure _C_ _m_ and the fingerprint of molecular
substructure sequence _C_ _f_ . The side effect is represented by using 0–0.1
standardized stochastic vector _C_ _s_ . _T_ = { _m, f, s_ } stands for the above
three feature types, respectively. We construct a node embedding neural
network to explore the potential associations between drug nodes and
side effect nodes in the bipartite graph _G_ . In advance of learning, the raw
feature _C_ _t_ with type _t_ ∈ _T_ is transformed to the unified dimensional
vector _F_ _t_ through a feature-type associate single-layer fully connected
neural network. Specifically, a node embedding layer includes aggre­
gation and update operations. The node embedding can be aggregated
and updated several times by overlapping layers. Nevertheless, with the
deepening of the neural network, the information of a single node will
soon cover almost the entire graph _G_, which leads to node assimilation
and excessive smoothness [42]. The structure of bipartite graph _G_ that
relatively simple and narrow is more suitable for a shallow model.
In fact, we conceive a two-layer node embedding learning module,
which proved reasonable and effective in the robust experiments. In the
first layer, we input the drug molecular structure embedding _C_ _m_ and the
side effect feature _C_ _s_ and output the intermediate hidden state, and then
unite the drug molecule fingerprint _C_ _f_ are fed into the second layer
network to obtain the final embedding.


_2.5.1. Aggregating neighbors information_
Given a bipartite graph _G_, _N_ ( _v_ ) denotes neighbor set which link to
node _v_ . If _v_ represents a drug, _N_ ( _v_ ) represents all known side effects of
drug _v_, otherwise, _N_ ( _v_ ) represents drugs. Formally, the information
aggregation of _N_ ( _v_ ) is described as follows:



ultimately reconstruct matrix _Y_ [’] = { _y_ [’] _ij_



}, where _y_ [’] _ij_ [denotes the proba­]



bility of link between drug _d_ _i_ and side effect _s_ _j_ .


_2.2. Overview of ides-HE_


The whole components of idse-HE are illustrated in Fig. 1. The model
is divided into three stages, feature processing, embedding learning and
matrix reconstruction. In the feature processing module, we apply the
MPNN [33] with set2set output [34] and the discrete wavelet transform
(DWT) [35,36] to respectively extract the molecular structure features
and substructure sequence features of the drug and create standardized
stochastic vectors to stand for the side effect. In the embedding learning
process, we design a two-layer node embedding structure that updates
each node feature by aggregating representations of the node and
neighbors in the bipartite graph _G_ . In the matrix reconstruction stage,
the embeddings of the drug and the side effect are projected to low
dimension space and then multiplied to generate a new adjacency ma­
trix _Y_ [’] . Next, we introduce the algorithm formula of our framework in
detail.


_2.3. 2D molecular fingerprint representation_


The simplified molecular-input line entry system (SMILES) is a
specification in the form of a line notation for describing the structure of
chemical species using short ASCII strings [37]. We download the
SMILES file of the drug from _[https://go.drugbank.com/](https://go.drugbank.com/)_ and convert it into
the 256-digit hexadecimal string as the drug molecular fingerprint
through OpenBabel software ( _[http://openbabel.org/](http://openbabel.org/)_ ). Subsequently,
DWT is presented to transform molecular fingerprint and obtain two sets
of coefficients. One half is 128-dimensional approximation coefficient
representing valid information, and the other is detail coefficient as
invalid interference. Therefore, we select approximate coefficient as the
drug eigenvector, which can be formulated as:


_C_ _f_ = [ _f_ 1 _, f_ 2 _, f_ 3 _,_ ⋯ _, f_ 128 ] (1)


_2.4. Graph embedding representation_


According to the SMILES string of the drug, we employ the RDKit tool
to generate the drug molecular structure graph m, in which nodes
represent atoms and edges represent chemical bonds. Node information
comprises atomic attributes, including atom type, atomic number, atom
degree, electrons, hybridization, aromatic, etc. Edge information is
represented by chemical bond attributes: bond type, conjugated, ring,
etc. For example, MUFFIN [38] merely involved the number and
chirality of the atom and type and direction of the bond. DeepPurpose

[39] built a multi-layer neural network as the message passing stage to
transmit atom/edge level chemical descriptors among the atoms and
edges. Finally, it outputs the drug molecular feature through the readout
function (e.g. mean/sum).
Accordingly, we intend to use a graph neural network to abstract
chemical information of molecular structure graph. For chemical
descriptor, we adopt eight atomic attributes (i.e. type, degree, implicit



where _G_ _t_ ( _v_ ) ∈ R _[dim]_ [×][1], _h_ ( _e_ ) denotes weight of edge ( _v, u_ ), and _M_ ( _v_ ) =
∑ _u_ ∈ _N_ ( _v_ ) _,e_ =( _v,u_ ) _[h]_ [(] _[e]_ [)][ stands for a normalization term. ]


_2.5.2. Updating node embedding_
Given the aggregated neighbor information _G_ _t_ ( _v_ ) of all nodes _v_ . The
update process can be formulated as follows:


_E_ ( _v_ ) = _LeakyReLU_ ( _FC_ ( _CONCAT_ ( _C_ _t_ ( _v_ ) _, G_ _t_ ’ ( _v_ ) ) ) ) _, t_ ! = _t_ [’] _._ (4)

The final embedding _E_ ( _v_ ) of node _v_ is obtained by transforming the
concatenation of the original representation _C_ _t_ ( _v_ ) and the aggregated
information _G_ _t_ ’ ( _v_ ) through a single-layer neural network _FC_, and a
nonlinear activation function _LeakyReLU_ [43] in turn. Note that if node _v_
is a drug, _C_ _t_ ( _v_ ) denotes the splice of _C_ _f_ ( _v_ ) and _C_ _m_ ( _v_ ), _G_ _t_ ’ ( _v_ ) represents
eigenvectors of side effects corresponding to drug _v_ (i.e. _t_ [’] = _s_ ).



_G_ _t_ ( _v_ ) = ∑

_u_ ∈ _N_ ( _v_ ) _,e_ =( _v,u_ )∈ _E_



_h_ ( _e_ )
(3)
_M_ ( _v_ ) ~~_[F]_~~ _[t]_ [(] _[u]_ [)] _[,]_



3


_L. Yu et al._ _Journal of Biomedical Informatics 131 (2022) 104098_


**3. Data and experiment**


_3.1. Datasets_


The datasets used in our work are curated in a previous study [45],
including 1020 drugs, 5599 side effects and 133,750 positive samples
indicating that known drug side effect pairs. In other words, positive
samples account for 2.342% of the entire sample set. The dataset with
the quantity of 5,710,980 is extremely skewed. These data are collected
from DrugBank( _[https://go.drugbank.com](https://go.drugbank.com)_ ), SIDER( _[http://sideeffects.embl.](http://sideeffects.embl.de/)_
_[de/](http://sideeffects.embl.de/)_ ), and PubChem( _[https://pubchem.ncbi.nlm.nih.gov/](https://pubchem.ncbi.nlm.nih.gov/)_ ). Fig. 2 demon­
strates the distribution of real drug-side effect pairs, 1682 side effects are
caused by only one drug, 4 drugs induce a single side effect. 1007 of the
1020 drugs have SMILES or MOL files, which can calculate drug
information.


_3.2. Evaluation standard_


In this work, we adopt the metrics commonly appeared in unbal­
anced binary-class problems to assess the performance of classifiers,
including AUPR, MRR, F1 and MCC, which are defined as follows:



**Fig. 2.** The statistics of positive samples in the dataset. The lower left shows the
distribution of known drug side effect pairs, the upper indicates the number of
drugs for the side effects, and the right represents the number of side effects
caused by the drugs.


_2.6. Generalized matrix factorization_


Reconstruct the matrix _Y_ [’ ] and minimize the difference with the

original matrix _Y_ can be written as follows:


_Y_ [’] = _E_ _d_ ( _v_ ) × _L_ _d_ × _R_ _[T]_ _s_ [×] _[ E]_ _[S]_ [(] _[u]_ [)] _[T]_ (5)


where _L_ _d_ _, R_ _s_ ∈ R _[dim]_ [×] _[hid ]_ are projection matrics. × represents matrix
multiplication and _T_ represents matrix transposition.
Similar reconstruction matrix strategy [24,44] has also been utilized
to predict link probability. To be more specific, drug embedding _E_ _d_ and
side effect embedding _E_ _s_ are respectively linear converted into two in­
termediate matrices by _L_ _d_ and _R_ _s_, and then row-wise inner product to
reconstruct the new matrix _Y_ [’] . In the end, the elements in the recon­
struction matrix _Y_ [’ ] represent the relevance scores between drugs and
side effects.




 - AUPR: Area Under Precision-Recall (AUPR) curve.

 - MRR: Mean reciprocal rank. The reciprocal mean of the predicted
value rankings for all positive samples. For the convenience of dis­
playing in the same coordinate system as other metrics, we scale it up
by a factor of _N/_ 10 (N = 133750) to be between 0 and 1.

 - F1: The F1 score can be interpreted as a harmonic mean of the pre­


’ ’
cision and recall, where the average parameter is set to binary .

 MCC: Matthews correlation coefficient.


_3.3. Experiment setup_


In the experiment, we prepare 617-dimensional drug molecular
structure embedding, 128-dimensional drug molecular fingerprint and
1024-dimensional side effect eigenvector as original features. The hid­
den size ( _dim_ in formula 3–5) of the node embedding layer is set to 1024.
For matrix reconstruction, the dimension of the intermediate transition
matrix is set to 512 ( _hid_ in formula 5). idse-HE can be trained end-to-end
by performing _Adam_ [46] optimizer and 0.001 learning rate to minimize
the cross-entrop _y_ loss of all edges between matrix _Y_ and _Y_ [’] . All inde­
pendent experiment results are expressed as mean ± standard deviation
based on 10-fold cross-validation. We randomly divide 5% training data
as the validation set. The model iterates training until the epoch for
which the experimental indicators in the validation set stabilized ex­
ceeds the patient value which is set to 500. The entire framework is
implemented on the PyTorch platform and GPU hardware.



**Fig. 3.** The performance of idse-HE under different hyperparameters setting.


4


_L. Yu et al._ _Journal of Biomedical Informatics 131 (2022) 104098_


**Fig. 4.** The results of baselines and idse-HE with four metrics.



_3.4. Searching for optimal L2 coefficient and dropout_


In this section, we survey three hyperparameters settings that affect
the performance of idse-HE, including _l_ 2 coefficient, dropout, and
dimension. _l_ _l_ 2 loss item in the loss
2 coefficient is the weight of the
function and is used to limit the range of model parameters and avoid
_l_
the occurrence of overfitting. Similar to the 2 coefficient, the purpose of
dropout is to balance training level and test results to achieve maximum
generalization ability.
To select the best hyperparameters, we execute grid search on the
validation set. Fig. 3a demonstrates the variation trend of all metrics by
varying the abscissa _l_ 2 coefficient from 0.0005 to 0.05. We find that the
green broken line reaches the best value on F1 at 0.001, the other
metrics are highest at 0.005. Fig. 3b indicates the impact of dropout on
the performance in different set values. The model achieves the opti­
mum when dropout is 0.1. Fig. 3c shows the results of idse-HE when the
dimension is 256, 512, 1024 and 2048, respectively. We observe that
1024 is the best point on all metrics. Hence, we conclude that our model
sets 0.005 as _l_
2 coefficient, 0.1 as dropout and 1024 as embedding
dimension.



_3.5. Performance analysis and comparison_


For the sake of evaluating the outperformance of idse-HE, we
compare our model with the following state-of-art models:


 - MNMF [45]: MNMF reconstructs the adjacency matrix of drugs and
side effects using the NMF method. The matrix scores are marked as
the weights to perform heat diffusion in the drug-drug similarity
network, and the final diffusion scores are considered the predicted
impact of drugs on the corresponding side effects.

 - MSVD: MSVD is an improved method of MNMF, which replaces NMF
with TruncatedSVD.

 - GMF [24]: Generalized Matrix Factorization can be interpreted as the
neural network form of matrix factorization. GMF performs the
element-wise product for the potential feature vectors and outputs
the result through a fully connected layer. We use word embedding
as the latent representation.


Fig. 4 shows the results of idse-HE and the upper baselines. By
comparison, idse-HE remarkable outperforms the suboptimal method by
0.14% on AUPR, 4.96% on F1, and 3.43% on MCC. These data
demonstrate that our model achieves the best performance compared
with all baselines. Specifically, MNMF and MSVD are machine learning



5


_L. Yu et al._ _Journal of Biomedical Informatics 131 (2022) 104098_


**Table 1**

Results of idse-HE and the variants.


Method (mean AUPR MRR F1 MCC

± SD)


idse-HE-1 l 0.6472 ± 0.9438 ± 0.5471 ± 0.5522 ±

0.0027 0.0033 0.0054 0.0063

idse-HE-3 l **0.6569 ±** 0.8237 ± 0.5521 ± 0.5643 ±

**0.0135** 0.0182 0.0168 0.0177

idse-HE-MS 0.6442 ± 0.9222 ± 0.5561 ± 0.5652 ±

0.0163 0.0401 0.0146 0.0124

idse-HE-FP 0.6562 ± 0.9306 ± 0.5563 ± 0.5669 ±

0.0030 0.0171 0.0092 0.0075

idse-HE 0.6545 ± **0.9464 ±** **0.5574 ±** **0.5680 ±**

0.0062 **0.0038** **0.0084** **0.0063**


**Fig. 6.** The top 30 potential drug side effects sorted by predicted score in the
negative samples. The blue nodes on the left represent the drugs, the orange
nodes on the right represent the side effects, the gray edges represent the
verified pairs, and the red edges represent the unproved.


cause the model to be under-learned or over-smooth. For idse-HE-MS

and idse-HE-FP using a single feature, idse-HE shows relatively
obvious advantages on MRR. This reveals the effectiveness of fusion
features that combine molecular structure and molecular fingerprint.
Overall, idse-HE has the best performance and is close to other var­
iants, indicating that our model design is greatly reasonable and robust,
whether the model structure or feature selection.


_3.7. Visualization analysis of link representation_



**Fig. 5.** Low-dimensional visualization of link representations through UMAP.


methods. GMF is a neural network model with better learning ability
and feature representation. However, these baselines ignore the chem­
ical structure and physicochemical properties of the drug.


_3.6. Variant study_


To reveal the robustness and rationality of our model, we design four
different variants from the perspective of feature selection and model
structure. The specific details of each variant are described as follow:


 - **idse-HE-1l** : idse-HE dismantles the second-level node embedding
layer.

 - **idse-HE-3l** : idse-HE splices a node embedding layer.

 - **idse-HE-MS** : idse-HE simply utilizes molecular structure graph
embedding learned from MPNN as drug representation.

 - **idse-HE-FP**
: idse-HE only uses molecular fingerprint obtained by
wavelet transform as drug feature.


All variants and idse-HE are executed under the same experimental
conditions. The contrasts are shown in Table 1, and the optimum of
metrics is bold. idse-HE-1l performs worst than other structural variants
with more embedding layers. idse-HE-3l is 0.24% higher than idse-HE in
terms of AUPR but is lower on F1 and MCC. These analyses indicate that
the two-layer embedding learning can best discover the potential in­
formation of bipartite graph G, and the one-layer or the three-layer may



In this section, we visualize and analyze the link representations
learned in an end-to-end manner to evaluate the performance of idseHE. Taking the sample size and visualization effect into consideration,
we select all positive samples and equivalent random negative samples.
To be more specific, the representation of the link between drugs and
side effects is formed by element-wise addition of embeddings of drugs
and side effects. Finally, the link representations are projected and
visualized in the 2D space by uniform manifold approximation and
projection (UMAP) [47]. As shown in Fig. 5, gray stands for known drug
side effects, red indicates unknown, most of the scatters are concen­
trated in two clusters, especially, the negative samples located in the
lower region of the figure are highly dense. Visualization results show
that idse-HE can effectively learn embedded representations of drugs
and side effects, and accurately identify drug side effects.


_3.8. Case study_


This section aims to test the real performance of idse-HE. In the
predicted results, we analyze the top 30 potential drug side effects sorted
in descending of predicted scores in the negative samples and illustrate
in Fig. 6. We search for supporting evidence from drugs.com, ADReCS

[48] database and other databases or literature libraries for these pre­
dicted results. In the top10 drug side effect pairs, we verify 6 of the pairs.
14 of the top 20 pairs, and 19 of the top 30 pairs are also confirmed. In
Fig. 6, the blue nodes on the left represent the drugs, the orange nodes on
the right represent the side effects, the gray edges represent the verified
pairs, and the red edges represent the unproved.
In addition, we find three drugs related to COVID-19 in the dataset,
including Ritonavir, Chloroquine, and Ivermectin. These drugs were
approved by the American Food and Drug Administration (FDA) to treat



6


_L. Yu et al._ _Journal of Biomedical Informatics 131 (2022) 104098_


**Fig. 7.** Predictive analytics on ’dissocial’ scenarios. (a) The top 30 drugs in ascending of the number of caused side effects. (b) The top 30 side effects in ascending of
the number of associated drugs. The blue line represents the prediction accuracy in known positive samples, and the green line represents the complete (including
known positive samples and confirmed potential positive samples) accuracy in the predicted positive samples. The x-axis represents the index of nodes (starting at 0)
and the degree (i.e. number of associated drugs or side effects) of nodes in the bipartite graph _G._



certain diseases and repurpose for the treatment of COVID-19. Among
them, Ritonavir and Ivermectin are in clinical trials. Chloroquine was
urgently authorized use by the FDA in March 2020. After investigation,
294 of 617 predicted side effects of Ritonavir are correctly identified. 45
positive samples are confirmed in the 150 potential side effects of
Ivermectin. For Chloroquine, 56 hits from 173 side effects.
In positive samples, many drugs or side effects with only one or few
associated partners, we call such drugs and side effects as ’dissocial’. To
investigate these scenarios, we analyze the prediction results of the top
30 drugs and side effects, respectively, according to the ascending order
of the connection degree of drug nodes and side effect nodes in the
bipartite graph _G_ . In Fig. 7a, the hit rate of drugs that caused only one or
two side effects is initially almost 0. But as the number of associated side
effects increased, the prediction effect improved significantly, and the
accuracy of the last 10 drugs basically stabilized around 0.4. Similarly,
in the side effect scenario, as the number of drugs associated with the
side effect increases, the prediction accuracy improves.
The above evidence reflects the ability of idse-HE to identify drug
side effects and has a certain reference value for the research of drugs to
treat the new coronavirus.


**4. Conclusion**


In this study, we propose a novel drug side effect prediction frame­
work, called idse-HE, a graph neural network using a hybrid embedding
strategy. This architecture can fully extract medicinal chemistry infor­
mation and drug entity associations respectively from the molecular
structure graph of the drug and the biological entity network where the
drug is located. Compared with other advanced methods under the same
conditions, the predicted result of our model has improved significantly.
According to the variant study, we demonstrate that idse-HE is cor­
rectness in structural design and stability in performance, and the hybrid
embedding can effectively fuse the characteristics of the drug. In the
end, we have verified several real cases in the predicted results by
searching other databases or literature.
In our work, the scope of our data sources is still narrow. In future
work, we consider introducing heterogeneous data sources to expand
the drug biology network, including target proteins, substructures,
pathways, therapeutic markers, and leverage KG embedding methods to
extract drug features [49]. Moreover, we may also test drug similarity



calculation methods, such as fingerprint Tanimoto coefficient, atc code,
text mining, STITCH, KEGG, and other databases to form a denser drug
network [22,50,51]. In addition, we will also try polypharmacy side
effect prediction in the case of combined use of multiple drugs [52,53].


**Author contributions**


Yu and Qiu conceived and formulated the research plan. Yu and
Cheng implemented the extraction of features, model construction and
experiments. Yu and Qiu drafted and revised the manuscript. Xiao and
Lin supervised the project. All authors contributed to the content of this
paper, and approved the final manuscript.


**Declaration of Competing Interest**


The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.


_Acknowledgements_


Not applicable.


_Funding_


This work was supported by the grants from the National Natural
Science Foundation of China (No. 31860312, 62062043, 62162032),
Natural Science Foundation of Jiangxi Province, China (NO.
20202BAB202007, 20171ACB20023), the Department of Education of
Jiangxi Province (GJJ211349, GJJ180703), the International Coopera­
tion Project of the Ministry of Science and Technology, China (NO.
2018-3-3), Science and Technology Plan Project of Jingdezhen City,
China (20212GYZD009-01).


_Availability of data and materials_


The dataset, code and materials used in this project can be freely
available at: _http://bioinfo.jcu.edu.cn/idse-HE, https://github.com/yuliyi/_
_idse-HE._



7


_L. Yu et al._ _Journal of Biomedical Informatics 131 (2022) 104098_



**References**


[[1] I.R. Edwards, J.K. Aronson, Adverse drug reactions: definitions, diagnosis, and](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0005)
[management, The Lancet 356 (9237) (2000) 1255–1259.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0005)

[[2] S.K. Sahu, A. Anand, Drug-drug interaction extraction from biomedical texts using](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0010)
[long short-term memory network, J. Biomed. Inform. 86 (2018) 15–24.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0010)

[[3] S. Vilar, R. Harpaz, E. Uriarte, L. Santana, R. Rabadan, C. Friedman, Drug—drug](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0015)
[interaction through molecular structure similarity analysis, J. Am. Med. Inform.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0015)
[Assoc. 19 (6) (2012) 1066–1074.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0015)

[[4] S. Whitebread, J. Hamon, D. Bojanic, L. Urban, Keynote review: In vitro safety](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0020)
[pharmacology profiling: an essential tool for successful drug development, Drug](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0020)
[Discovery Today 10 (21) (2005) 1421–1433.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0020)

[[5] J. Li, S. Zheng, B. Chen, A.J. Butte, S.J. Swamidass, Z. Lu, A survey of current](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0025)
[trends in computational drug repositioning, Briefings Bioinf. 17 (1) (2016) 2–12.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0025)

[[6] Y. Yamanishi, E. Pauwels, M. Kotera, Drug Side-Effect Prediction Based on the](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0030)
[Integration of Chemical and Biological Spaces, J. Chem. Inf. Model. 52 (12) (2012)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0030)
[3284–3292.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0030)

[[7] J. Scheiber, B. Chen, M. Milik, S.C.K. Sukuru, A. Bender, D. Mikhailov,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0035)
[S. Whitebread, J. Hamon, K. Azzaoui, L. Urban, M. Glick, J.W. Davies, J.L. Jenkins,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0035)
[Gaining Insight into Off-Target Mediated Effects of Drug Candidates with a](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0035)
[Comprehensive Systems Chemical Biology Analysis, J. Chem. Inf. Model. 49 (2)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0035)
[(2009) 308–317.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0035)

[[8] Y. Pouliot, A.P. Chiang, A.J. Butte, Predicting adverse drug reactions using publicly](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0040)
[available PubChem BioAssay data, Clin. Pharmacol. Therap. 90 (1) (2011 Jul)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0040)
[90–99.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0040)

[9] E. Munoz, V. Nov˜ [´aˇcek, P.-Y. Vandenbussche, Using drug similarities for discovery](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0045)
[of possible adverse reactions, AMIA Annu. Symp. Proc. 2016 (2017) 924–933.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0045)

[[10] B. Perozzi, R. Al-Rfou, S. Skiena, DeepWalk: online learning of social](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0050)
[representations, in: Proceedings of the 20th ACM SIGKDD international conference](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0050)
[on Knowledge discovery and data mining, Association for Computing Machinery:](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0050)
[New York, New York, USA, 2014, pp. 701–710.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0050)

[[11] A. Grover, J. Leskovec, node2vec: Scalable Feature Learning for Networks, in:](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0055)
[Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0055)
[Discovery and Data Mining, Association for Computing Machinery, San Francisco,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0055)
[California, USA, 2016, pp. 855–864.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0055)

[[12] B. Chen, F. Li, S. Chen, R. Hu, L. Chen, Link prediction based on non-negative](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0060)
[matrix factorization, PLoS One 12 (8) (2017), e0182968.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0060)

[[13] Y. Zheng, H. Peng, S. Ghosh, C. Lan, J. Li, Inverse similarity and reliable negative](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0065)
[samples for drug side-effect prediction, BMC Bioinf. 19 (13) (2019) 554.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0065)

[[14] Y. Luo, X. Zhao, J. Zhou, J. Yang, Y. Zhang, W. Kuang, J. Peng, L. Chen, J. Zeng,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0070)
[A network integration approach for drug-target interaction prediction and](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0070)
[computational drug repositioning from heterogeneous information, Nat. Commun.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0070)
[8 (1) (2017) 573.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0070)

[[15] D. Weininger, SMILES, a chemical language and information system. 1.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0075)
[Introduction to methodology and encoding rules, J. Chem. Inf. Comput. Sci. 28 (1)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0075)
[(1988) 31–36.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0075)

[[16] N.M. O’Boyle, M. Banck, C.A. James, C. Morley, T. Vandermeersch, G.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0080)
[R. Hutchison, Open Babel: An open chemical toolbox, J. Cheminf. 3 (1) (2011) 33.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0080)

[[17] E.L. Willighagen, J.W. Mayfield, J. Alvarsson, A. Berg, L. Carlsson, N. Jeliazkova, S. Kuhn, T. Pluskal, M. Rojas-Cherto, O. Spjuth, G. Torrance, C.T. Evelo, R. Guha, ´](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0085)
[C. Steinbeck, The Chemistry Development Kit (CDK) v2.0: atom typing, depiction,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0085)
[molecular formulas, and substructure searching, J. Cheminf. 9 (1) (2017) 33.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0085)

[[18] G. Landrum, RDKit: Open-Source Cheminformatics and Machine Learning. htt](https://www.rdkit.org/)
[ps://www.rdkit.org/.](https://www.rdkit.org/)

[[19] L.D. Vine, G. Zuccon, B. Koopman, L. Sitbon, P. Bruza, Medical semantic similarity](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0095)
[with a neural language model, in: Proceedings of the 23rd ACM International](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0095)
[Conference on Conference on Information and Knowledge Management,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0095)
[Association for Computing Machinery, Shanghai, China, 2014, pp. 1819–1822.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0095)

[[20] J. Liu, S. Zhao, X. Zhang, An ensemble method for extracting adverse drug events](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0100)
[from social media, Artif. Intell. Med. 70 (2016) 62–76.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0100)

[[21] H. Eslami Manoochehri, M. Nourani, Drug-target interaction prediction using semi-](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0105)
[bipartite graph model and deep learning, BMC Bioinf. 21 (4) (2020) 248.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0105)

[[22] H. Liang, L. Chen, X. Zhao, X. Zhang, Prediction of Drug Side Effects with a Refined](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0110)
[Negative Sample Selection Strategy, Comput. Math. Methods Med. 2020 (2020)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0110)
[1573543.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0110)

[[23] S. Dey, H. Luo, A. Fokoue, J. Hu, P. Zhang, Predicting adverse drug reactions](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0115)
[through interpretable deep learning framework, BMC Bioinf. 19 (21) (2018) 476.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0115)

[[24] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, T.-S. Chua, Neural collaborative filtering, in:](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0120)
[Proceedings of the 26th International Conference on World Wide Web,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0120)
[International World Wide Web Conferences Steering Committee, Perth, Australia,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0120)
[2017, pp. 173–182.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0120)

[[25] R. Celebi, H. Uyar, E. Yasar, O. Gumus, O. Dikenelli, M. Dumontier, Evaluation of](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0125)
[knowledge graph embedding approaches for drug-drug interaction prediction in](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0125)
[realistic settings, BMC Bioinf. 20 (1) (2019) 726.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0125)




[26] M. Wang, Predicting rich drug-drug interactions via biomedical knowledge graphs

[[27] T. Thand text jointly embedding. arXiv preprint arXiv:1712.08875, ´eo, W. Johannes, R. Sebastian, G. Eric, B. Guillaume, Complex Embeddings](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0135) **2017** .

[[28] E. Mufor Simple Link Prediction, PMLR 48 (2016) 2071noz, V. Nov˜](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0135) [´aˇcek, P.-Y. Vandenbussche, Facilitating prediction of adverse –2080.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0135)
[drug reactions by using knowledge graphs and multi-label learning models,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0140)
[Briefings Bioinf. 20 (1) (2019) 190–202.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0140)

[29] T.N. Kipf, M. Welling, Semi-supervised classification with graph convolutional
networks. arXiv preprint arXiv:1609.02907, **2016** .

[[30] M. Zhang, Y. Chen, Link prediction based on graph neural networks. 32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montr´eal,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0150)
[Canada, 2018.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0150)

[[31] Y. Zheng, W. Zhao, C. Sun, Q. Li, Drug side-effect prediction using heterogeneous](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0155)
[features and bipartite local models, Comput. Mater. Continua 60 (2) (2019).](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0155)

[[32] K. Bleakley, Y. Yamanishi, Supervised prediction of drug-target interactions using](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0160)
[bipartite local models, Bioinformatics (Oxford, England) 25 (18) (2009)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0160)
[2397–2403.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0160)

[[33] J. Gilmer, S.S. Schoenholz, P.F. Riley, O. Vinyals, G.E. Dahl, Neural message](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0165)
[passing for quantum chemistry, in: P. Doina, T. Yee Whye (Eds.), Proceedings of](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0165)
[the 34th International Conference on Machine Learning, PMLR: Proceedings of](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0165)
[Machine Learning Research, 2017, pp. 1263–1272.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0165)

[34] O. Vinyals, S. Bengio, M. Kudlur, Order matters: Sequence to sequence for sets,
arXiv preprint arXiv:1511.06391, 2015.

[35] W. Qiu, Z. Lv, Y. Hong, J. Jia, X. Xiao, BOW-GBDT: A GBDT Classifier Combining
With Artificial Neural Network for Identifying GPCR–Drug Interaction Based on
Wordbook Learning From Sequences, Front. Cell Devel. Biol., 2021, 8 (1789).

[[36] J. Hu, Y. Li, J.-Y. Yang, H.-B. Shen, D.-J. Yu, GPCR–drug interactions prediction](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0180)
[using random forest with drug-association-matrix-based post-processing](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0180)
[procedure, Comput. Biol. Chem. 60 (2016) 59–71.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0180)

[[37] A. Toropov, A. Toropova, D.V. Mukhamedzhanova, I. Gutman, Simplified](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0185)
[molecular input line entry system (SMILES) as an alternative for constructing](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0185)
[quantitative structure-property relationships (QSPR), Indian J. Chem. – Sect. A](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0185)
[Inorg. Phys. Theore. Anal. Chem. 44 (2005) 1545–1552.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0185)

[[38] Y. Chen, T. Ma, X. Yang, J. Wang, B. Song, X. Zeng, MUFFIN: multi-scale feature](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0190)
[fusion for drug–drug interaction prediction, Bioinformatics (Oxford, England) 37](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0190)
[(17) (2021) 2651–2658.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0190)

[[39] K. Huang, T. Fu, L.M. Glass, M. Zitnik, C. Xiao, J. Sun, DeepPurpose: a deep](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0195)
[learning library for drug–target interaction prediction, Bioinformatics (Oxford,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0195)
[England) 36 (22–23) (2020) 5545–5547.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0195)

[[40] S. Hochreiter, J. Schmidhuber, Long Short-Term Memory, Neural Comput. 9 (8)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0200)
[(1997) 1735–1780.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0200)

[41] O. Vinyals, M. Fortunato, N. Jaitly, Pointer networks, arXiv preprint arXiv:
1506.03134 **2015** .

[42] M. Henaff, J. Bruna, Y. LeCun, Deep convolutional networks on graph-structured
data, arXiv preprint arXiv:1506.05163 2015.

[[43] A.L. Maas, A.Y. Hannun, A.Y. Ng, Rectifier nonlinearities improve neural network](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0215)
[acoustic models, Proc. icml 30 (2013).](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0215)

[[44] F. Wan, L. Hong, A. Xiao, T. Jiang, J. Zeng, NeoDTI: neural integration of neighbor](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0220)
[information from a heterogeneous network for discovering new drug–target](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0220)
[interactions, Bioinformatics (Oxford, England) 35 (1) (2019) 104–111.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0220)

[[45] M. Timilsina, M. Tandan, M. d’Aquin, H. Yang, Discovering Links Between Side](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0225)
[Effects and Drugs Using a Diffusion Based Method, Sci. Rep. 9 (1) (2019) 10436.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0225)

[[46] D. Kingma, J. Ba, Adam: a method for stochastic optimization. International](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0230)
[Conference on Learning Representations, 2014.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0230)

[[47] L. McInnes, J. Healy, N. Saul, L. Grossberger, UMAP: Uniform Manifold](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0235)
[Approximation and Projection, J. Open Source Softw. 3 (2018) 861.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0235)

[[48] M.-C. Cai, Q. Xu, Y.-J. Pan, W. Pan, N. Ji, Y.-B. Li, H.-J. Jin, K. Liu, Z.-L. Ji,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0240)
[ADReCS: an ontology database for aiding standardization and hierarchical](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0240)
[classification of adverse drug reaction terms, Nucl. Acids Res. 43 (Database issue)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0240)
[(2015) D907–D913.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0240)

[49] X. Lin, Z. Quan, Z.-J. Wang, T. Ma, X. Zeng, in: KGNN: Knowledge Graph Neural
Network for Drug-Drug Interaction Prediction, IJCAI, pp. 2739–2745, 2020.

[[50] M. Liu, Y. Wu, Y. Chen, J. Sun, Z. Zhao, X.W. Chen, M.E. Matheny, H. Xu, Large-](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0250)
[scale prediction of adverse drug reactions using chemical, biological, and](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0250)
[phenotypic properties of drugs, J. Am. Med. Informat. Associat.: JAMIA 19 (e1)](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0250)
[(2012) e28–35.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0250)

[[51] M. Kuhn, D. Szklarczyk, S. Pletscher-Frankild, T. Blicher, C. von Mering, L. Jensen,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0255)
[P. Bork, STITCH 4: Integration of protein-chemical interactions with user data,](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0255)
[Nucleic Acids Res. 42 (2013).](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0255)

[[52] C. Lee, Y.-P.-P. Chen, Prediction of drug adverse events using deep learning in](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0260)
[pharmaceutical discovery, Briefings Bioinf. 22 (2020).](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0260)

[[53] H. Xu, S. Sang, H. Lu, Tri-graph Information Propagation for Polypharmacy Side](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0265)
[Effect Prediction.. 33rd Conference on Neural Information Processing Systems](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0265)
[(NeurIPS 2019), Vancouver, Canada, 2020.](http://refhub.elsevier.com/S1532-0464(22)00114-9/h0265)



8


