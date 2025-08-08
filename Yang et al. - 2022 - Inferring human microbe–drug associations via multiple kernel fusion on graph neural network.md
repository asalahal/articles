[Knowledge-Based Systems 238 (2022) 107888](https://doi.org/10.1016/j.knosys.2021.107888)


[Contents lists available at ScienceDirect](http://www.elsevier.com/locate/knosys)

# Knowledge-Based Systems


[journal homepage: www.elsevier.com/locate/knosys](http://www.elsevier.com/locate/knosys)

# Inferring human microbe–drug associations via multiple kernel fusion on graph neural network


Hongpeng Yang [a] [,] [d], Yijie Ding [b] [,] [∗], Jijun Tang [c] [,] [d], Fei Guo [a] [,] [∗]


a _School of Computer Science and Engineering, Central South University, Changsha 410083, China_
b _Yangtze Delta Region Institute (Quzhou), University of Electronic Science and Technology of China, Quzhou, 324000, PR China_
c _Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen, 518005, China_
d _School of Computer Science and Technology, College of Intelligence and Computing, Tianjin University, Tianjin, 300350, China_



a r t i c l e i n f o


_Article history:_
Received 21 September 2021
Received in revised form 12 November 2021
Accepted 2 December 2021
Available online 10 December 2021


[Dataset link: https://github.com/guofei-tju/](https://github.com/guofei-tju/MKGCN)
[MKGCN](https://github.com/guofei-tju/MKGCN)


_MSC:_

00-01

99-00


_Keywords:_
Microbe–drug association
Graph convolutional network
Multiple kernel fusion
Dual graph regularized least square
Bipartite network


**1. Introduction**



a b s t r a c t


Complex and diverse microbial communities have certain impacts on human health, and specific drugs
are needed to treat diseases caused by microbes. However, most of the discovery of associations
between microbes and drugs is through biological experiments, which are time-consuming and
expensive. Therefore, it is crucial to develop an effective and computational model to detect novel
microbe–drug associations. In this study, we propose a model based on Multiple Kernel fusion on
Graph Convolutional Network, called MKGCN, for inferring novel microbe–drug associations. Our model
is built on the heterogeneous network of microbes and drugs to extract multi-layer features, through
Graph Convolutional Network (GCN). Then, we respectively calculate the kernel matrix by embedding
features on each layer, and fuse multiple kernel matrices based on the average weighting method.
Finally, Dual Laplacian Regularized Least Squares is used to infer new microbe–drug associations by
the combined kernel in microbe and drug spaces. Compared with the existing tools for detecting
biological bipartite networks, our model has excellent prediction effect on three datasets via three
types of cross-validation. Furthermore, we also conduct a case study of the SARS-Cov-2 virus and
make a deduction about drugs that may be able to associate with COVID-19. We have proved the
accuracy of the prediction results through the existing literature.
© 2021 Published by Elsevier B.V.



Human microbiome is a complex and diverse community,
which has an important impact on human health. The community contains trillions of microorganisms like bacteria, archaea,
viruses, as well as microbial eukaryotes such as fungi, protozoa, and helminths, and affects overall health and homeostasis
by actively participating in human metabolism and regulating
the immune system [1]. A balanced microbiota has shown to
play an important role in health sustenance. However, an alteration in the makeup of human microbiota (dysbiosis) can lead to
life-threatening illnesses [2]. The microbe–drug association has
attracted more and more attention. Understanding how human
microorganisms affect the efficacy and side effects of drugs has
a certain role in promoting drug development. Whereas, most of
the detection of associations between microorganisms and drugs
is through biological experiments, which is very time-consuming
and expensive. Predicting potential microbe and drug associations through computational methods can be used as an auxiliary


∗ Corresponding authors.
_E-mail addresses:_ [wuxi_dyj@csj.uestc.edu.cn (Y. Ding), guofei@csu.edu.cn](mailto:wuxi_dyj@csj.uestc.edu.cn)
(F. Guo).


[https://doi.org/10.1016/j.knosys.2021.107888](https://doi.org/10.1016/j.knosys.2021.107888)
0950-7051/ © 2021 Published by Elsevier B.V.



method for biological experiments, thereby improving the efficiency of drug development and understanding the interaction
between microorganisms and drugs more efficiently.

Currently, models specifically designed for predicting
microbe–drug associations are still relatively rare. For example,
Long et al. [3] proposed a framework GCNMDA based on Graph
Convolutional Network (GCN) [4] for the microbe–drug association prediction, which applied Conditional Random Field to
ensure that similar nodes have similar representations. It is worth
noting that microbe–drug associations detection is also a link
prediction problem in biological bipartite networks [5–7]. Therefore, many models used in biological bipartite networks can also
be applied to detect microbe and drug networks. For example,
drug–disease association prediction can reduce the overall drug
development cost and shorten the development time, and has
the advantage of using low-risk compounds [8,9]. Disease gene
prediction is the task of predicting the most likely candidate disease genes. It is an indispensable problem in biomedical research
and is of great significance for understanding the internal causes
of diseases [10,11]. Predicting associations between miRNAs and
diseases is beneficial to the diagnosis and treatment of complex
human diseases [12,13].


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_



Multiple Kernel Learning (MKL) [14] is a common method
to improve prediction ability in biological bipartite networks.
MKL first calculates the multi-kernel matrices through the multiple information of the sample. Then the optimal kernel matrix is obtained through multiple kernel fusion. For example,
Chen et al. [15] developed a model for miRNA–disease association prediction, called MKRMDA based on MKL and Kronecker
regularized least squares, which could automatically optimizing the combination of multiple kernels for disease and miRNA.
Ding et al. [16,17] proposed multiple information fusion models
to identify drug–target interactions and drug-side effect associations. Yan et al. [18] designed MKLC-BiRW to predict new
drug–target interactions by integrating diverse drug-related and
target-related heterogeneous information. MKL improves the predictive ability of the model through the combination of a variety
of information, which requires samples to have the characteristics
of a variety of information sources to construct different kernel
matrices. Therefore, MKL is not applicable to samples with fewer
types of features.

In this paper, we propose a new model Multiple Kernel fusion
on Graph Convolutional Network, called MKGCN, for predicting
microbe–drug associations. As we all know, Graph Convolutional
Network (GCN) is being widely used in various biological problems [3,19–21]. In a multi-layer GCN, the embedding features
obtained on each layer represent different structural information.
For example, features in the first layer represent the information
of directly connected nodes, while features on higher layer represent the aggregation of information between multi-hop neighbor
nodes. Here, we apply multi-layer GCN to extract a variety of
different structural information of nodes in the graph, which can
solve the problem of insufficient sample information sources on
multiple kernel fusion. Our model is based on the heterogeneous
network of microbes and drugs to extract multi-layer embedding features through GCN, calculate the kernel matrix by the
embedding features on each layer, and fuse multiple kernel matrices based on an average weighting method. Finally, Dual Graph
Regularized Least Squares (DLapRLS) [17] is used to predict new
microbe–drug associations by the combined kernel in microbe
and drug spaces. We compare our proposed model (MKGCN)
with other existing models designed to predict biological bipartite
network and our model has excellent prediction effects on three
datasets via three types of cross-validation. Furthermore, we also
conduct a case study about the SARS-Cov-2 virus and predict
drugs that may be able to associate with COVID-19, and then
the accuracy of prediction results has been proved through the
existing literature. This case study shows that our new model can
accurately discover new microbe–drug associations.


**2. Materials and methods**


_2.1. Problem definition_


Inferring novel associations in human microbe–drug network
can be regarded as a kind of biological bipartite network prediction. We can represent microbes and drugs as two different
types of nodes in the network. The node set of _N_ _d_ drugs is
defined as _D_ = [{] _d_ 1 _,_ _d_ 2 _, . . .,_ _d_ _N_ _d_ } . Similarly, describing the node
set of _N_ _m_ microbes as _M_ = [{] _m_ 1 _,_ _m_ 2 _, . . .,_ _m_ _N_ _m_ } . Edges in the
network are associations between microbes and drugs, which can
be expressed as an adjacency matrix **Y** ∈ **R** _[N]_ _[d]_ [×] _[N]_ _[m]_ . Here, **Y** _i_ _,_ _j_ = 1
means one microbe _m_ _j_ (1 ≤ _j_ ≤ _N_ _m_ ) is associated with one drug
_d_ _i_ (1 ≤ _i_ ≤ _N_ _d_ ). On the contrary, **Y** _i_ _,_ _j_ = 0 denotes the association
is unknown. Our task is to obtain a prediction matrix **F** [∗] of the
same size as the **Y** to predict the unknown associations. In this
study, our model Multiple Kernel fusion on Graph Convolutional
Network (MKGCN) is designed for inferring novel microbe–drug
associations. The flowchart of our proposed method is shown in
Fig. 1.



**Table 1**
Summary of the three datasets.


Datasets Microbes Drugs Associations


MDAD 173 1373 2470

aBiofilm 140 1720 2884

DrugVirus 95 175 933


_2.2. Datasets_


We use three well-known microbe–drug association datasets
that appeared in previous studies [3]. The first is MDAD dataset [1]

from the Sun et al. [22]. After removing redundancy information, the MDAD dataset includes 2470 known associations of
1373 drugs and 173 microbes for this study. The second dataset
is aBiofilm [2] [23], which stores resource of anti-biofilm agents
and their potential implications in antibiotic drug resistance. The
aBiofilm dataset has 23 unique anti-biofilm agent types including
1720 unique anti-biofilm agents(drugs) and 140 unique targeted
micro-organism. We finally select 2884 microbe–drug associations for our experiments. The last one is DrugVirus dataset [3]

which records the activity and development of related compounds of a variety of human viruses including the SARS-CoV-2.
Ultimately, we selected 933 related association information from
95 viruses and 175 drugs as our experimental data. Table 1
provides the detailed data size of the above three datasets. In our
research, we use these three datasets to evaluate the predictive
power of our model.


_2.3. Drug–drug similarity_


The drug similarity (kernel) is constructed by integrating drug
structural similarity and drug Gaussian kernel. The integrated
drug similarity **K** _[d]_ _s_ [between two drugs] [ {] _[d]_ _[i]_ _[,]_ _[ d]_ _[j]_ [} ∈] _[D]_ [ is defined as]
follows:

**K** _[d]_ _s_ [(] _[d]_ _[i]_ _[,]_ _[ d]_ _[j]_ [)] [ =] { _DS_ ( _GD_ _d_ _i_ _,_ _d_ _j_ ( ) _d_ + 2 _iGD_ _,_ _d_ ( _dj_ ) _i_ _,_ _,_ _d_ _j_ ) _,_ _if DSother_ ( _d_ _i_ _,w_ _d_ _j_ _ise_ ) ̸= 0 (1)


where _DS_ ( _d_ _i_ _,_ _d_ _j_ ) is drug structural similarity matrix, measured using SIMCOMP2 [24]; _GD_ ( _d_ _i_ _,_ _d_ _j_ ) is the Gaussian interaction profile
(GIP) kernel similarity between drugs, which is used to supplement the missing entries in structural similarity. Specifically,
_GD_ ( _d_ _i_ _,_ _d_ _j_ ) is calculated as follows:


_GD_ ( _d_ _i_ _,_ _d_ _j_ ) = exp( − _η_ _d_ ∥ _Y_ _d_ _i_ − _Y_ _d_ _j_ ∥ [2] ) (2)


where _Y_ _d_ _i_ represents the _i_ -row in the adjacency matrix as interaction profiles for drug _d_ _i_ ; _η_ _d_ represents the normalized kernel
bandwidth [25], defined as follows:



_η_ _d_ = _η_ _d_ [′] _[/]_ [(] [1]

_N_ _d_



_N_ _d_
∑ ∥ _Y_ _d_ _i_ ∥ [2] ) (3)


_i_ = 1



where _η_ _d_ [′] [is the original bandwidth.]


_2.4. Microbe–microbe similarity_


Similarly, the microbe similarity is constructed by integrating microbe functional similarity and microbe Gaussian kernel.
The integrated microbe similarity **K** _[m]_ _f_ between two microbes
{ _m_ _i_ _,_ _m_ _j_ } ∈ _M_ is defined as follows:


_FM_ ( _m_ _i_ _,_ _m_ _j_ ) + _GM_ ( _m_ _i_ _,_ _m_ _j_ )
**K** _[m]_ _f_ [(] _[m]_ _[i]_ _[,]_ _[ m]_ _[j]_ [)] [ =] { _GM_ ( _m_ 2 _i_ _,_ _m_ _j_ ) _,_ _,_ _if FMother_ ( _m_ _i_ _,w_ _mise_ _j_ ) ̸= 0 (4)


1 [http://www.chengroup.cumt.edu.cn/MDAD/.](http://www.chengroup.cumt.edu.cn/MDAD/)
2 [http://bioinfo.imtech.res.in/manojk/abiofilm/.](http://bioinfo.imtech.res.in/manojk/abiofilm/)
3 [https://drugvirus.info/.](https://drugvirus.info/)



2


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_


**Fig. 1.** The overview of our proposed method.



where _FM_ ( _m_ _i_ _,_ _m_ _j_ ) is microbe functional similarity matrix, measured using the method in [26]; _GM_ ( _m_ _i_ _,_ _m_ _j_ ) is the Gaussian interaction profile kernel similarity between microbes, calculated
as follows:


_GM_ ( _m_ _i_ _,_ _m_ _j_ ) = exp( − _η_ _m_ ∥ _Y_ _m_ _i_ − _Y_ _m_ _j_ ∥ [2] ) (5)


where _Y_ _m_ _i_ represents the _i_ -column in the adjacency matrix as
interaction profiles for microbe _m_ _i_ ; _η_ _m_ represents the normalized
kernel bandwidth [25], defined as follows:



_W_ [(] _[l]_ [)] ∈ **R** [(] _[N]_ _[d]_ [+] _[N]_ _[m]_ [)] [×] _[k]_ _[l]_ is a weight matrix for the _l_ th neural network
layer and _k_ _l_ is the dimensionality of embeddings of _l_ th layer GCN;
_σ_ ( - ) is a non-linear activation function.

In our methods, we use ReLU (Rectified Linear Unit) as the
activation function. For the first layer, we construct the initial
embedding _H_ [(0)] as follows.



0 _Y_
_H_ [(0)] =
_Y_ _[T]_ 0

[



]



(9)



_η_ _m_ = _η_ _m_ [′] _[/]_ [(] [1]

_N_ _m_



_N_ _m_
∑ ∥ _Y_ _m_ _i_ ∥ [2] ) (6)


_i_ = 1



where _η_ _m_ [′] [is the original bandwidth.]


_2.5. Heterogeneous network_


In order to incorporate the network information in data integration, we build a heterogeneous biological network including
a microbe network **K** _[m]_ _f_ [, a drug network] **[ K]** _s_ _[d]_ [, and an association]
network between microbes and drugs. Finally, we construct the
heterogeneous network defined by the adjacency matrix _A_ ∈
**R** [(] _[N]_ _[d]_ [+] _[N]_ _[m]_ [)] [×] [(] _[N]_ _[d]_ [+] _[N]_ _[m]_ [)] :



_A_ = _Y_ **K** _[T][d]_ _s_ **K** _Y_ _[m]_

[ _f_



]



(7)



_2.7. Combined kernel on graph embedding_


Multiple embeddings can be computed by multi-layer GCN
model, which represent information of different graph structures.
In detail, _H_ 0 represents the original features of nodes in the
heterogeneous graph, and _H_ 1 aggregates the first-order neighbor information of nodes on the basis of original features. This
means that each additional layer will aggregate the information
on neighbor once more ( _l_ -layer fuses the _l_ th-order neighbor information for each node). Since the embedding of each layer
represents different information, we use the embedding of each
layer as different feature vectors to calculate multiple kernel
matrices.

For the embedding of each layer _H_ _i_ ( _i_ = 1 _, . . .,_ _L_ ), we can
divide it into two parts, the first _N_ _d_ lines are used as drug embeddings _H_ _i_ _[d]_ [, and the last] _[ N]_ _[m]_ [ lines are used as microbe embeddings]



∈ **R** [(] _[N]_ _[d]_ [+] _[N]_ _[m]_ [)] [×] _[k]_ _[i]_, _H_ _i_ _[d]_ [∈] **[R]** _[N]_ _[d]_ [×] _[k]_ _[i]_ [ and] _[ H]_ _i_ _[m]_ ∈
]



_2.6. Graph convolutional network_


Graph Convolutional Network (GCN) is a neural network that
can learn low dimensional representation, but it is applied to
graph structure. GCN is ingeniously designed to extract features
from graph, so that we can obtain graph embeddings to solve the
problems of node classification, graph classification, link prediction and so on.

Specifically, given the adjacency matrix of the heterogeneous
network _A_ defined above, the GCN of the heterogeneous network
can be defined as follows:



_H_ _i_ _[m]_ [, where] _[ H]_ _[i]_ [ =] [ _HH_ _i_ _[m]_ _i_ _[d]_




[1]

2 _AD_ [−] 2 [1]



_H_ [(] _[l]_ [)] = _f_ ( _H_ [(] _[l]_ [−] [1)] _,_ _A_ ) = _σ_ ( _D_ [−] 2 [1]



2 _H_ [(] _[l]_ [−] [1)] _W_ [(] _[l]_ [−] [1)] ) (8)



**R** _[N]_ _[m]_ [×] _[k]_ _[i]_ . We use GIP to calculate the kernel matrices of drug and
microbe embeddings for each layer as follows:

**K** _[d]_ _h_ _l_ [=] [ exp(] [−] _[γ]_ _[h]_ _l_ [∥] _[H]_ _l_ _[d]_ [(] _[i]_ [)] [ −] _[H]_ _l_ _[d]_ [(] _[j]_ [)] [∥] [2] [)] (10)

**K** _[m]_ _h_ _l_ [=] [ exp(] [−] _[γ]_ _[h]_ _l_ [∥] _[H]_ _l_ _[m]_ [(] _[i]_ [)] [ −] _[H]_ _l_ _[m]_ [(] _[j]_ [)] [∥] [2] [)] (11)


where _H_ _l_ _[d]_ [(] _[i]_ [) and] _[ H]_ _l_ _[m]_ [(] _[i]_ [) are profiles of the] _[ i]_ [-row in the] _[ l]_ [-layer]
drug and microbe embeddings; _γ_ _h_ _l_ denotes the corresponding
bandwidth.

Since different embeddings represent various structural information, the kernel composed of different embeddings will
represent the similarity between nodes from different views.
Combined with existing similarity matrices, we have the kernel
sets of drug space _S_ _[d]_ = { **K** _[d]_ _s_ _[,]_ **[ K]** _[d]_ _h_ 1 _[, . . .,]_ **[ K]** _h_ _[d]_ _L_ [}] [ and microbe space]
_S_ _[m]_ = { **K** _[m]_ _f_ _[,]_ **[ K]** _[m]_ _h_ 1 _[, . . .,]_ **[ K]** _h_ _[m]_ _L_ [}] [.]



where _H_ [(] _[l]_ [)] is the _l_ -layer embedding of nodes, where _l_ = 1 _, . . .,_ _L_ ;
_D_ = _diag_ ( [∑] _j_ _[N]_ = _[d]_ 1 [+] _[N]_ _[m]_ _A_ _i_ _,_ _j_ ) is the diagonal node degree matrix of _A_ ;



3


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_



To improve the performance of predicting microbe–drug associations, we integrate above kernels (in two spaces, respectively)
with multiple kernel fusion, which can combine multiple kernel
matrices by the weighted sum method. The combined kernel can
be defined as follows:



**K** _d_ =


**K** _m_ =



_L_ + 1
∑ _ω_ _i_ _[d]_ _[S]_ _i_ _[d]_ (12)


_i_ = 1


_L_ + 1
∑ _ω_ _i_ _[m]_ _[S]_ _i_ _[m]_ (13)


_i_ = 1



where _S_ _i_ _[d]_ [and] _[ S]_ _i_ _[m]_ are _i_ th kernels in drug and microbe kernel set,
_ω_ _i_ _[d]_ [and] _[ ω]_ _i_ _[m]_ [are the corresponding weight of each kernel. Here, we]
set _ω_ _i_ _[d]_ [=] _[ ω]_ _i_ _[m]_ [=] _L_ + 11 [.]


_2.8. Dual Laplacian regularized least squares model_


We apply the Dual Laplacian Regularized Least Squares
(DLapRLS) [17] framework to predict associations. DLapRLS is a
model based on kernel matrix of two feature spaces. It adds the
graph regularization to improve the prediction ability. The loss
function can be defined as follows:


min _J_ =∥ **K** _d_ _**α**_ _d_ + ( **K** _m_ _**α**_ _m_ ) _[T]_ − 2 **Y** _train_ ∥ [2] _F_
+ _λ_ _d_ _tr_ ( _**α**_ _[T]_ _d_ **[L]** _[d]_ _**[α]**_ _[d]_ [)] [ +] _[ λ]_ _[m]_ _[tr]_ [(] _**[α]**_ _m_ _[T]_ **[L]** _[m]_ _**[α]**_ _[m]_ [)] (14)


where ∥·∥ _F_ is the Frobenius norm, **Y** _train_ ∈ **R** _[N]_ _[d]_ [×] _[N]_ _[m]_ is the adjacency
matrix for microbe–drug associations in the training set; _**α**_ _d_ and
_**α**_ _mT_ ∈ **R** _N_ _d_ × _N_ _m_ are trainable matrices; **K** _d_ ∈ **R** _N_ _d_ × _N_ _d_ and **K** _m_ ∈
**R** _[N]_ _[m]_ [×] _[N]_ _[m]_ are fused kernels in two feature spaces respectively.

**L** _d_ ∈ **R** _[N]_ _[d]_ [×] _[N]_ _[d]_ and **L** _m_ ∈ **R** _[N]_ _[m]_ [×] _[N]_ _[m]_ are the normalized Laplacian
matrices, as follows:


1 _/_ 2
**L** _d_ = **D** [−] _d_ [1] _[/]_ [2] _∆_ _d_ **D** _d_ _[,]_ _∆_ _d_ = **D** _d_ − **K** _d_ (15)

**L** _m_ = **D** [−] _m_ [1] _[/]_ [2] _∆_ _m_ **D** [1] _m_ _[/]_ [2] _[,]_ _∆_ _m_ = **D** _m_ − **K** _m_ (16)


where **D** _d_ ( _k_ _,_ _k_ ) = [∑] _[N]_ _t_ = _[d]_ 1 **[K]** _[d]_ [(] _[k]_ _[,]_ _[ t]_ [) and] **[ D]** _[m]_ [(] _[k]_ _[,]_ _[ k]_ [)] [ =] [ ∑] _[N]_ _t_ = _[m]_ 1 **[K]** _[m]_ [(] _[k]_ _[,]_ _[ t]_ [) are]
diagonal matrices.

The prediction **F** [∗] for microbe–drug associations from two
feature spaces are combined as follows:



**F** [∗] = **[K]** _[d]_ _**[α]**_ _[d]_ [ +] [ (] **[K]** _[m]_ _**[α]**_ _[m]_ [)] _[T]_

2


_2.9. Optimization_



(17)



_∂_ _J_
By letting _**α**_ _[m]_ [=] [ 0, we can obtain:]


( **K** _m_ **K** _m_ + _λ_ _m_ **L** _m_ ) _**α**_ _m_ = **K** _m_ [ 2 **Y** _[T]_ _train_ [−] _**[α]**_ _[d]_ _T_ **K** _dT_ ]

_**α**_ _m_ = ( **K** _m_ **K** _m_ + _λ_ _m_ **L** _m_ ) [−] [1] **K** _m_ [ 2 **Y** _[T]_ _train_ [−] _**[α]**_ _[d]_ _T_ **K** _dT_ ] (21)


In our study, we first initialize all the trainable parameters
randomly, and then calculate the loss function by forward propagation in each iteration. In the process of back propagation,
the parameters of GCN are updated by Adam, and then the
parameters of DLapRLS are updated by Eq (19) and (21).

The overview of our method is shown in Algorithm 1. The
input to the model is the known associations matrix **Y** and the
similarity matrices of drug **K** _[d]_ _s_ [and microbe] **[ K]** _[m]_ _s_ [. Construct a het-]
erogeneous network according to the input matrices and initialize
the embedding _H_ [(0)] . In each iteration, the GCN is used to calculate
the embedding of each layer of the nodes through the forward
propagation and the kernel matrices are calculated according to
the embeddings. After that, we fuse multiple kernel matrices in
two spaces separately. Then, the embedding of GCN is updated by
back propagation, and the parameters of DLapRLS are updated by
iterative function. After the iterative update, the model outputs
the predicted association matrix through Eq. (17).


**Algorithm 1** Algorithm of our proposed method.


**Input:** Known associations **Y** ∈ **R** _[N]_ _[d]_ [×] _[N]_ _[m]_, drug similarity matrix
**K** _[d]_ _s_ [∈] **[R]** _[N]_ _[d]_ [×] _[N]_ _[d]_ [, microbe similarity matrix] **[ K]** _[m]_ _f_ [∈] **[R]** _[N]_ _[m]_ [×] _[N]_ _[m]_ [, random]

initial matrices _**α**_ _d_ ∈ **R** _[N]_ _[d]_ [×] _[N]_ _[m]_ and _**α**_ _m_ ∈ **R** _[N]_ _[m]_ [×] _[N]_ _[d]_ . Parameters
_λ_ _d_, _λ_ _m_, _γ_ _h_ _l_ ( _l_ = 1 _, ...,_ _L_ ) and _N_ the number of iterations for
MKGCN;
**Output:** Prediction of **F** [∗] ∈ **R** _[N]_ _[d]_ [×] _[N]_ _[m]_ ;

1: Construct the heterogeneous network defined by the adjacency matrix _A_ ∈ **R** [(] _[N]_ _[d]_ [+] _[N]_ _[m]_ [)] [×] [(] _[N]_ _[d]_ [+] _[N]_ _[m]_ [)] using Eq. (7);

2: Construct the initial embedding _H_ [(0)] using Eq. (9) and do
forward propagation by GCN;

3: **for** _i_ = 1 → _N_ **do**


4: **for** _l_ = 1 → _L_ **do**

5: Calculate the _l_ -layer kernel matrices **K** _[d]_ _h_ _l_ [and] **[ K]** _h_ _[m]_ _l_ [according]

to _l_ -layer embedding _H_ [(] _[l]_ [)] using Eq. (11);

6: **end for**

7: Calculate the combined kernel **K** _d_ and **K** _m_ in two spaces
using Eq. (13);

8: Calculate loss and update the _l_ -layer embedding _H_ [(] _[l]_ [)] ( _l_ =
1 _, ...,_ _L_ ) by Adam;

9: Update **L** _d_ and **L** _m_ using Eq. (15) and Eq. (16);

10: Update _**α**_ _d_ and _**α**_ _m_ using Eq. (19) and Eq. (21);

11: **end for**

12: Output **F** [∗] = **[K]** _[d]_ _**[α]**_ _[d]_ [+] [(] 2 **[K]** _[m]_ _**[α]**_ _[m]_ [)] _[T]_


**3. Results**


We first introduce the selection of model parameters using
5-fold cross-validation (5-CV) on MDAD dataset. Then, under the
same condition, the influence of different models can be tested for
the predictive analysis. Next, we use three cross validation methods (2-CV, 5-CV and 10-CV) on three datasets (MDAD, aBiofilm
and DrugVirus) to compare with other existing tools. Finally, we
use our model to conduct case studies on selected microbes and
drugs predictions.


_3.1. Experiments setting_


The _k_ -fold cross validation ( _k_ -CV) is widely used to evaluate
the prediction performance. In the process of cross validation, all
associations are divided into _k_ parts averagely. In each fold, one



There are two kinds of parameters need to be optimized in our
model. The first is about GCN parameters, which are optimized
by Adam optimizer [27] to minimize the loss function Eq. (13).
The second is about the parameters of DLapRLS. We can get the
iterative function directly by calculating the partial derivatives.

In the process of optimizing _**α**_ _d_, we first suppose that _**α**_ _m_ is
known, and then calculate the partial derivative of the object
function with respect to _**α**_ _d_ as follows:

_∂_ _J_ = 2 **K** _d_ ( **K** _d_ _**α**_ _d_ + _**α**_ _mT_ **K** _mT_ − 2 **Y** _train_ ) + 2 _λ_ _d_ **L** _d_ _**α**_ _d_ (18)
_**α**_ _d_

By letting _**α**_ _[∂]_ _[d]_ _[J]_ [=] [ 0, we can obtain:]


( **K** _d_ **K** _d_ + _λ_ _d_ **L** _d_ ) _**α**_ _d_ = **K** _d_ [ 2 **Y** _train_ − _**α**_ _mT_ **K** _mT_ ]


_**α**_ _d_ = ( **K** _d_ **K** _d_ + _λ_ _d_ **L** _d_ ) [−] [1] **K** _d_ [ 2 **Y** _train_ − _**α**_ _mT_ **K** _mT_ ] (19)


Similarly, we calculate the partial derivative of the loss function with respect to _**α**_ _m_ as follows:

_∂_ _J_ = 2 **K** _m_ ( **K** _m_ _**α**_ _m_ + _**α**_ _dT_ **K** _dT_ − 2 **Y** _Ttrain_ [)] [ +] [ 2] _[λ]_ _[m]_ **[L]** _[m]_ _**[α]**_ _[m]_ (20)
_**α**_ _m_



4


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_


**Fig. 2.** AUPR under different numbers of iterations.



of them is selected as the test set, and the rest as training set
for the training and testing of the model. A total of _k_ folds are
carried out. We use two evaluation indicators: the area under the
receiver operating characteristic curve (AUC) and the area under
the precision recall curve (AUPR).

In our study, we directly set the number of layers _L_ = 3, set
the embedding dimension to _k_ 1 = 128 _,_ _k_ 2 = 64 _,_ _k_ 3 = 32 and
learning rate in the optimization algorithm is set to 0.001.


_3.1.1. Parameters evaluation_

In our study, several parameters, such as _λ_ _d_, _λ_ _m_, _γ_ _h_ _l_ ( _l_ = 1 _,_
_. . .,_ _L_ ) and _N_ the number of iterations, are critical to our model.
In the parameter selection, we consider all combinations of following values: the range of iterations being {1 _,_ 2 _,_ 3 _, . . .,_ 15};
{2 [−] [5] _, . . .,_ 2 [0] _, . . .,_ 2 [5] } for _γ_ _h_ _l_, _λ_ _d_ and _λ_ _m_ . In the process of selecting
parameters, all experiments are carried out on MDAD dataset, and
evaluated under 5-CV.

The number of iterations _N_ controls the times of updates of
the trainable parameters. Fig. 2 shows the AUPR values under
different numbers of iterations. When the number of iterations
is 5, the AUPR values tend to be stable. In order to fully converge
the model, we choose the number of iterations to be 10.

The predicted effect is affected by the parameter _γ_ _h_ _l_ ( _l_ =
1 _, . . .,_ _L_ ), since different _γ_ _h_ _l_ results in different **K** _[d]_ _h_ _l_ [and] **[ K]** _h_ _[m]_ _l_ [.][ Fig. 3]
shows the effect of changes in _γ_ _h_ _l_ on the predictive performance.
It can be observed that a larger _γ_ _h_ _l_ has a negative impact on
the predictive performance. For _γ_ _h_ 1, the optimal value is reached
at the minimum, and AUPR decreases as _γ_ _h_ 1 increases. When
_γ_ _h_ _l_ is too small, the kernel matrix will be close to 1, so the
parameters outside the value range will not be considered. For
_γ_ _h_ 2 and _γ_ _h_ 3, there is little change when they are smaller, but
AUPR also decreases as the value increases. Our model obtains
best AUPR on MDAD dataset with _γ_ _h_ 1 = 2 [−] [5], _γ_ _h_ 2 = 2 [−] [2] and
_γ_ _h_ 3 = 2 [−] [3] .

The _λ_ _d_ and _λ_ _m_ represent the weights of graph regular terms
in DLapRLS, and are important parameters of the model. In the
process of parameter selection, the highest AUPR is also used as
the index. Fig. 4 shows the AUPR values for different _λ_ _d_ and _λ_ _m_
models. It can be seen that when the difference between _λ_ _d_ and
_λ_ _m_ is small, the effect of the model is relatively better. On the



contrary, when the difference between _λ_ _d_ and _λ_ _m_ is large, the
results will be worse. Our model obtains best AUPR on MDAD
dataset with _λ_ _d_ = 2 [−] [3] and _λ_ _m_ = 2 [−] [4] under 5-CV.


_3.1.2. Performance of different kernels on graph embedding_

An important structure in the MKGCN model is the Graph Convolutional Network. Different layers of GCN can obtain different
node embeddings, and then obtain multiple kernel matrices according to different graph embedding information. In this section,
we discuss the effects of the kernel matrix produced by different layers and the known similarity matrix, and the difference
between single-kernel and multi-kernel of mode.

We use three layers of GCN in our study. Many studies

[8,20,28] have shown that the performance of GCN encoders
decrease as higher convolutional layers. This phenomenon may
be caused by excessive smoothing of GCNs. Therefore, we use
h _l_ +MKGCN to represent the MKGCN model using the kernel
matrix obtained in the _l_ layer ( _l_ = 1 _,_ 2 _,_ 3). Furthermore, in
order to compare the better performance of the kernel matrix
generated by graph convolution, we only use the known microbe
similarity matrix **K** _[m]_ _f_ and drug similarity matrix **K** _[d]_ _s_ [to apply to]
the model, named s+MKGCN. MKGCN is model of assign average
weights to above four kernel matrices.

All models evaluated by 5-CV on MDAD dataset in terms of
AUC and AUPR are shown in Fig. 5 and Table 2. Comparing singlekernel models, AUC and AUPR of h 1 +MKGCN and h 2 +MKGCN are
higher than h 3 +MKGCN and s-MKGCN, which means that the
kernel matrix generated by the first and second layers of GCN
contain more information than the third layer and the known
similarity matrix, so h 1 +MKGCN and h 2 +MKGCN achieve better
results. Therefore, the kernel matrix generated by GCN embeddings is an effective method to describe the relationship between
nodes. On the other hand, comparing MKGCN and single-kernel
models, the performance of MKGCN is significantly better than
that of single-kernel model, which indicates that multi-kernel
model can combine more information for prediction. The kernel
matrix generated by graph convolution can provide supplementary information for the model to improve the prediction effect.
Therefore, we select the mean weighted method to build the
MKGCN model in the following study.



5


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_


**Fig. 3.** AUPR of models with different _γ_ _h_ 1, _γ_ _h_ 2 and _γ_ _h_ 3 .


**Fig. 4.** AUPR of models with different _λ_ _d_ and _λ_ _m_ . The yellow color is the higher value, and blue color is the lower value.


6


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_


**Fig. 5.** AUC and AUPR of five different models on MDAD dataset.



**Table 2**

Performance of MKGCN based on different kernels.


Models AUC AUPR


h 1 + MKGCN 0.9249 0.6181
h 2 + MKGCN 0.8720 0.6048
h 3 + MKGCN 0.7458 0.1562
s + MKGCN 0.7152 0.3401

mean + MKGCN 0.9876 0.9603


_3.1.3. Comparison to existing outstanding tools_

We compare our method with some existing methods for
biological bipartite network prediction. For example, KATZHMDA [29], WMGHMDA [30] and NTSHMDA [31] are used to
predict the association of microbial diseases. IMCMDA [32] and
GCMDR [33] are used for microRNA–disease association prediction and identifying miRNA–drug resistance relationships, respectively. BLM-NII [34] and WNN-GIP [35] are developed to address
drug–target interactions problem. GCNMDA [3] is a GCN-based
framework for predicting new microbe–drug associations.

We use three cross validation methods (2-CV, 5-CV and
10-CV) on three datasets (MDAD, aBiofilm and DrugVirus) to
compare with other existing tools, and all results are shown in
Table 3, Table 4, and Table 5. For instance, under 5-CV, MKGCN
achieves the best AUC and AUPR on MDAD dataset (AUC: 0.9876,
AUPR: 0.9603), aBiofilm dataset (AUC: 0.9926, AUPR: 0.9777)
and DrugVirus dataset (AUC: 0.9821, AUPR: 0.9048). However,
under 2-CV, our method gets the second best AUPR (0.8347)
on DrugVirus which is about 0.07 less than GCNMDA. It may
be since the amount of data is too small, causing our model
to perform poorly. Our method MKGCN also has achieved the
best performance compared with other methods via 10-CV on
MDAD dataset (AUC: 0.9883, AUPR: 0.9662), aBiofilm dataset
(AUC: 0.9951, AUPR: 0.9830) and DrugVirus dataset (AUC: 0.9843,
AUPR: 0.90151). All results clearly show that our model has an
excellent effect in predicting microbe–drug associations. The kernel matrices formed by multi-layer GCN embeddings can provide
multi-source information and supplement the known kernel to
improve the predictive ability of the model.


_3.1.4. Case study on COVID-19_

In this section, we further test the predictive effect of MKGCN
by case study based on DrugVirus dataset. The pandemic caused
by the new SARS-CoV-2 virus is affecting global health, and there
is an urgent need for effective drugs for prevention and treatment. Therefore, we choose SARS-CoV-2 as a case study to test the



**Table 3**
Comparison between our method and other state-of-the-art methods on MDAD
dataset.


2-CV 5-CV 10-CV


Methods AUC AUPR AUC AUPR AUC AUPR


KATZHMDA 0.8365 0.8259 0.8723 0.8384 0.8929 0.8403

NTSHMDA 0.7816 0.7314 0.8302 0.7924 0.8547 0.8299

WMGHMDA 0.8519 0.8201 0.8654 0.8381 0.8729 0.8452

IMCMDA 0.7285 0.7618 0.7466 0.7773 0.7466 0.7727

GCMDR 0.8432 0.8423 0.8485 0.8509 0.8590 0.8634

BLM-NII 0.8583 0.8469 0.9231 0.9263 0.8644 0.8792

WNN-GIP 0.8256 0.8299 0.8721 0.8922 0.8711 0.8863

GCNMDA 0.9384 0.9316 0.9423 0.9376 0.9420 0.9359

MKGCN **0.9820** **0.9400** **0.9876** **0.9603** **0.9883** **0.9662**


**Table 4**
Comparison between our method and other state-of-the-art methods on aBiofilm
dataset.


2-CV 5-CV 10-CV


Methods AUC AUPR AUC AUPR AUC AUPR


KATZHMDA 0.8887 0.8909 0.9013 0.9020 0.9076 0.9098

NTSHMDA 0.8005 0.7284 0.8213 0.7639 0.8211 0.7767

WMGHMDA 0.8309 0.8804 0.8451 0.8903 0.8448 0.8839

IMCMDA 0.7608 0.8404 0.7750 0.8572 0.7747 0.8578

GCMDR 0.8764 0.8842 0.8772 0.8847 0.8732 0.8786

BLM-NII 0.9023 0.9259 0.9256 0.9338 0.9425 0.9463

WNN-GIP 0.8773 0.9235 0.9019 0.9408 0.9055 0.9363

GCNMDA 0.9508 0.9424 0.9517 0.9488 0.9500 0.9496

MKGCN **0.9831** **0.9520** **0.9926** **0.9777** **0.9951** **0.9830**


**Table 5**
Comparison between our method and other state-of-the-art methods on
DrugVirus dataset.


2-CV 5-CV 10-CV


Methods AUC AUPR AUC AUPR AUC AUPR


KATZHMDA 0.7221 0.7124 0.7809 0.7554 0.7895 0.7670

NTSHMDA 0.6976 0.6543 0.7389 0.6973 0.7473 0.7058

WMGHMDA 0.6882 0.7294 0.7230 0.7687 0.7232 0.7621

IMCMDA 0.5739 0.6579 0.6235 0.6962 0.6378 0.7186

GCMDR 0.8029 0.8062 0.8243 0.8206 0.8280 0.8271

BLM-NII 0.7570 0.7602 0.8913 0.8922 0.9012 0.9024

WNN-GIP 0.7154 0.7626 0.8002 0.8436 0.8048 0.8484

GCNMDA 0.8576 **0.9043** 0.8986 0.9038 0.9062 0.9127

MKGCN **0.9540** 0.8347 **0.9821** **0.9048** **0.9843** **0.9151**


predictive ability of our model, while also predicting drugs that
may be effective for treatment. We remove all drug associations



7


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_



**Table 6**
The top 50 predicted SARS-CoV-2-associated drugs.


Rank Drug Evidence Rank Drug Evidence


1 Remdesivir [40] 26 Entecavir –
2 Ribavirin [36] 27 Telbivudine –
3 Memantine [41] 28 Cepharanthine [42]
4 Silvestrol [37] 29 Topotecan –
5 Favipiravir [38] 30 Sofosbuvir [43]
6 Mycophenolic [44] 31 Labyrinthopeptin A1 –
7 Amodiaquine [45] 32 Dibucaine –
8 Emetine [40] 33 Darunavir –
9 Chloroquine [36] 34 BCX4430 (Galidesivir) [43]
10 Saracatinib – 35 4-HPR (Fenretinide) [46]
11 Arbidol (Umifenovir) [47] 36 Lopinavir [40]
12 Roscovitine (Seliciclib) [48] 37 Nitazoxanide [49]
13 Quinacrine [50] 38 N-MCT –
14 Obatoclax [51] 39 Glycyrrhizin [52]
15 Luteolin [53] 40 Azacitidine –
16 EIPA (amiloride) [54] 41 Oseltamivir [55]
17 Nelfinavir [56] 42 Lobucavir –
18 Alisporivir [57] 43 Brequinar –
19 Niclosamide [58] 44 Atovaquone –
20 Monensin – 45 Salinomycin [59]
21 Quinine [60] 46 Daclatasvir [61]
22 Chlorpromazine [62] 47 Asunaprevir –
23 Doxycycline [63] 48 Azaribine –
24 Camostat [64] 49 Amantadine [65]
25 Clevudine – 50 Rimantadine –


related to SARS-CoV-2, and then sort the prediction scores in
descending order.

Table 6 shows Top-50 predicted drugs, some of which have
entered clinical trials or the literature has proven to inhibit or
treat SARS-CoV-2. Choy et al. reported that Remdesivir (the 1st),
Emetine (the 8 _th_ ) and Lopinavir (the 36 _th_ ) have antiviral effects
on SARS-CoV-2 virus. Lopinavir inhibits SARS-CoV-2 replication
with EC 50 at 26.63 µ M and combination of remdesivir and emetine shows synergistic effect in vitro. Wang et al. [36] findings
reveal that Remdesivir (the 2nd) and Chloroquine (the 9 _th_ ) are
highly effective in the control of COVID-19 infection in vitro.
Natural products can inhibit different coronavirus targets and
viral enzymes replication [37] such as Iguesterin, Cryptotanshinone and Silvestrol (the 4 _th_ ). Favipiravir, which is predicted as
top 5th, is currently undergoing clinical studies and shows better
therapeutic responses on COVID-19 in terms of disease progression and viral clearance [38]. Moreover, there are some predicted
drugs that are not currently supported by the literature, but
they are included in the list of Broad-Spectrum Antiviral Agents
(BSAAs) [39], which are worth considering for further research
on the new SARS-CoV-2 virus, such as Topotecan (the 29 _th_ ),
Labyrinthopeptin A1 (the 31st). and Dibucaine (the 32nd) etc.

These prediction results show the ability of MKGCN model
to predict potential associations in the microbe–drug network.
Among the predicted drugs related to the SARS-CoV-2 virus, only
one of the top 10 drugs has no literature support. Among the
predicted top 20 and 30 drugs, 90% and 80% of the drugs are
supported by the literature and proved to be possible for treating
or preventing the SARS-CoV-2 virus.


**4. Discussion and conclusion**


At present, the pandemic caused by the SARS-CoV-2 virus
is affecting global health, and there is an urgent need to find
effective drugs for treatment and prevention. This means that
the human microbes have an impact on human health. Therefore, the research on associations between microbes and drugs
has attracted more attention. Accurately predicting the potential connection between microorganisms and drugs can promote
the development of specific drugs and understand the internal



connection between microorganisms and drugs. However, most
of the discovery on associations between microbes and drugs is
through biological experiments, which are time-consuming and
expensive. Therefore, it is necessary to develop a computational
method to predict microbe–drug associations. In this study, we
propose a model called MKGCN, to predict the potential association in human microbes and drugs. The model first extracts the
embedding features of microbes and drugs in a heterogeneous
network combining microbe network and drug network by using
GCN. Secondly, multi-layer embeddings are calculated to obtain
the multiple kernel matrices. Then, we combine multiple kernels
by means of average weighting. Finally, the combined kernel
matrix is used to predict by DLapRLS method.

The advantage of multi-kernel fusion is that it can effectively
and fully describe the relationship between samples by combining multi-kernel information, so as to improve the prediction
ability of the prediction model. However, in traditional multiple
kernel learning, the kernel matrices are constructed by extracting
the features of a variety of known information sources of the
sample itself, but this does not apply to samples with few known
sources of information. Therefore, we use GCN to construct kernel
matrix from the perspective of different layer features to solve
the above problems. Different from traditional multiple kernel
learning, our method, which constructs kernel matrices by extracting a variety of embedding features through multi-layer GCN,
can provide a variety of kernel matrices, and has achieved the
purpose of using multiple informations. MKGCN has excellent
performance on three existing microbe–drug association datasets
(MDAD, aBiofilm and DrugVirs). In the test, our model has excellent prediction effects. In addition, we also conducted a case study
on the SARS-Cov-2 virus. Among the predicted drugs related
to the SARS-CoV-2 virus, only one of the top 10 drugs has no
literature support. Among the predicted top 20 and 30 drugs, 90%
and 80% of the drugs are supported by the literature and proved
to be possible to treat or prevent the SARS-CoV-2 virus. This
case study shows that our novel model (MKGCN) can accurately
discover new microbe–drug associations.


**CRediT authorship contribution statement**


**Hongpeng Yang:** Conceptualization, Methodology, Software,
Data curation, Writing – original draft. **Yijie Ding:** Visualization,
Writing – review & editing, Formal analysis. **Jijun Tang:** Investigation, Conceptualization, Supervision. **Fei Guo:** Methodology,
Supervision, Writing – review & editing.


**Declaration of competing interest**


The authors declare that they have no known competing financial interests or personal relationships that could have appeared
to influence the work reported in this paper.


**Availability of data and material**


[The datasets and corresponding codes are available at https:](https://github.com/guofei-tju/MKGCN)
[//github.com/guofei-tju/MKGCN.](https://github.com/guofei-tju/MKGCN)


**Acknowledgments**


This work is supported by a grant from National Natural Science Foundation of China (NSFC 61902271, 62172296, 61972280)
and National Key R&D Program of China (2020YFA0908400).



8


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_



**References**


[[1] L.J. Marcos-Zambrano, K. Karaduzovic-Hadziabdic, T. Loncar Turukalo, P.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb1)

[Przymus, V. Trajkovik, O. Aasmets, M. Berland, A. Gruca, J. Hasic, K. Hron,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb1)
[et al., Applications of machine learning in human microbiome studies: a](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb1)
[review on feature selection, biomarker identification, disease prediction](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb1)
[and treatment, Front. Microbiol. 12 (2021) 313.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb1)

[[2] S.A. Whiteside, H. Razvi, S. Dave, G. Reid, J.P. Burton, The microbiome of](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb2)

[the urinary tract—a role beyond infection, Nat. Rev. Urol. 12 (2) (2015) 81.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb2)

[[3] Y. Long, M. Wu, C.K. Kwoh, J. Luo, X. Li, Predicting human microbe–drug](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb3)

[associations via graph convolutional network with conditional random](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb3)
[field, Bioinformatics 36 (19) (2020) 4918–4927.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb3)

[4] T.N. Kipf, M. Welling, Semi-supervised classification with graph convolu
[tional networks, 2016, arXiv preprint arXiv:1609.02907.](http://arxiv.org/abs/1609.02907)

[[5] Z.-C. Zhang, X.-F. Zhang, M. Wu, L. Ou-Yang, X.-M. Zhao, X.-L. Li, A graph](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb5)

[regularized generalized matrix factorization model for predicting links in](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb5)
[biomedical bipartite networks, Bioinformatics 36 (11) (2020) 3474–3481.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb5)

[6] F. Zhang, X. Wang, Z. Li, J. Li, Transrhs: A representation learning method

for knowledge graphs with relation hierarchical structure, in: IJCAI, 2020,
pp. 2987–2993.

[[7] Z. Li, X. Wang, J. Li, Q. Zhang, Deep attributed network representation](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb7)

[learning of complex coupling and interaction, Knowl.-Based Syst. 212](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb7)
[(2021) 106618.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb7)

[[8] Z. Yu, F. Huang, X. Zhao, W. Xiao, W. Zhang, Predicting drug–disease](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb8)

[associations through layer attention graph convolutional network, Brief.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb8)
[Bioinform. (2020).](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb8)

[[9] H. Luo, J. Wang, M. Li, J. Luo, X. Peng, F.-X. Wu, Y. Pan, Drug reposi-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb9)

[tioning based on comprehensive similarity measures and bi-random walk](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb9)
[algorithm, Bioinformatics 32 (17) (2016) 2664–2671.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb9)

[[10] D.-H. Le, Machine learning-based approaches for disease gene prediction,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb10)

[Brief. Funct. Genom. 19 (5–6) (2020) 350–363.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb10)

[[11] S. Van Dam, U. Vosa, A. van der Graaf, L. Franke, J.P. de Magalhaes,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb11)

[Gene co-expression analysis for functional classification and gene–disease](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb11)
[predictions, Brief. Bioinform. 19 (4) (2018) 575–592.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb11)

[[12] L. Jiang, Y. Ding, J. Tang, F. Guo, Mda-skf: similarity kernel fusion for](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb12)

[accurately discovering mirna-disease association, Front. Genet. 9 (2018)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb12)
[618.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb12)

[[13] L. Jiang, Y. Ding, J. Tang, F. Guo, Mda-skf: similarity kernel fusion for](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb13)

[accurately discovering mirna-disease association, Front. Genet. 9 (2018)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb13)
[618.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb13)

[[14] M. Gönen, E. Alpaydın, Multiple kernel learning algorithms, J. Mach. Learn.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb14)

[Res. 12 (2011) 2211–2268.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb14)

[[15] X. Chen, Y.-W. Niu, G.-H. Wang, G.-Y. Yan, Mkrmda: multiple kernel](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb15)

[learning-based kronecker regularized least squares for mirna–disease](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb15)
[association prediction, J. Transl. Med. 15 (1) (2017) 1–14.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb15)

[[16] Y. Ding, J. Tang, F. Guo, Identification of drug-side effect association via](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb16)

[semisupervised model and multiple kernel learning, IEEE J. Biomed. Health](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb16)
[Inf. 23 (6) (2018) 2619–2632.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb16)

[[17] Y. Ding, J. Tang, F. Guo, Identification of drug-target interactions via](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb17)

[dual laplacian regularized least squares with multiple kernel fusion,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb17)
[Knowl.-Based Syst. (2020) 106254.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb17)

[[18] X.-Y. Yan, S.-W. Zhang, C.-R. He, Prediction of drug-target interaction by](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb18)

[integrating diverse heterogeneous information source with multiple kernel](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb18)
[learning and clustering methods, Comput. Biol. Chem. 78 (2019) 460–467.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb18)

[19] P. Han, P. Yang, P. Zhao, S. Shang, Y. Liu, J. Zhou, X. Gao, P. Kalnis,

Gcn-mf: disease-gene association identification by graph convolutional
networks and matrix factorization, in: Proceedings of the 25th ACM
SIGKDD international conference on knowledge discovery & data mining,
2019, pp. 705–713.

[[20] J. Li, S. Zhang, T. Liu, C. Ning, Z. Zhang, W. Zhou, Neural inductive](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb20)

[matrix completion with graph convolutional networks for mirna-disease](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb20)
[association prediction, Bioinformatics 36 (8) (2020) 2538–2546.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb20)

[[21] T. Zhao, Y. Hu, L.R. Valsdottir, T. Zang, J. Peng, Identifying drug–target](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb21)

[interactions based on graph convolutional network and deep neural](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb21)
[network, Brief. Bioinform. (2020).](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb21)

[[22] Y.-Z. Sun, D.-H. Zhang, S.-B. Cai, Z. Ming, J.-Q. Li, X. Chen, Mdad: a special](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb22)

[resource for microbe-drug associations, Front. Cell. Infect. Microbiol. 8](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb22)
[(2018) 424.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb22)

[[23] A. Rajput, A. Thakur, S. Sharma, M. Kumar, Abiofilm: a resource of anti-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb23)

[biofilm agents and their potential implications in targeting antibiotic drug](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb23)
[resistance, Nucleic Acids Res. 46 (D1) (2018) D894–D900.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb23)

[[24] M. Hattori, N. Tanaka, M. Kanehisa, S. Goto, Simcomp/subcomp: chemical](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb24)

[structure search servers for network analyses, Nucleic Acids Res. 38 (2)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb24)
[(2010) W652–W656.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb24)

[[25] T. Van Laarhoven, S.B. Nabuurs, E. Marchiori, Gaussian interaction profile](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb25)

[kernels for predicting drug–target interaction, Bioinformatics 27 (21)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb25)
[(2011) 3036–3043.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb25)

[[26] O.K. Kamneva, Genome composition and phylogeny of microbes predict](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb26)

[their co-occurrence in the environment, PLoS Comput. Biol. 13 (2) (2017)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb26)
[e1005366.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb26)




[27] D.P. Kingma, J. Ba, Adam: A method for stochastic optimization, 2014, arXiv

[preprint arXiv:1412.6980.](http://arxiv.org/abs/1412.6980)

[28] Q. Li, Z. Han, X.-M. Wu, Deeper insights into graph convolutional networks

for semi-supervised learning, in: Proceedings of the AAAI Conference on
Artificial Intelligence, Vol. 32, 2018.

[[29] X. Chen, Y.-A. Huang, Z.-H. You, G.-Y. Yan, X.-S. Wang, A novel approach](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb29)

[based on katz measure to predict associations of human microbiota with](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb29)
[non-infectious diseases, Bioinformatics 33 (5) (2017) 733–739.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb29)

[[30] Y. Long, J. Luo, Wmghmda: a novel weighted meta-graph-based model](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb30)

for predicting human [microbe-disease](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb30) association on heterogeneous
[information network, BMC Bioinformatics 20 (1) (2019) 1–18.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb30)

[[31] J. Luo, Y. Long, Ntshmda: prediction of human microbe-disease association](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb31)

[based on random walk by integrating network topological similarity,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb31)
[IEEE/ACM Trans. Comput. Biol. Bioinform. 17 (4) (2018) 1341–1351.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb31)

[[32] X. Chen, L. Wang, J. Qu, N.-N. Guan, J.-Q. Li, Predicting mirna–disease](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb32)

[association based on inductive matrix completion, Bioinformatics 34 (24)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb32)
[(2018) 4256–4265.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb32)

[[33] Y. a. Huang, P. Hu, K.C. Chan, Z.-H. You, Graph convolution for predicting](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb33)

[associations between mirna and drug resistance, Bioinformatics 36 (3)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb33)
[(2020) 851–858.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb33)

[[34] J.-P. Mei, C.-K. Kwoh, P. Yang, X.-L. Li, J. Zheng, Drug–target interaction pre-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb34)

[diction by learning from local information and neighbors, Bioinformatics](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb34)
[29 (2) (2013) 238–245.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb34)

[[35] T. Van Laarhoven, E. Marchiori, Predicting drug-target interactions for new](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb35)

[drug compounds using a weighted nearest neighbor profile, PLoS One 8](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb35)
[(6) (2013) e66952.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb35)

[[36] M. Wang, R. Cao, L. Zhang, X. Yang, J. Liu, M. Xu, Z. Shi, Z. Hu, W. Zhong, G.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb36)

[Xiao, Remdesivir and chloroquine effectively inhibit the recently emerged](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb36)
[novel coronavirus (2019-ncov) in vitro, Cell Res. 30 (3) (2019) 269–271.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb36)

[[37] M. Boozari, H. Hosseinzadeh, Natural products for covid-19 prevention and](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb37)

[treatment regarding to previous coronavirus infections and novel studies,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb37)
[Phytother. Res. (2020).](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb37)

[[38] Q. Cai, M. Yang, D. Liu, J. Chen, D. Shu, J. Xia, X. Liao, Y. Gu, Q. Cai, Y. Yang,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb38)

[et al., Experimental treatment with favipiravir for covid-19: an open-label](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb38)
[control study, Engineering 6 (10) (2020) 1192–1198.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb38)

[[39] S.H. Basha, Corona virus drugs–a brief overview of past, present and future,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb39)

[J. PeerScientist 2 (2) (2020) e1000013.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb39)

[[40] K.-T. Choy, A.Y.-L. Wong, P. Kaewpreedee, S.F. Sia, D. Chen, K.P.Y. Hui,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb40)

[D.K.W. Chu, M.C.W. Chan, P.P.-H. Cheung, X. Huang, et al., Remdesivir,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb40)
[lopinavir, emetine, and homoharringtonine inhibit sars-cov-2 replication](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb40)
[in vitro, Antiviral Res. 178 (2020) 104786.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb40)

[[41] S. Hasanagic, F. Serdarevic, Potential role of memantine in the prevention](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb41)

[and treatment of covid-19: its antagonism of nicotinic acetylcholine](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb41)
[receptors and beyond, Eur. Respir. J. 56 (2) (2020).](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb41)

[[42] H.-H. Fan, L.-Q. Wang, W.-L. Liu, X.-P. An, Z.-D. Liu, X.-Q. He, L.-H. Song,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb42)

[Y.-G. Tong, Repurposing of clinically approved drugs for treatment of](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb42)
[coronavirus disease 2019 in a 2019-novel coronavirus-related coronavirus](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb42)
[model, Chin. Med. J. (2020).](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb42)

[[43] A.A. Elfiky, Ribavirin, remdesivir, sofosbuvir, galidesivir, and tenofovir](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb43)

[against sars-cov-2 rna dependent rna polymerase (rdrp): A molecular](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb43)
[docking study, Life Sci. 253 (2020) 117592.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb43)

[[44] F. Kato, S. Matsuyama, M. Kawase, T. Hishiki, H. Katoh, M. Takeda, Antiviral](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb44)

[activities of mycophenolic acid and imd-0354 against sars-cov-2, Microbiol.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb44)
[Immunol. 64 (9) (2020) 635–639.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb44)

[[45] M. Hagar, H.A. Ahmed, G. Aljohani, O.A. Alhaddad, Investigation of some](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb45)

[antiviral n-heterocycles as covid 19 drug: Molecular docking and dft](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb45)
[calculations, Int. J. Mol. Sci. 21 (11) (2020) 3922.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb45)

[[46] I. Orienti, G.A. Gentilomi, G. Farruggia, Pulmonary delivery of fenretinide:](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb46)

[a possible adjuvant treatment in covid-19, Int. J. Mol. Sci. 21 (11) (2020)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb46)
[3812.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb46)

[[47] Z. Zhu, Z. Lu, T. Xu, C. Chen, G. Yang, T. Zha, J. Lu, Y. Xue, Arbidol](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb47)

[monotherapy is superior to lopinavir/ritonavir in treating covid-19, J.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb47)
[Infect. 81 (1) (2020) e21–e23.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb47)

[[48] N. Fathi, N. Rezaei, Lymphopenia in covid-19: Therapeutic opportunities,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb48)

[Cell Biol. Int. 44 (9) (2020) 1792–1797.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb48)

[[49] M.T. Kelleni, Nitazoxanide/azithromycin combination for covid-19: A sug-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb49)

[gested new protocol for early management, Pharmacol. Res. 157 (2020)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb49)
[104874.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb49)

[[50] M. Salas Rojas, R. Silva Garcia, E. Bini, V. Pérez de la Cruz, J.C. León Con-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb50)

[treras, R. Hernández Pando, F. Bastida Gonzalez, E. Davila-Gonzalez, M.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb50)
[Orozco Morales, A. Gamboa Domínguez, et al., Quinacrine, an antimalarial](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb50)
[drug with strong activity inhibiting sars-cov-2 viral replication in vitro,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb50)
[Viruses 13 (1) (2021) 121.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb50)

[[51] F.S. Varghese, E. van Woudenbergh, G.J. Overheul, M.J. Eleveld, L. Kurver, N.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb51)

[van Heerbeek, A. van Laarhoven, P. Miesen, G. den Hartog, M.I. de Jonge,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb51)
[et al., Berberine and obatoclax inhibit sars-cov-2 replication in primary](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb51)
[human nasal epithelial cells in vitro, Viruses 13 (2) (2021) 282.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb51)

[[52] P. Luo, D. Liu, J. Li, Pharmacological perspective: glycyrrhizin may be an](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb52)

[efficacious therapeutic agent for covid-19, Int. J. Antimicrob. Ag. 55 (6)](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb52)
[(2020) 105995.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb52)



9


_H. Yang, Y. Ding, J. Tang et al._ _Knowledge-Based Systems 238 (2022) 107888_




[[53] R. Yu, L. Chen, R. Lan, R. Shen, P. Li, Computational screening of antagonists](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb53)

[against the sars-cov-2 (covid-19) coronavirus by molecular docking, Int. J.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb53)
[Antimicrob. Ag. 56 (2) (2020) 106012.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb53)

[[54] O.O. Glebov, Understanding sars-cov-2 endocytosis for covid-19 drug](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb54)

[repurposing, FEBS J. 287 (17) (2020) 3664–3671.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb54)

[[55] A. Belhassan, S. Chtita, H. Zaki, T. Lakhlifi, M. Bouachrine, Molecular dock-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb55)

[ing analysis of n-substituted oseltamivir derivatives with the sars-cov-2](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb55)
[main protease, Bioinformation 16 (5) (2020) 404.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb55)

[[56] N. Yamamoto, S. Matsuyama, T. Hoshino, N. Yamamoto, Nelfinavir inhibits](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb56)

[replication of severe acute respiratory syndrome coronavirus 2 in vitro,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb56)
[2020, BioRxiv.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb56)

[[57] L. Softic, R. Brillet, F. Berry, N. Ahnou, Q. Nevers, M. Morin-Dewaele,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb57)

[S. Hamadat, P. Bruscella, S. Fourati, J.-M. Pawlotsky, et al., Inhibition of](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb57)
[sars-cov-2 infection by the cyclophilin inhibitor alisporivir (debio 025),](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb57)
[Antimicrob. Agents Chemother. 64 (7) (2020).](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb57)

[[58] N.C. Gassen, J. Papies, T. Bajaj, F. Dethloff, J. Emanuel, K. Weckmann, D.E.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb58)

[Heinz, N. Heinemann, M. Lennarz, A. Richter, et al., Analysis of sars-cov-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb58)
[2-controlled autophagy reveals spermidine, mk-2206, and niclosamide as](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb58)
[putative antiviral therapeutics, 2020, BioRxiv.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb58)

[[59] S.K.S. Pindiprolu, C.S.P. Kumar, V.S.K. Golla, P. Likitha, S. Chandra, R.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb59)

[Ramachandra, Pulmonary delivery of nanostructured lipid carriers for](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb59)
[effective repurposing of salinomycin as an antiviral agent, Med. Hypotheses](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb59)
[143 (2020) 109858.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb59)




[[60] M. Große, N. Ruetalo, R. Businger, S. Rheber, C. Setz, P. Rauch, J. Auth, E.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb60)

[Brysch, M. Schindler, U. Schubert, Evidence that quinine exhibits antiviral](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb60)
[activity against sars-cov-2 infection in vitro, 2020.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb60)

[[61] A. Sadeghi, A. Ali Asgari, A. Norouzi, Z. Kheiri, A. Anushirvani, M. Montazeri,](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb61)

[H. Hosamirudsai, S. Afhami, E. Akbarpour, R. Aliannejad, et al., Sofosbuvir](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb61)
[and daclatasvir compared with standard of care in the treatment of pa-](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb61)
[tients admitted to hospital with moderate or severe coronavirus infection](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb61)
[(covid-19): a randomized controlled trial, J. Antimicrob. Chemother. 75](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb61)
[(11) (2020) 3379–3385.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb61)

[[62] M. Plaze, D. Attali, A. Petit, M. Blatzer, E. Simon-Loriere, F. Vinckier, A.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb62)

[Cachia, F. Chrétien, R. Gaillard, Repurposing of chlorpromazine in covid-19](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb62)
[treatment: the recovery study, Encephale (2020) S35–S39.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb62)

[[63] C. Conforti, R. Giuffrida, I. Zalaudek, N. Di Meo, Doxycycline, a widely used](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb63)

[antibiotic in dermatology with a possible anti-inflammatory action against](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb63)
[il-6 in covid-19 outbreak, Dermatol. Ther. (2020).](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb63)

[[64] M. Hoffmann, H. Hofmann-Winkler, J.C. Smith, N. Krüger, L.K. Sørensen, O.S.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb64)

[Søgaard, J.B. Hasselstrøm, M. Winkler, T. Hempel, L. Raich, et al., Camostat](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb64)
[mesylate inhibits sars-cov-2 activation by tmprss2-related proteases and](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb64)
[its metabolite gbpa exerts antiviral activity, 2020, BioRxiv.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb64)

[[65] R. Araújo, J.D. Aranda-Martínez, G.E. Aranda-Abreu, Amantadine treatment](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb65)

[for people with covid-19, Arch. Med. Res. 51 (7) (2020) 739–740.](http://refhub.elsevier.com/S0950-7051(21)01054-6/sb65)



10


