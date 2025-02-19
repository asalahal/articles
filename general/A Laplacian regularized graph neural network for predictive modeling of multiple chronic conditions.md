                                                                       Contents   lists   available   at   ScienceDirect
                                        Computer  Methods  and  Programs  in  Biomedicine
                                                             journal homepage: www.elsevier.com/locate/cmpb
A  Laplacian  regularized  graph  neural  network  for  predictive  modeling  of
multiple  chronic  conditions
Julian    Carvajal    Rico   a, Adel    Alaeddini   a,∗, Syed    Hasib    Akhter    Faruqui   b, Susan    P.    Fisher-Hoch   c,
Joseph    B.    Mccormick   c
a        Department of Mechanical Engineering, The University of Texas at San Antonio, San Antonio, TX, 78249, United States of America
b        Department of Engineering Technology, Sam Houston State University, Huntsville, Tx, 77341, United States of America
c        School of Public Health Brownsville, The University of Texas Health Science Center at Houston, Houston, TX, 78520, United States of America
A  R  T  I  C  L  E               I  N  F  O                 A  B  S  T  R  A  C  T
Keywords:                                                    Background and Goals: One of the biggest diﬃculties       facing healthcare systems today is the prevalence of
Graph neural network                                         multiple chronic diseases (MCC). Mortality and the development of new chronic illnesses are more likely in those
Multiple chronic conditions                                  with MCC. Pre-existing diseases and risk factors speciﬁc to the patient have an impact on the complex stochastic
Risk factors                                                 process that guides the evolution of MCC. This study’s goal is to use a brand-new Graph Neural Network (GNN)
Laplacian regularization                                     model to examine the connections between speciﬁc chronic illnesses, patient-level risk factors, and pre-existing
                                                             conditions.
                                                             Methods: We propose a graph neural network model to analyze the relationship between ﬁve chronic conditions
                                                             (diabetes, obesity, cognitive impairment, hyperlipidemia, and hypertension). The proposed model adds a graph
                                                             Laplacian regularization term to the loss function, which aims to improve the parameter learning process and
                                                             accuracy of the GNN based on the graph structure. For validation, we used historical data from the Cameron
                                                             County Hispanic Cohort (CCHC).
                                                             Results: Evaluating the Laplacian regularized GNN on data from 600 patients, we expanded our analysis from two
                                                             chronic conditions to ﬁve chronic conditions. The proposed model consistently surpassed a baseline GNN model,
                                                             achieving an average accuracy of ≥   89% across all combinations. In contrast, the performance of the standard
                                                             model declined more markedly with the addition of more chronic conditions. The Laplacian regularization
                                                             provided consistent predictions for adjacent nodes, beneﬁcial in cases with shared attributes among nodes.
                                                             Conclusions: The incorporation of Laplacian regularization in our GNN model is essential, resulting in enhanced
                                                             node categorization and better predictive performance by harnessing the graph structure. This study underscores
                                                             the signiﬁcance of considering graph structure when designing neural networks for graph data. Future research
                                                             might further explore and reﬁne this regularization method for various tasks using graph-structured data.
1.    Introduction                                                                                  renal failure, cardiovascular disease, and visual impairment [6]. Obe-
                                                                                                    sity adversely aﬀects both mental and physical health [7]and raises the
    The age of technology and medicine has made it possible for peo-                                risk of heart disease, stroke, and several cancers [8]. Decreased indepen-
ple to have a long and healthy life. There is a veriﬁable global increase                           dence and an elevated risk of injury are two consequences of cognitive
in life expectancy [1]. Although this increase in lifespan is notable, it                           impairment [9]. Elevated cholesterol levels, or hyperlipidemia, are a
is accompanied by an escalating prevalence of chronic disease-related                               primary cause of heart disease and stroke risk [10]. Hypertension can
disabilities,  including  but  not  limited  to  diabetes,  hypertension  [2],                      harm the kidneys and is a common cause of heart disease and stroke
obesity [3], cognitive impairment [4][5], hyperlipidemia, etc. The pres-                            [11]. These disorders frequently result in a decreased quality of life, an
ence of more than one chronic condition is deﬁned as Multiple Chronic                               increased reliance on healthcare services, and a diminished capacity to
Conditions (MCC). Chronic conditions like diabetes have signiﬁcant im-                              carry out daily tasks. To manage these diseases, constant medical atten-
plications for patient health, leading to several complications including                           tion, dietary adjustments, and frequent medication are needed. Clinical
 *     Corresponding author.
    E-mail address:   adel.alaeddini@utsa.edu  (A.   Alaeddini).
https://doi.org/10.1016/j.cmpb.2024.108058
Received 16 August 2023; Received in revised form 25 January 2024; Accepted 2 February 2024

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
data  on  MCC,  when  analyzed,  are  often  complex  in  nature  (patterns                       trast, our proposed Graph Neural Network (GNN) model addresses these
of evolution and their interactions) [12]. Nevertheless, these complex                            limitations by employing a message-passing mechanism. This approach
interactions  can  be  represented  in  a  graphical  model  [13],  in  which                     allows for dynamic information exchange between nodes (patients), ef-
modiﬁable and non-modiﬁable risk factors interconnect with patients’                              fectively capturing the intricate relationships and dependencies within
characteristics. A change in modiﬁable factors (through healthy lifestyle                         the data [27]. Our proposed model introduces a novel element to the
choices) can eﬀectively cause a change in these interconnections. A pos-                          ﬁeld of chronic condition prediction by incorporating Laplacian regular-
itive (healthy) lifestyle change is proven to increase life expectancy free                       ization into a Graph Neural Network framework [28]. This integration
of major MCC [2]. Furthermore, these lifestyle alterations have been                              is a signiﬁcant enhancement, allowing for more nuanced and accurate
shown to promote improved lipid levels and weight management, thus                                representation of complex patient data, particularly in the context of
mitigating  potential  complications     [3].  Additionally,  it  has  been  ob-                  multiple chronic conditions. Unlike traditional models, our GNN with
served that these changes may contribute to a deceleration or even a                              Laplacian regularization eﬀectively captures and analyzes the intricate
possible reversal of cognitive impairment [4,5,14].                                               interdependencies in patient data, providing a more reﬁned approach
    MCCs have been extensively studied and analyzed in the literature,                            to MCC prediction [27]. This advancement not only contributes to the
focused primarily on the inﬂuence of various risk factors over MCCs                               existing literature by oﬀering a more sophisticated analytical tool but
[13]. Alaeddini et al. [12]   proposed a latent Markov regression cluster-                        also sets a new standard for accuracy in healthcare predictive model-
ing model to identify major transitions connecting chronic conditions                             ing.
in MCC networks and validated the proposed model using a large Vet-                               In this work we propose a Laplacian regularized graph neural network
eran Aﬀairs (VA) dataset. Isvoranu et al. [15]   proposed an Ising model                          (LR-GNN)  which  leverages  the  structure  of  the  MCC  network  during
and eLasso technique to build a weighted network and the extended                                 model training to improve the model’s performance. Our main contri-
Bayesian  Information  Criterion.  With  this  network,  they  analyze  the                       butions are:
connection between psycho-pathologies to determine possible pathways
for developing MCCs. With the ﬁndings of this model, Isvoranu et al. de-                           1.     We employ Laplacian regularization to improve the predictive per-
termined gender as a risk factor for comorbidity within patients with                                  formance of the GNN model on the node classiﬁcation task. Speciﬁ-
hypertension,  cerebral  vascular  disease,  and  anxiety  [15].  A  pheno-                            cally, we added a Laplacian regularization term to the loss function
typic  disease  network  was  proposed  by  Zhou  et  al.  [16]where  they                             that encourages the smoothness of the node embeddings and pre-
applied  a  community  detection  algorithm  to  identify  a  cluster  of  co-                         serves the graph structure information in our model. We compared
occurring  conditions  within  the  comorbidity  network.  Faruqui  et  al.                            our model with other GNN models that use diﬀerent regularization
proposed a functional continuous time Bayesian network (FCTBN) to                                      techniques or architectures and showed that our model achieves
analyze the dynamic eﬀect of patients’ modiﬁable lifestyle behaviors                                   state-of-the-art performance on several benchmark datasets related
and their interaction with non-modiﬁable socio-demographics, and pre-                                  to multiple chronic conditions [29].
existing conditions on the appearance of new MCCs [17]. They further                               2.     We evaluated the eﬃcacy        of the proposed Laplacian regularized
extended this to model the FCTBN parameter as a nonlinear state-space                                  GNN as well as the classic GNN model across a multitude of tasks,
model using an extended Kalman Filter to identify the modiﬁable fac-                                   employing a diverse range of graph structures. The model’s perfor-
tors that can help patients attain a healthy lifestyle [18].                                           mance was examined through the lens of accuracy in relation to
    In 1997 Sperduti et al. [19]   applied a neural network to an acyclic-                             inherent graph properties such as degree distribution and commu-
directed   graph,   motivating   early   studies   in   Graph   Neural   Network                       nity structure. Interestingly, we observed that the structure of the
(GNN). Gori et al. [20]   ﬁrst introduced a recursive neural network that                              graph signiﬁcantly inﬂuences the performance of the model, sug-
used the notion of graph neural network. This algorithm can be used                                    gesting that enhancing the graph structure could notably bolster
on all types of graph problems e.g., node classiﬁcation, link prediction,                              the model’s accuracy and generalizability.
etc. Later Scarselli et al. [21]   extended the concept of GNN and its ca-
pabilities to directly process diﬀerent types of graphs, such as acyclic,                         The remainder of the paper is structured as follows. Section   2   presents
cyclic, directed, and undirected. Lu et al. [22]    proposed a framework                          the preliminaries and the proposed methodology. Section    3    discusses
for predicting the risk of chronic disease based on GNNs, based on the                            the study population and validation results. Section  4  provides the sum-
construction of a weighted graph between the similarity of the patients                           mary and concluding remarks.
that suﬀer from the same diseases. Sun et al. [23]    proposed a graph
embedding model for disease prediction based on Electronic Medical                                2.    Proposed methodology
Records (EMRs), and learn the latent node embeddings, enabling an ac-
curate disease prediction for new patients. The model learns a speciﬁc                            2.1.     Deﬁnitions
target node’s representation by iteratively aggregating neighboring in-
formation. Zhang et al. [24]    proposed Diﬀusion-Convolutional Neural
Network (DCNN), a cutting-edge deep learning architecture capable of                              2.1.1.     Bipartite graph
processing  complicated  graph-based  data.  DCNN  utilizes  a  diﬀusion-                             A bipartite graph consists of two disjoint and independent sets of
based convolutional layer and SortPooling    layer to transform variable-                         nodes, denoted as       U   and       V, where |𝐔        | =   𝑚      and |𝐕        | =   𝑛   [30]. Each node
sized graph representations into ﬁxed-length feature vectors for predic-                          𝑢𝑖   ∈           𝐔          is  connected  by  an  edge   𝑒𝑖  ∈         𝐄         to  at  least  one  node       𝑣𝑗     ∈           𝐕    ,
tion and they can capture both local and global graph features based on                           and no two nodes within the same set are connected. The graph can
data. The DCNN may be used to model complex associations between                                  be represented as             (𝐔      , 𝐕      , 𝐄     ), where 𝐄          ⊆          𝐔            ×     𝐕        and |𝐄       | =    𝑘    [31]. This
individuals, their medical histories, and other contributing factors, such                        kind of graph can describe many networks (e.g., social, biological, and
as lifestyle, genetics, and environment, in the context of MCC predic-                            chemical)     [32],  such  as  a  user-item  relationship  in  an  Online  Social
tions. Zitnik et al. [25]utilized graph convolutional networks (GCNs)                             Rating Network [33]. The adjacency matrix A    of a bipartite graph has
to analyze the eﬀect of mixing diﬀerent medications. Furthermore, they                            dimensions 𝑚       ×   𝑛    and is deﬁned by A𝑖𝑗 =1    if (𝑢𝑖, 𝑣𝑗) ∈          𝐄    , and A𝑖𝑗 =0
were able to simulate complicated drug interactions with chronic ill-                             otherwise.
ness with the help of the proposed GCN.
Traditional neural networks, while eﬀective in many scenarios, exhibit                            2.1.2.     Bipartite graph projection
certain  limitations  in  predicting  chronic  diseases  due  to  their  linear                       Most of the network analysis models including GNNs can be deﬁned
data processing approach [26]. They often fail to capture the complex                             over the general form of a graph             (𝐕      , 𝐄     ). The graph        can be created
dependencies and relational information crucial in patient data. In con-                          from a bipartite graph by projecting the edges that connect one set of

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
Fig.        1.        Generating graph,               from data. A bipartite graph of patients and their related features (chronic conditions, risk factors, medication, etc.) are generated
from electronic healthcare data. The bipartite graph is then projected to a full-form graph    ′  of patients with similar conditions to be passed to the proposed GNN
model.
nodes onto the other set of nodes. The two sets of nodes in the bipartite                                         𝑈      ={𝑢1,𝑢2 ,  …        ,𝑢𝑚   }
graph are disjoint and independent from each other.                                                               𝐸     ={(𝑣𝑖,𝑢𝑗)|𝑣𝑖  ∈    𝑉,𝑢𝑗   ∈      𝑈     }
     Two  possible  projections  can  be  obtained  from  a  bipartite  graph.
The ﬁrst projection involves the edges 𝐄       projected onto the set of nodes                                    An  illustration  of  the  graph  creation  process  can  be  found  in  Fig.     1.
𝐔    , and the second projection involves the edges 𝐄        projected onto the                                   Next, a connected graph,       ′  =(𝐔      , 𝐄     ′), is generated by projecting the
set of nodes 𝐕        [34].                                                                                       patient nodes 𝑣𝑖   based on their connections to 𝑢𝑖. The weights of the
     Given a bipartite graph     (𝐔      , 𝐕      , 𝐄     )   with |𝐔      (    )| =   𝑛1, |𝐕      (    )| =   𝑛2, and edges 𝑒𝑖  in 𝐺    ′   are assigned using the Euclidean Distance between each
|𝐄     (    )| =   𝑚  , you can obtain the projection of the bipartite graph       onto                         pair of nodes 𝑣𝑖, considering their categorical node features (𝑥𝑣 ):
the vertex set of nodes 𝐔        concerning the vertex set of nodes 𝐕        by cre-
ating a unipartite graph     ′(𝐔      , 𝐄     ′). Here’s how you construct     ′:                               𝑤    (𝑒𝑖)=    ||𝑥𝑣 (𝑣𝑖)−      𝑥𝑣 (𝑣𝑗)||2                                                                                                                                                                                                                        (1)
For each pair of nodes 𝑢𝑖, 𝑢𝑗   ∈         𝐔    , add an edge (𝑢𝑖, 𝑢𝑗) ∈         𝐄     ′  if and only if
there exists a node 𝑣𝑘     ∈         𝐕        such that (𝑢𝑖, 𝑣𝑘 ) ∈         𝐄       and (𝑢𝑗, 𝑣𝑘 ) ∈         𝐄    .2.3.     Convolutional graph neural network
In simpler terms, to create the unipartite graph     ′  from    , you connect
two nodes from the set 𝐔        with an edge in 𝐄     ′  if they are both connected                                    Convolutional GNNs (ConvGNNs) can be considered a generaliza-
to the same node in 𝐕        in the original bipartite graph    . Similarly, the                                 tion of traditional Convolutional Neural Networks (CNNs) for process-
projection of the bipartite graph         for the vertex set of nodes 𝐕         with                             ing graph data. These networks are designed to handle non-Euclidean
respect to the vertex set of the nodes  𝐔         can be made by constructing                                     data structures, such as graphs, by employing localized convolution-like
a unipartite graph           ′′(𝑉,   𝐸    ′′); for each pair of nodes 𝑣𝑖, 𝑣𝑗    ∈         𝐕    , add an          operations on the graph’s vertices. An undirected graph       can be repre-
edge  (𝑣𝑖, 𝑣𝑗) ∈          𝐄     ′′  if and only if there exists a node  𝑢𝑘      ∈          𝐔         such that    sented as     (𝐕      , 𝐄     )   or     (𝐕      , 𝐄     , 𝐴   ), where 𝐕       is the set of nodes with |𝐕        | =   𝑛,
(𝑢𝑘 , 𝑣𝑖) ∈      𝐸      and (𝑢𝑘 , 𝑣𝑗) ∈         𝐄    .                                                            𝐄       is the set of edges with |𝐄       | =   𝑚  , and 𝐴      denotes the adjacency matrix
     After creating the unipartite graphs      ′(𝐔      , 𝐄     ′)    and      ′′(𝐕      , 𝐄     ′′), each      of the graph.
vertex  in  the  projected  graphs  will  have  a  new  degree,  denoted  by                                      Adjacency matrices, 𝐴  , can be either unweighted or weighted. In an un-
deg(𝑢)     for vertices in   𝐔         and   deg(𝑣)     for vertices in   𝐕    . The degree of                    weighted graph, 𝐴   𝑖,𝑗 =1  if there is an edge between nodes 𝑖and 𝑗, while
a  vertex  represents  the  number  of  edges  connected  to  that  vertex  in                                    𝐴   𝑖,𝑗 =0   if there is no edge. For a weighted graph, 𝐴   𝑖,𝑗 =    ℚ       (0 ≤         𝐴   𝑖,𝑗 <   1)
the projected graph. Additionally, it is important to note that the edge                                          if there is an edge between nodes 𝑖  and 𝑗, and 𝐴   𝑖,𝑗 =0    if there is no
weights in the projected graphs can be used to represent the number of                                            edge. The degree matrix of  𝐴       can be denoted as the diagonal matrix
shared connections between the vertices in the original bipartite graph.                                          𝐷   , where 𝐷    𝑖,𝑖=     ∑         𝑛𝑗=1   𝐴   𝑖,𝑗. GNNs utilize the adjacency matrix and the
Higher edge weights indicate a stronger relationship between the ver-                                             degree matrix, along with node feature matrices, to learn meaningful
tices [35]. The developed graph     ′  can now be used to gain insights into                                     representations of the graph data. These representations can then be
the relationships between nodes in the original bipartite graph [36,37].                                          used for various tasks, such as node classiﬁcation, link prediction, and
                                                                                                                  graph classiﬁcation [38]. Once the graph data is represented using ad-
2.2.     Proposed methodology for graph generation                                                                jacency and degree matrices, as well as node features, we can employ
                                                                                                                  GNNs to learn meaningful representations and perform various tasks on
     We  begin  by  cleaning  and  ﬁltering  the  data  from  the  electronic                                     the graph data.
healthcare  record  (EHR)  from  the  Cameron  County  Hispanic  Cohort                                             1.            Node     feature     matrix:     Let’s  represent  the  node  features  for  all
(CCHC). We considered a total of ﬁve chronic conditions for this work                                                    nodes in the graph as a matrix 𝑋         ∈        ℝ      𝑛×𝑑 , where 𝑛   is the number of
(Diabetes,  Obesity,  Hypertension,  Hyperlipidemia,  and  Cognitive  im-                                                nodes and 𝑑    is the dimension of the node features [38].
pairment).  Patient-related  risk  factors  like  smoking  status,  age  cate-                                      2.            Message-passing framework:   GNNs use a message-passing archi-
gories, blood pressure levels, and family history were also extracted.                                                   tecture to aggregate the data from nearby nodes. Every node in the
These factors are used to create a latent representation of each condi-                                                  GNN gets messages from its neighbors and modiﬁes its features as
tion or disease. A bipartite graph,     (𝐔      , 𝐕      , 𝐄     ), is then constructed, where                          necessary for each layer. Let’s say that ℎ (𝑙)𝑣         stands for the feature
𝑣𝑖   represents each patient and 𝑢𝑖   represents categorical risk factors or                                             vector of node 𝑣    at layer 𝑙. At layer 𝑙, the message-passing proce-
features. The bipartite graph can be represented as:                                                                     dure is described as:
    (𝐔      ,   𝐕      ,   𝐄     )                                                                                      ℎ (𝑙+1)𝑣                         =  GeLU(𝑊          (𝑙)   ⋅
                                                                                                                                                                                                                        (2)
𝑉      ={𝑣1,𝑣2 ,  …        ,𝑣𝑛 }                                                                                                 AGGREGATE({ℎ (𝑙)𝑢          ∶   𝑢   ∈      𝑁       (𝑣)}))

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
Fig.      2.      The proposed Graph Neural Network Model. Graph Laplacian loss function is used to train the GNN model, where the edges of the graphs are updated to
improve the model’s performance.
       Here, 𝑁       (𝑣)   denotes the neighbors of node 𝑣, 𝑊          (𝑙)    is the learnable                        messages are aggregated based on their source nodes, employing a sum
       weight matrix at layer 𝑙, and the AGGREGATE function combines                                                  operation,  which  allows  for  an  equal  contribution  from  all  incoming
       the information from the neighboring nodes. GeLU is used as the                                                messages. The original node representations and aggregated messages
       activation function [39]. A generic framework for message-passing                                              are then combined and processed through another FFN to create up-
       neural networks was proposed by Gilmer et al. (2017) [27]and is                                                dated  node  representations.  If  normalization  is  enabled,  the  updated
       now widely used in GNN research.                                                                               representations undergo L2 normalization, which updates the node em-
  3.            Aggregation  function:    The AGGREGATE function can be imple-                                        beddings considering their neighboring nodes.
       mented in various ways. For example, a popular choice is to use                                                The source and target node indices are extracted from the edges. The
       the mean of the neighboring node features [39].:                                                               representations   of   the   target   nodes   (i.e.,   neighbors)   are   then   gath-
       AGGREGATE({ℎ (𝑙)                                                                                               ered. Following the above three steps (prepare, aggregate, update), the
                                𝑢          ∶   𝑢   ∈      𝑁       (𝑣)})   =∑                                          method ﬁnally returns the updated node embeddings [44]. This pro-
                                        1                ℎ (𝑙)                                                        cess emphasizes the inﬂuence of the node’s neighborhood in updating
                                    |𝑁       (𝑣)|𝑢∈  𝑁     (𝑣)𝑢                                                                                                                                                      (3)its own representation, thereby ensuring that the ﬁnal embeddings cap-
  4.            Readout  function:    After  𝐿       layers of message passing, a readout                             ture the local graph structure around each node. This strategy has been
       function is used to generate the graph-level output. The readout                                               found eﬀective in various applications where graph-structured data is
       function, denoted as READ-   OUT, can be a simple summation or a                                               involved. GCL is applied to perform a message-passing operation be-
       more complex pooling operation:                                                                                tween  neighbor  nodes  and  create  a  unique  representation  using  the
                                                                                                                      previous extended representations and the edge weights. A fully con-
       ℎ 𝐺       =  READOUT   ( {ℎ (𝐿  )                                                                              nected FNN is used to perform the ﬁnal node classiﬁcation.
                                      𝑣               ∶   𝑣   ∈      𝑉     })                                                                                                                                                                (4)Fig.   2   shows the illustration of the proposed model. Such techniques
  5.            Graph-level  output:    Finally, the graph-level output can be used                                   have been previously used in [42], [45], [46], [47]    and, [48]. While
       for various tasks, such as graph classiﬁcation, by passing it through                                          training the model, a polynomial decay [49]learning rate scheme is ap-
       a fully connected layer with a suitable activation function:                                                   plied.
                                                                                                                      Additionally, our model incorporates Laplacian regularization (Eq. (8))
       𝑦𝐺       =   𝑓  (𝑊          (𝐿  +1)               ⋅     ℎ 𝐺    )                                                                                                                                                                                                                               (5)to impose smoothness in the feature space of the graph [50]. Speciﬁ-
       Here, 𝑊          (𝐿  +1)     is the learnable weight matrix for the output layer,                              cally, the Laplacian matrix 𝐿       =   𝐷      −    𝐴  , where 𝐴     represents the adjacency
       and  𝑓       is an activation function, such as softmax or sigmoid, de-                                        matrix  (Eq.     (6)),  and   𝐷         the  degree  matrix  (Eq.     (7)),  is  used  in  the
       pending on the task.                                                                                           regularization term. The purpose is to encourage the model to assign
                                                                                                                      similar predictions to neighbor nodes in the graph, hence capturing the
We generate a node      𝑣𝑖   representation by aggregating its own features                                           inherent  topological  information  more  eﬀectively.  This  improves  the
𝑥𝑣   and neighbors’ features   𝑥𝑢. Diﬀerent from RecGNNs [40], ConvGNNs                                               generalizability of the proposed GNN model. This regularization term is
stack multiple graph convolutional layers to extract high-level node rep-                                             deﬁned mathematically as follows:
resentations   [41], [42], [41]. ConvGNNs play a central role in building                                                      {
up many other complex GNN models. Graph Convolutional layers are                                                      𝐴   𝑖𝑗 =     1,           if   there   is   an   edge   between   nodes    𝑖 and    𝑗,
used  to  project  node  features  into  a  linearly  and  separable  low-level                                                    0,           otherwise.                                                                                                                                                                                                                                          (6)
dimensional space in order to have a more unique representation and                                                   𝐷    𝑖𝑖=  degree(𝑖)=    ∑        𝐴   𝑖𝑗                                                                                                                                                                                                                                  (7)
together with the adjacency matrix of the graph [43].                                                                                              𝑗
2.4.     Proposed graph neural network                                                                                𝐿   𝑟   =   𝜆             ⋅    mean   ( (𝑦𝑖 −Σ(     𝑦𝑗            ⋅     𝐿   𝑖𝑗))2 )                                                                                                                                                                                     (8)
                                                                                                                      where, 𝐿   𝑟   is the Laplacian Regularization term, 𝜆    is a hyperparameter,
     The implementation of the Graph Convolutional Layer (GCL) [42]                                                   𝑦𝑖  is the prediction for node 𝑖, 𝑦𝑗   is the prediction for node 𝑗, 𝐿   𝑖𝑗  is the
in this context aims to capture both the characteristics of nodes and                                                 Laplacian matrix element for nodes 𝑖 and 𝑗, and Σ     denotes the summa-
the graph’s structure, with particular emphasis on neighborhood infor-                                                tion over all nodes 𝑗. The Laplacian Regularization term is then added
mation. The process starts with transforming the node representations                                                 to the main loss function, contributing to the total loss, 𝐿       =    𝐿   𝑚       +   𝐿   𝑟,
through a feed-forward neural network (FFN), which are then assigned                                                  where 𝐿   𝑚     is the main loss sparse categorical cross-entropy. This regular-
weights depending on their connections in the graph. Subsequently, the                                                ization scheme is especially useful when dealing with graph-structured

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
data, as it incorporates the topology of the graph into the model’s learn-                                                         Table      1
ing process, which is a novel feature compared to conventional deep                                                                Statistics of the population used in this analysis from
learning models [42]. This promotes homogeneity in the predictions for                                                             CCHC dataset.
nodes that are adjacent or proximal within the graph. In the implemen-                                                                Smokes                                                                                        Body  Mass  Index  (BMI)
tation of our Graph Neural Network model, speciﬁc hyperparameters                                                                     Yes:  265  (44.2%)                                                BMI<18.5:  1  (0.01%)
were carefully selected to optimize performance. Table    A1    provides a                                                            No:  234  (39.0%)                                                   25<=BMI<30:  174  (29%)
comprehensive overview of these hyperparameters, detailing their re-                                                                  No  Answer:  101  (16.8%)                 30  <=BMI<40:  288  (48%)
spective conﬁguration.                                                                                                                                                   BMI>=40:  67  (11.2%)
                                                                                                                                      Parents  Diabetic                                                    Parents  Blood  Pressure
3.    Result and discussion                                                                                                           Yes:  191.5  (25.0%)                                        Yes:  297  (49.5%)
                                                                                                                                      No:  343  (25.0%)                                                   No:  255  (37.5%)
3.1.     Study population                                                                                                             Borderline:  57  (25.0%)                        No  Answer:  78  (13%)
                                                                                                                                      No  Answer:  9  (25.0%)
     We utilized Cameron County Hispanic Cohort (CCHC) dataset to val-                                                                Cholesterol  Level                                                Blood  Pressure  (BP)
idate the proposed model. The CCHC is a cohort study that began in                                                                    High:  239  (40%)                                                  High:  247:  (41%)
2004 and is primarily made up of Mexican Americans (98% of the co-                                                                    Low:  361  (60%)                                                     Low:  353:  (59%)
hort). Participants were chosen at random from a community with sig-                                                                  Mean  Diastolic  BP                                            Mean  Systolic  BP
niﬁcant health inequalities along the Texas-Mexico border. The CCHC                                                                   Min:  38                                                                                       Min:  83
has 5,020 patients currently enrolled and uses a revolving recruitment                                                                Max:  108                                                                                 Max:  195
method. The model’s requirements for inclusion was having valid infor-                                                                Avg:  72.24                                                                          Avg:  120.48
mation about risk variables for individuals in the survey including age,                                                              Age                                                                                                        Mini-Mental  State
body mass index, typical blood pressure, cholesterol level, cigarette us-                                                             Min:  19                                                                                       Avg:  26.92
age, and family medical history. A total of 1,816 patients fulﬁlled this                                                              Max:  93                                                                                      Out  of  30
criterion, but to validate the model only 600 patients were used.                                                                     Avg:  48
While the initial information was provided by patients, it was system-                                                                Gender                                                                                          Total  Population
atically  collected  and  veriﬁed  by  healthcare  professionals,  including                                                          Male:  319                                                                              600
doctors  and  nurses.  This  approach  minimizes  the  risk  of  recall  bias                                                         Female:  281
and other common errors associated with self-reporting. The healthcare
professionals involved in data collection are trained to elicit accurate
and detailed information, ensuring the reliability of the dataset. While
we acknowledge that no data collection method is entirely free from
potential biases or errors, the combination of patient self-reports and
professional oversight in our study provides a robust and trustworthy
foundation for our analysis.
In   the   development   and   validation   of   our   model,   the   dataset   from
the  Cameron  County  Hispanic  Cohort  (CCHC)  underwent  a  rigorous
data cleaning process. We ensured the completeness of patient infor-
mation by including only those records that had complete data on all
risk factors associated with the ﬁve chronic conditions investigated. To
safeguard against any modeling inaccuracies due to missing data, we
meticulously ﬁltered out individuals with incomplete information.
The summary of the CCHC data for the selected patients is shown    in
Table   1.
3.2.     Model performance of Laplacian regularized graph neural network
     In our study, we conducted an evaluation of Laplacian regularized
Graph Neural Networks (GNNs) using a comprehensive dataset of 600
patients. Our approach involved gradually expanding the analysis from
a combination of 2 chronic conditions (2CC) to 5 chronic conditions
(5CC).  The  performance  of  our  proposed  model  consistently  outper-
formed a similar GNN model trained solely using binary cross entropy.                                                                      Fig.      3.   Laplacian   Regularization   Validation.
Speciﬁcally, our proposed model achieved an average accuracy of over
(≥    )  89%  across  all  combinations,  while  the  performance  of  the  reg-                                   (Obesity,  Diabetes,  Hypertension,  Hyperlipidemia,  and  Cognitive  Im-
ular models deteriorated more drastically as more chronic conditions                                               pairment), and comparing the accuracy of the model by predicting the
were incorporated into the analysis. These ﬁndings highlight the eﬀec-                                             category of each patient with and without the regularization, the results
tiveness of incorporating Laplacian regularization in GNNs to enhance                                              are shown in Fig.   3.
their performance, particularly when simultaneously dealing with mul-                                               In other words, the Laplacian regularization encourages the model to
tiple chronic conditions. The Laplacian regularization encourages the                                              yield similar predictions for adjacent nodes. This property is particu-
model  to  yield  similar  predictions  for  adjacent  nodes.  This  property                                      larly  beneﬁcial  in  cases  where  neighboring  nodes  are  likely  to  share
is particularly beneﬁcial in cases where neighboring nodes are likely                                              similar characteristics and hence, belong to the same class. The regular-
to  share  similar  characteristics  and  hence,  belong  to  the  same  class.                                    ization helps the model to capitalize on this underlying graph structure
To  validate  the  implementation  of  the  Laplacian  Regularization,  the                                        and  smoothens  the  node  representations  over  the  graph.  Moreover,
model  was  implemented  using  600  patients,  starting  from  2  Chronic                                         the Laplacian regularization introduces an additional constraint to the
Conditions (Obesity and Diabetes) until reaching 5 Chronic Conditions                                              model  that  helps  in  mitigating  overﬁtting.  It  can  be  considered  as  a

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
                       Table      2
                       Accuracy of the model after removing relevant and non-relevant edges from the initial graph. We show the results for four diﬀerent
                       setups: (i) model for 2 chronic conditions, (ii) model for 3 chronic conditions, (iii) model for 4 chronic conditions, and (iv) model for
                       ﬁve chronic conditions (CC).
                          Edges          Accuracy  2CC                                                                   Accuracy  3CC                                                                   Accuracy  4CC                                                                   Accuracy  5CCEdges Removed
                                         Non-relevant                 Relevant                 Non-relevant                 Relevant                 Non-relevant                 Relevant                 Non-relevant                 Relevant
                          4907                      93.59%                                       93.59%                     92.68%                                       92.68%                      90.68%                                       90.68%                     90.51%                                       90.51%                     0
                          4757                      89.47%                                       80.35%                     89.21%                                       79.21%                      88.95%                                       78.95%                     88.96%                                       78.95%                     150
                          4607                      85.45%                                       75.25%                     84.21%                                       73.68%                      82.95%                                       74.95%                     84.21%                                       73.68%                     150
                          3757                      80.85%                                       70.13%                     79.68%                                       69.16%                      77.21%                                       68.95%                     78.95%                                       68.42%                     850
                          2907                      75.35%                                       66.16%                     75.42%                                       66.89%                      74.21%                                       67.21%                     73.68%                                       63.16%                     850
                          2057                      70.28%                                       62.68%                     69.89%                                       58.63%                      68.42%                                       59.68%                     68.42%                                       57.89%                     850
                          1207                      65.16%                                       56.51%                     65.63%                                       55.11%                      63.68%                                       56.16%                     63.16%                                       53.16%                     850
                          657                           60.16%                                       39.95%                     59.37%                                       39.14%                      57.89%                                       40.37%                     57.89%                                       37.89%                     550
Fig.      4.      Test accuracy:   a) Based on non-relevant edges removal for using a population of 100 patients. b) Based on relevant edges removal for using a population of
100 patients.
form  of  structural  prior  that  guides  the  model’s  learning  process.  By                                                          of the proposed method. Additionally, we also check how much each
incorporating information from the graph structure, the regularization                                                                   edge type (both relevant and non-relevant) aﬀects the accuracy of the
promotes  a  better  generalization  capability.  In  conclusion,  Laplacian                                                             model.  Table     2     shows     the  accuracy  of  the  results  recorded  from  our
regularization demonstrates to be a crucial part of our model, resulting                                                                 experiments. We conducted ﬁve experiments-
in a reliable and successful node categorization. It takes advantage of
the graph structure to produce better prediction performance, demon-                                                                        1.     The graph is generated for 2 chronic conditions (diabetes and obe-
strating how crucial it is to take the graph structure into account when                                                                          sity) and 100 patients’ data.
designing neural networks for graph data. The regularization method                                                                         2.     The graph is generated for 3 chronic conditions (diabetes, obesity,
may be further investigated and tuned in future work for diﬀerent tasks                                                                           and hyperlipidemia) and 100 patients’ data.
using graph-structured data.                                                                                                                3.     The graph is generated for 4 (diabetes, obesity, hyperlipidemia, and
                                                                                                                                                  hypertension) chronic conditions and 100 patients’ data.
3.3.     GNN model validation                                                                                                               4.     The graph is generated for 5 chronic conditions (diabetes, obesity,
                                                                                                                                                  hyperlipidemia, hypertension, and cognitive impairment) and 100
      A subset of 100 patients was randomly selected from CCHC to vali-                                                                           patients’ data.
date the model’s performance. Sixteen diﬀerent graphs were generated
based on the number of edges between the nodes for validation pur-                                                                       A total of 64 diﬀerent graphs, 16 per case were generated for this pur-
poses.  Two  diﬀerent  edge  removal  methodologies  were  applied  to  a                                                                pose. Figs.   4.a) and 4.b)    summarize the results for all the experiments
connected graph (the initial graph had 4907 edges),                                                                                      visually. While we show the results for both edge removal methodolo-
                                                                                                                                         gies in Table   2.
  1.     Removing non-relevant edges (lowest edge weights). This is to re-                                                               From Table    2    we    see that the removal of relevant edges has a more
        move edges between patients who are not related or not similar in                                                                substantial impact on the model’s accuracy compared to non-relevant
        characteristics.                                                                                                                 edges. In the case of 2 chronic conditions (2CC) setup, where the fo-
  2.     Removing relevant edges (high edge weights). This is to remove                                                                  cus  is  on  two  chronic  conditions,  the  model  achieves  high  accuracy
        edges between patients who are strongly similar in characteristics.                                                              levels for both relevant and non-relevant predictions, with a slight de-
                                                                                                                                         crease in accuracy as edges are removed from the graph. However, in
The determination of the relevance of the edges was determined by how                                                                    the 5 chronic conditions (5CC) setup, the accuracy is severely aﬀected
similar the nodes are based on the Euclidean distance between the fea-                                                                   by  the  removal  of  relevant  edges,  resulting  in  a  noticeable  decrease
tures of the patients. These experiments are set to check the sensitivity                                                                in  accuracy.  In  both  cases,  the  accuracy  drop  is  in  linear  relation  to

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
the number of edges being removed. This is the same for all the CC                              rate for the outcome of possible multiple chronic conditions, enabling
cases. These ﬁndings highlight the need to balance the removal of both                          accurate node classiﬁcation as the model’s complexity increases (by in-
relevant and non-relevant edges to optimize the model’s performance                             creasing the number of chronic conditions). In the future, we plan to
and suggest the potential for developing a method to achieve this bal-                          explore  advanced  edge  removal  strategies  to  strike  a  better  balance
ance. Such a method could improve accuracy by selectively removing                              between relevant and non-relevant edges and reﬁne the Laplacian reg-
edges while considering the varying impact of diﬀerent edge types. The                          ularization method for optimal performance. Furthermore, we plan to
proposed graph Laplacian does exactly that, while an active learning                            explore the complex relationships in the graphs that are being learned
framework updates the model’s edges to attain the highest accuracy. In                          through this algorithm for better explanations in the healthcare domain
the next section, we report the results attained from running the pro-                          to provide medical practitioners with better analytical tools. Looking
posed algorithm for diﬀerent combinations of chronic conditions.                                ahead, our future work aims to reﬁne the structure of our model’s graph
                                                                                                through advanced techniques like Graph Autoencoders (GAEs) or other
3.4.     Study limitations and considerations                                                   unsupervised learning methodologies. A   primary focus will be reﬁning
                                                                                                the  graph  construction  process  to  minimize  bias  by  developing  algo-
    Our study, while presenting novel insights into predictive modeling                         rithms that learn directly from electronic health records to form patient
for chronic conditions, acknowledges certain limitations. Our study uti-                        similarity graphs. Beyond chronic disease management, the potential
lizes the Cameron County Hispanic Cohort (CCHC), a comprehensive                                applications of GNN models can be extended into diverse healthcare
and detailed dataset, while acknowledging its demographic speciﬁcity.                           domains. For instance, it could be adapted for infectious disease track-
This particular focus, while oﬀering in-depth insights, might limit the                         ing, where data integration is crucial, or personalized medicine, where
direct applicability of our ﬁndings to populations with diﬀering demo-                          it could aid in tailoring treatment regimens based on patient similari-
graphic characteristics or healthcare systems. The lack of readily avail-                       ties proﬁles.
able, multivariate medical datasets is a barrier to fast generalization,                        The proposed Laplacian Regularized Graph Neural Network model of-
particularly when it comes to chronic conditions. Future research could                         fers  signiﬁcant  beneﬁts  across  the  healthcare  spectrum.  For  patients,
extend this model’s validation to a broader array of datasets as they be-                       it  promises  personalized  risk  assessment  for  multiple  chronic  condi-
come available, enhancing its applicability across various demograph-                           tions, potentially leading to earlier interventions and better health out-
ics. Additionally, while the model accounts for a comprehensive set of                          comes.  Healthcare  providers  can  utilize  the  model  to  identify  at-risk
risk factors, there may be external variables not encompassed within                            patients more eﬀectively, optimizing resource allocation and enhancing
our dataset that could inﬂuence chronic condition outcomes. These con-                          the precision of treatment plans. For policymakers, the insights from
siderations are crucial for interpreting the results, as they highlight the                     the model can inform public health strategies and healthcare policies by
delicate balance between model speciﬁcity and the broader applicabil-                           highlighting population-speciﬁc needs and predicting future healthcare
ity in diverse healthcare settings.                                                             demands. By addressing these multifaceted beneﬁts, our model stands
The  proposed  model  stands  to  signiﬁcantly  advance  healthcare  out-                       to improve predictive healthcare analytics, ultimately contributing to
comes by facilitating early diagnosis and personalized treatment strate-                        more proactive and patient-centric healthcare systems.
gies for chronic conditions. Through its predictive analytics, healthcare
providers can better identify high-risk patients, potentially leading to                        CRediT authorship contribution statement
more timely and eﬀective interventions which can be reﬂected in po-
tential cost savings for healthcare providers. However, integrating such                            Julian Carvajal         Rico:    Software, Validation, Visualization, Writing
a model into clinical practice comes with responsibilities; it necessitates                     –   original   draft,   Data   curation.             Adel      Alaeddini:      Conceptualization,
thorough validation to avoid over-reliance on predictive outcomes and                           Investigation,  Methodology,  Supervision,  Writing  –  review  &  editing.
to ensure that the model’s recommendations complement, rather than                              Syed        Hasib        Akhter Faruqui:   Data curation, Formal analysis, Method-
replace, clinical judgment. This dual approach will maximize beneﬁts                            ology, Project administration.          Susan          P.  Fisher-Hoch:    Funding acqui-
for  patient  care  while  minimizing  potential  risks  associated  with  the                  sition, Resources, Supervision, Writing – review & editing.         Joseph         B.
model’s application.                                                                            Mccormick:   Funding acquisition, Resources, Supervision.
While  our  current  model  focuses  on  ﬁve  speciﬁc  chronic  conditions
(diabetes, obesity, cognitive impairment, hyperlipidemia, and hyperten-                         Declaration of competing interest
sion), it possesses an inherent scalability. The framework is designed to
accommodate additional chronic conditions by integrating relevant pa-                               The authors declare that they have no known competing ﬁnancial
tient data and associated risk factors into the existing graph structure.                       interests or personal relationships that could have appeared to inﬂuence
Specially, adding the Laplacian regularization prunes dense connection                          the work reported in this paper.
that makes the model more light compared to traditional dense mod-
els thus adding additional chronic conditions should cause less issues in                       Appendix       A
terms of scalability. Future iterations of the model could thus expand
its scope to cover a broader spectrum of chronic diseases, enhancing its                            A simple Graph Neural Network [21]model is used for node the
applicability and utility in diverse clinical scenarios.                                        classiﬁcation tasks. The model is implemented using TensorFlow and
                                                                                                consists of several key components. We describe the components below-
4.    Conclusion
                                                                                                  1.            Initialization:    In the initialization stage, the model unpacks the
    This study introduced and analyzed the performance of the Lapla-                                  graph info, which includes node features, edges, and edge weights.
cian regularized graph neural network. The ﬁndings highlight the crit-                                If edge weights are not provided, they are set to one and then nor-
ical  role  of  graph  connectivity  and  edges’  relevance  in  optimizing  a                        malized so that they sum to 1.
model’s performance. Our study used a comprehensive dataset of 600                                2.            Pre-processing Layer:   The initial segment of the proposed model
patients across various chronic conditions. Through iterative edge re-                                encompasses  a  pre-processing  layer,  which  is  essentially  a  Feed-
moval  experiments,  we  observed  that  the  accuracy  of  the  model  de-                           Forward Network (FFN). This preliminary layer serves to process
clined  as  the  number  of  edges  decreased,  with  a  more  pronounced                             the incoming node features, thereby generating initial node rep-
impact when removing relevant edges compared to non-relevant ones.                                    resentations. This helps to transform the raw input features into
Furthermore, the incorporation of Laplacian regularization signiﬁcantly                               a more useful embedding features. This feature set is then further
improves a model’s performance and decreases the miss-classiﬁcation                                   processed within the subsequent layers of the model.

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
                                                           Table      A1
                                                           Graph Neural Network Hyperparameters.
                                                             Number  Chronic  Conditions                                                            2                                                   3                                                   4                                              5
                                                             Model                                                 hidden_units                                                [64,64]                         [64,64]                         [64,64]                    [64,64]
                                                                                     dropout_rate                                                0.02                                      0.02                                      0.02                                 0.02
                                                                                     num_epochs                                                  2500                                   3000                                   3000                              5000
                                                                                     batch_size                                                           16                                              32                                              32                                         32
                                                                                     patience                                                                 1500                                   1500                                   1500                              1500
                                                             Learning  Rate                 starter_learning_rate                 0.00005                      0.00005                      0.00005                 0.00005
                                                                                     end_learning_rate                             0.000007                 0.000007                 0.00001                 0.00001
                                                                                     power                                                                          0.5                                           0.5                                           0.5                                      0.5
                                                                                     decay_steps                                                     9000                                   9000                                   9000                              9000
  3.            Graph   Convolution   Layer:     The  initial  node  representations  are                                    References
       then passed through the ﬁrst Graph Convolution Layer (GCL). The
       GCL performs a message-passing operation to create new represen-                                                        [1]    H. Beltrán-Sánchez, S. Soneji, E. Crimmins, Past, present, and future of healthy life
       tations of the nodes. It uses neighborhood information and edge                                                               expectancy, Cold Spring Harb. Perspect. Med. 5 (2015) a025957.
       weights to create a unique representation that captures both local                                                      [2]    Y. Li, J. Schoufour, D.D. Wang, K. Dhana, A. Pan, X. Liu, M. Song, G. Liu, H.J. Shin,
                                                                                                                                     Q. Sun, et al., Healthy lifestyle and life expectancy free of cancer, cardiovascular
       and global information about the nodes in the graph [51].                                                                     disease, and type 2 diabetes: prospective cohort study, BMJ 368 (2020).
  4.            Skip Connection:   A skip connection is added between the output                                               [3]    S.T. Kuna, D.M. Reboussin, K.E. Borradaile, M.H. Sanders, R.P. Millman, G. Zam-
       of the ﬁrst GCL and its input (the pre-processed node representa-                                                             mit, A.B. Newman, T.A. Wadden, J.M. Jakicic, R.R. Wing, et al., Long-term eﬀect
       tions). This step helps to preserve the original information through                                                          of weight loss on obstructive sleep apnea severity in obese patients with type 2 dia-
       the network and also helps to mitigate the vanishing gradient prob-                                                           betes, Sleep 36 (2013) 641–649.
                                                                                                                               [4]    H.  Shimada,  T.  Doi,  S.  Lee,  H.  Makizako,  Reversible  predictors  of  reversion
       lem.                                                                                                                          from mild cognitive impairment to normal cognition: a 4-year longitudinal study,
  5.            Post-processing Layer:   The output of the skip connection is then                                                   Alzheimer’s Res. Ther. 11 (2019) 1–9.
       passed through another FFN, the post-processing layer, which fur-                                                       [5]    M. Kivipelto, F. Mangialasche, T. Ngandu, Lifestyle interventions to prevent cog-
       ther processes the node embeddings.                                                                                           nitive impairment, dementia and Alzheimer disease, Nat. Rev. Neurol. 14 (2018)
                                                                                                                                     653–666.
  6.            Logit Computation:    This layer undertakes the task of computing                                              [6]    D. Tomic, J.E. Shaw, D.J. Magliano, The Burden and Risks of Emerging Complica-
       the logits associated with each class.                                                                                        tions of Diabetes Mellitus, Nature Reviews Endocrinology, vol.  18, Nature Publishing
                                                                                                                                     Group, 2022, pp.  525–539. Number: 9.
This    combination    of    pre-processing,    graph    convolution,    and    post-                                          [7]    C. Avila, A.C. Holloway, M.K. Hahn, K.M. Morrison, M. Restivo, R. Anglin, V.H.
processing  allows  the  model  to  learn  complex  patterns  in  both  the                                                          Taylor, An overview of links between obesity and mental health, Curr. Obes. Rep. 4
                                                                                                                                     (2015) 303–310.
features of the nodes and the structure of the graph, thereby produc-                                                          [8]    J. Weschenfelder, J. Bentley, H. Himmerich, J. Weschenfelder, J. Bentley, H. Him-
ing a powerful model for node classiﬁcation tasks in graphs. The logits                                                              merich, Physical and mental health consequences of obesity in women, in: Adipose
computed by the model can be used directly in a softmax cross-entropy                                                                Tissue, IntechOpen, 2018, https://www .intechopen .com /chapters /59223.
loss for training the model with labeled data.                                                                                 [9]    H. Ward, T.E. Finucane, M. Schuchman, Challenges related to safety and indepen-
     For the experimented model details are provided below in Table   A1                                                             dence, Med. Clin. North Am. 104 (2020) 909–917.
                                                                                                                             [10]    A. Alloubani, R. Nimer, R. Samara, Relationship between hyperlipidemia, cardio-
for reproducibility.                                                                                                                 vascular disease and stroke: a systematic review, Curr. Cardiol. Rev. 17 (2021)
                                                                                                                                     e051121189015.
Appendix       B.    Model learning rate                                                                                     [11]    M. Weldegiorgis, M. Woodward, The impact of hypertension on chronic kidney dis-
                                                                                                                                     ease and end-stage renal disease is greater in men than women: a systematic review
                                                                                                                                     and meta-analysis, BMC Nephrol. 21 (2020) 506.
     While training the proposed GNN model we updated the learning                                                           [12]    A. Alaeddini, C.A. Jaramillo, S.H. Faruqui, M.J. Pugh, Mining major transitions of
rate (LR) at each epoch. We utilized TensorFlow’s Polynomial Decay                                                                   chronic conditions in patients with multiple chronic conditions, Methods Inf. Med.
[49]    function is used to achieve this. The initial learning (ILR) rate is                                                         56 (2017) 391–400.
                                                                                                                             [13]    S.H.A.  Faruqui,  A.  Alaeddini,  J.  Wang,  S.P.  Fisher-Hoch,  J.B.  Mccormick,  J.C.
ﬁrst set at 0.00005 and then, in accordance with a polynomial decay                                                                  Rico, A model predictive control functional continuous time Bayesian network for
schedule, it steadily declines over time.  The ultimate learning (ULR)                                                               self-management of multiple chronic conditions, http://arxiv .org /abs /2205 .13639,
rate after the decay is 0.000001 and is applied across 9000 steps (DS).                                                              https://doi .org /10 .48550 /arXiv .2205 .13639, 2022, arXiv :2205 .13639  [cs ,  stat].
The decay’s polynomial power (P) is 0.5, which means that over time,                                                         [14]    J.A. Blumenthal, P.J. Smith, S. Mabe, A. Hinderliter, P.-H. Lin, L. Liao, K.A. Welsh-
                                                                                                                                     Bohmer, J.N. Browndyke, W.E. Kraus, P.M. Doraiswamy, et al., Lifestyle and neu-
the learning rate declines at a square root rate and is deﬁned as shown                                                              rocognition in older adults with cognitive impairments: a randomized trial, Neurol-
in following equation.                                                                                                               ogy 92 (2019) e212–e223.
                              (                               )    P                                                         [15]    A.-M. Isvoranu, E. Abdin, S.A. Chong, J.A. Vaingankar, D. Borsboom, M. Subrama-
                                                                                                                                     niam, Extended network analysis: from psychopathology to chronic illness, BMC
LR   =   (ILR   −    ULR)           ⋅1−      Current_StepDS        +  ULR                                                            Psychiatry 21 (2021).
                                                                                                                             [16]    D. Zhou, L. Wang, S. Ding, M. Shen, H. Qiu, Phenotypic disease network analysis
This strategy enables the model to converge more quickly early in train-                                                             to identify comorbidity patterns in hospitalized patients with ischemic heart disease
ing and also enables it to identify more accurate answers as training                                                                using large-scale administrative data, Healthcare 10 (2022).
goes on.                                                                                                                     [17]    S.H.A. Faruqui, A. Alaeddini, J. Wang, C.A. Jaramillo, M.J. Pugh, A functional model
                                                                                                                                     for structure learning and parameter estimation in continuous time Bayesian net-
                                                                                                                                     work: an application in identifying patterns of multiple chronic conditions, IEEE
Appendix       C.    Hardware and software speciﬁcations                                                                             Access (2021) 1.
                                                                                                                             [18]    S.H.A. Faruqui, A. Alaeddini, J. Wang, S.P. Fisher-Hoch, J.B. Mccormick, Dynamic
                                                                                                                                     functional continuous time Bayesian networks for prediction and monitoring of
     We use the following software and hardware for our experiments;                                                                 the impact of patients’ modiﬁable lifestyle behaviors on the emergence of multi-
Python,  leveraging  TensorFlow  and  Keras  for  neural  network  imple-                                                            ple chronic conditions, IEEE Access 9 (2021) 169092–169106.
mentation and NetworkX for graph analysis. The model was run on a                                                            [19]    A. Sperduti, A. Starita, Supervised neural networks for the classiﬁcation of struc-
                                                                                                                                     tures, in: IEEE Transactions on Neural Networks, in: IEEE, vol.  8, 1997, pp.  714–735.
high-performance computing setup equipped with a V100 GPU (32  GB),                                                          [20]    M. Gori, G. Monfardini, F. Scarselli, A new model for learning in graph domains, in:
52   GB of RAM, and dual Intel(R) Xeon(R) CPUs at 2.20   GHz.                                                                        IEEE, vol.  2, 2005, pp.  729–734.

J.      Carvajal      Rico,      A.      Alaeddini,      S.H.A.      Faruqui      et      al.
[21]    F. Scarselli, M. Gori, A.C. Tsoi, M. Hagenbuchner, G. Monfardini, The graph neural                               [36]    M.E. Newman, Networks, Oxford University Press, 2018.
      network model, IEEE Trans. Neural Netw. 20 (2009) 61–80.                                                           [37]    A.-L. Barabási, Network Science, Cambridge University Press, 2016.
[22]    H. Lu, S. Uddin, A weighted patient network-based framework for predicting chronic                               [38]    Z. Zhang, P. Cui, W. Zhu, Deep learning on graphs: a survey, IEEE Trans. Knowl.
      diseases using graph neural networks, Sci. Rep. 11 (2021).                                                               Data Eng. (2018).
[23]    Z. Sun, H. Yin, H. Chen, T. Chen, L. Cui, F. Yang, Disease prediction via graph neural                           [39]    T.N. Kipf, M. Welling, Semi-supervised classiﬁcation with graph convolutional net-
      networks, IEEE J. Biomed. Health Inform. 25 (2021) 818–826.                                                              works, in: Proceedings of the International Conference on Learning Representations
[24]    M. Zhang, Z. Cui, M. Neumann, Y. Chen, An end-to-end deep learning architecture                                        (ICLR), 2017, p.  1, arXiv :1609 .02907.
      for graph classiﬁcation, in: Proceedings of the AAAI Conference on Artiﬁcial Intelli-                              [40]    L. Ruiz, F. Gama, A. Ribeiro, Gated graph recurrent neural networks, IEEE Trans.
      gence, vol.  32, 2018, p.  1.                                                                                            Signal Process. 68 (2020) 6303–6318.
[25]    M. Zitnik, M. Agrawal, J. Leskovec, Modeling polypharmacy side eﬀects with graph                                 [41]    M. Henaﬀ, J. Bruna, Y. LeCun, Deep convolutional networks on graph-structured
      convolutional networks, Bioinformatics 34 (2018) i457–i466.                                                              data, http://arxiv .org /abs /1506 .05163, arXiv :1506 .05163  [cs], 2015.
[26]    L. Alzubaidi, J. Zhang, A.J. Humaidi, A. Al-Dujaili, Y. Duan, O. Al-Shamma, J. San-                              [42]    T.N. Kipf, M. Welling, Semi-supervised classiﬁcation with graph convolutional net-
      tamaría, M.A. Fadhel, M. Al-Amidie, L. Farhan, Review of deep learning: concepts,                                        works, Published as a conference paper at ICLR 2017, 2017.
      CNN architectures, challenges, applications, future directions, J. Big Data 8 (2021)                               [43]    J. Wu, Y. Zhang, R. Fu, Y. Liu, J. Gao, An eﬃcient       person clustering algorithm for
      53.                                                                                                                      open checkout-free groceries, http://arxiv .org /abs /2208 .02973, arXiv :2208 .02973
[27]    J. Gilmer, S.S. Schoenholz, P.F. Riley, O. Vinyals, G.E. Dahl, Neural message passing                                  [cs], 2022, version: 1.
      for quantum chemistry, in: Proceedings of the 34th International Conference on                                     [44]    M. Deﬀerrard, X. Bresson, P. Vandergheynst, Convolutional neural networks on
      Machine Learning (ICML), 2017, pp.  1263–1272, arXiv :1704 .01212.                                                       graphs with fast localized spectral ﬁltering, arXiv :1606 .09375, 2017.
[28]    H.  Yang,  K.  Ma,  J.  Cheng,  Rethinking  graph  regularization  for  graph  neural                            [45]    W.L.    Hamilton,    R.    Ying,    J.    Leskovec,    Inductive    representation    learning    on
      networks, http://arxiv .org /abs /2009 .02027, https://doi .org /10 .48550 /arXiv .2009 .                                large   graphs,  http://arxiv .org /abs /1706 .02216,   https://doi .org /10 .48550 /arXiv .
      02027, 2020, arXiv :2009 .02027  [cs ,  stat].                                                                           1706 .02216, 2018, arXiv :1706 .02216  [cs ,  stat].
[29]    J. Yang, X. Ju, F. Liu, O. Asan, T. Church, J. Smith, Prediction for the risk of                                 [46]    K.  Xu,  W.  Hu,  J.  Leskovec,  S.  Jegelka,  How  powerful  are  graph  neural  net-
      multiple chronic conditions among working population in the United States with                                           works?,   http://arxiv .org /abs /1810 .00826,   https://doi .org /10 .48550 /arXiv .1810 .
      machine learning models, IEEE Open Journal of Engineering in Medicine and Biol-                                          00826, 2019, arXiv :1810 .00826  [cs ,  stat].
      ogy 2 (2021) 291–298.                                                                                              [47]    F. Wu, T. Zhang, A.H.d. Souza Jr., C. Fifty, T. Yu, K.Q. Weinberger, Simplifying
[30]    J.-L. Guillaume, M. Latapy, Bipartite graphs as models of complex networks, Physica                                    graph convolutional networks, http://arxiv .org /abs /1902 .07153, https://doi .org /
      A 371 (2006) 795–813.                                                                                                    10 .48550 /arXiv .1902 .07153, 2019, arXiv :1902 .07153  [cs ,  stat].
[31]    J. Salvatore, Bipartite graphs and problem solving, 2007.                                                        [48]    Y.   Li,   D.   Tarlow,   M.   Brockschmidt,   R.   Zemel,   Gated   graph   sequence   neural
[32]    S. Banerjee, M. Jenamani, D.K. Pratihar, Properties of a projected network of a                                        networks, http://arxiv .org /abs /1511 .05493, https://doi .org /10 .48550 /arXiv .1511 .
      bipartite network, in: 2017 International Conference on Communication and Signal                                         05493, 2017, arXiv :1511 .05493  [cs ,  stat].
      Processing (ICCSP), 2017, pp.  0143–0147.                                                                          [49]    L.N. Smith, Cyclical learning rates for training neural networks, in: 2017 IEEE
[33]    Y. Matsuo, Track: Social networks and web 2.0 / session: Interactions in social com-                                   Winter  Conference  on  Applications  of  Computer  Vision  (WACV),  IEEE,  2017,
      munities community gravity: Measuring bidirectional eﬀects by trust and rating on                                        pp.  464–472.
      online social networks ABSTRACT, 2009.                                                                             [50]    S.H.A. Faruqui, A. Alaeddini, J. Wang, C.A. Jaramillo, M.J. Pugh, A functional model
[34]    S. Banerjee, M. Jenamani, D.K. Pratihar, Properties of a projected network of a                                        for structure learning and parameter estimation in continuous time Bayesian net-
      bipartite network, http://arxiv .org /abs /1707 .00912, arXiv :1707 .00912    [physics],                                 work: an application in identifying patterns of multiple chronic conditions, IEEE
      2017.                                                                                                                    Access 9 (2021) 148076–148089.
[35]    S. Zhang, R.-S. Wang, X.-S. Zhang, Identiﬁcation of overlapping community struc-                                 [51]    P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, Graph atten-
      ture in complex networks using fuzzy c-means clustering, Physica A 374 (2007)                                            tion networks, arXiv :1710 .10903, 2018.
      483–490.

