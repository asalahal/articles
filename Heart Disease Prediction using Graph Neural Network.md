                                                                     International Journal of
                            INTELLIGENT SYSTEMS AND APPLICATIONS IN
                                                                      ENGINEERING
                   ISSN:2147-67992147-6799                                       www.ijisae.org                                    Original Research Paper
                   Heart Disease Prediction using Graph Neural Network
   Rakhi Wajgi1, Tushar Champaneria2, Dipak Wajgi3, Yogesh Suryawanshi4, Dinesh Bhoyar5, Ajinkya
                                                                         Nilawar6
Submitted: 20/11/2023         Revised: 29/12/2023           Accepted: 09/01/2024
Abstract: Heart is an important organ playing vital role in the life of living organisms. Heart and circulatory disease encompasses a
range  of  conditions  affecting  the  heart  and  blood  vessels,  including  coronary  artery  disease,  arrhythmias,  and  heart failure
mechanism. Early detection of malfunctioning before failure of heart is necessary. This paper deals with the model built using Graph
Neural Network (GNN) to predict heart disease so that mortality rate caused due to sudden heart failure can be reduced. In order to
improve the accuracy of GNN-based model, different optimizers are used. They are help                         to optimize    or improve the model's
performance by iteratively updating its parameters to reach the optimal values that minimize the difference between predicted and
actual outputs. The proposed model is applied on a real dataset from kaggle containing 14 features. Out of all optimizers, RMSprop
outperforms otherswith accuracy of 91% and MSE of 48%.
Keywords: Cardiovascular disease, Graph Neural Network, Optimizer
 1.   Introduction                                                                   aggregate  information  from  a  node's                neighbourhood
In  computer  discipline,  graph  is  a           n  important      data             (connected   nodes)   to   update   node   representations,
structure   consisting ofnodes and edges            connecting those                 allowing for powerful learning and inference on graph                    -
nodes. The Graph element may or may not affect the                                   structured data.       Among     various     life     threatening
associated graph element. The feature of graph element                               disease,  heart  disease  has  gained  a  great  attention              in
can depend     upon features of another graph element.             GNN               research. The diagnosis of heart disease is             daunting task,
stands for Graph Neural Network, which is a type of                                  automatic  prediction  about  the  heart  condition  to  be
neural  network  designed  to  handle  data  structure  like                         carried out so that it can be effective to do the further
graphs. Unlike traditional neural networks that process                              treatment.  Numerous          habits  elevate  the  likelihood  of
data organized in a grid or sequence (such as images or                              heart disease, including smoking, a family background
text),  GNNs  are  specialized  for  dealing  with  graph                -           with heart disease, elevated cholesterol levels, obesity,
structured data, where entities (nodes) are connected by                             hypertension,           and           insufficient           physical
relationships (edges).A Graph neural network operates                                activity.Cardiovascular  disease  are  the  major  cause  of
on Graph Structure which has nodes and edges where                                   mortality  in  developing  countries  [7,8,9,10,11]  due  to
nodes  represents  the  entities  and  edges  represent  the                         random and busy life style. Logistic Regression (LR),
relationships between the entities and provide easy way                              Back    Propagation    Neural    Network    (BPNN),    and
to   do   some   node     -level,   edge-level   and   graph-level                   Multilayer   Perceptron   (MLP)   have   been   effectively
prediction tasks.GNNs aim to learn and understand the                                employed as decision         -making tools for predicting heart
underlying patterns, interactions, and representations of                            disease     using     individual      -specific     information[12].
nodes and edges in a graph. They perform operations that                                         Past   literature   suggest   that   hybrid   models
                                                                                     performs better in predicting heart related diseases such
1 Department of Computer Technology, Yeshwantrao Chavan College of                   as Random Forest, Multilayer Perceptron, SVM,  Bayes
Engineering, Nagpur, Maharashtra (India) rakhiwajgi17@gmail.com                      Net[13]. The organization of paper is as follows:
2,   Department   of   Computer   Engineering,   Governemnt   Engineering
College, Modasa, Gujarat,(   India)tushar.champaneria@gecmodasa.ac.in                • Section 2 briefs about literature survey and important
3  Department  of  Computer  Engineering,  St.  VincntPallotti  College  of            terminologies and notations related to GNN.
Engineering           and           Technology,           Maharashtra           (India)
dipak.wajgi@gmail.com                                                                • Section 3 deals with dataset used and methodology of
4 Department of Electronics Engineering, YeshwantraoChavan  College                    implementation  and  section  4  discusses  about  result
of Engineering , Nagpur, INDIA yogesh_surya8@rediffmail.com                            followed by conclusion and future            scope.
5   Department   of   Electronics   and   Telecommunication   Engineering,
Yeshwantrao Chavan College of Engineering, Nagpur, INDIA                             2. Literature Survey
dinesh.bhoyar23@gmail.com
6  Department  of  Electronics  and  Communication  Engineering,  Shri               The graph neural network is proposed                 in [1]   for drug-
Ramdeobaba  College  of  Engineering  and  Management,  Maharashtra                  disease  association  prediction  (GNDD)  framework  to
(India)                                                                              tackle the existing challenge of drug            -disease prediction
nilawarap1@rknec.edu                                                                 which   rely   on   assembling   multiple   drug   related
International Journal of Intelligent Systems and Applications in Engineering                                         IJISAE, 2024, 12(12s), 280–287 |  280

biological information. Assuming that user with similar                         multiple  fields  that  represent  node  connections  as  a
characteristics    would  interact  with  similar  items  is                    function of time and place.          The key idea of       STGNN
widely  used  in  recommender  system  to           eliminate  the              considers spatial dependency and temporal dependency
dependency on multi-data. Author in [2] aims to extend                          at the same time.STGNNs is used to evaluate on the US
the data representation and classification capabilities of                      country    level    COVID-19    dataset.    [5]    Propose
convolutional      neural      network      using      Graph                    Knowledge Graph Neural Network (KGNN) to resolve
Convolutional Neural Network (GCNN). The classifier                             drug disease interaction prediction. This framework can
uses structural connectivity inputs in the form of graph                        effectively      capture      drug      and      its      potential
Laplacian to generate cognitive status category label as                        neighbourhoods by mining their associated relation in
its  output.  For  the  purpose  of  predicting  next-period                    knowledge graph. Author in paper [6] described graph-
prescriptions, the author in [3] suggests a hybrid RNN                          based deep learning model deep2Conv to systematically
and  GNN  method  termed  RCNN.  RNN  is  used  to                              conclude     new  drug  disease  relationships  for  SARS-
describe patient status sequences, while GNN is utilized                        COV-2 drug repositioning. The fundamental idea behind
for presenting periodic medical event graphs.              RNN is               deep2Conv involves combining varied information from
popular     for    patient     longitudinal     medical     data                networks related to SARS-COV-2, including the drug-
representation    but    it    cannot    represent    complex                   drug  network,  drug-disease  network,  and  drug-target
interaction  of  different  medical  information  so  this                      network. This amalgamation aims to deduce potential
temporal  graphs  can  be  represented  by  Graph  neural                       drugs  for  SARS-COV-2  through  a  collective  graph
network. [4] Spatio-temporal graph neural networks, or                          convolutional network.
STGNNs, are a type of graph with applications across
                                                       Fig. 1. General working of GNN
2.1 Terminologies and Notations                                                 passing     to   encapsulateinformation   in   a   node.   This
Graph: Graph is defined as an ordered set and it can be                         information can be about neighboring nodes or GNNs
denoted by G = (V, E) where V represents the vertices                           are The general pipeline for GNN is shown in Fig. 1.The
and E represents the edges which is used to connect the                         input to the neural network is in terms of graph which
vertices.GNNs are being used by an increasing number                            can have node or edge embedding and the output is in the
of businesses to enhance recommendation systems, fraud                          form  of  node  classification  or  edge  classification  or
detection, and medication discovery. A detailed survey                          graph classification.
of GNN can be found in [14] Finding patterns in the                             3. Dataset and Methodology
relationships between data pieces is essential to these and                     The Dataset used in this research is the public health
many other applications. GNN applications in computer                           dataset which is dated from 1988 and it consist of four
graphics,   cybersecurity,   genomics,           recommendation                 dataset  which  is  Cleveland,Hungary,  Switzerland,  and
systems, and materials science etc. are being investigated                      Long Beach V. This dataset has 76 attributes including
by  researchers.  In  a  recent  work,  it  was  shown  that                    ground truth. Figure 2 shows the overall distributions of
GNNs   improved   arrival   time   forecasts   by   using                       all 14 features.
transportation maps as graphs [15]. GNN uses message
International Journal of Intelligent Systems and Applications in Engineering                                  IJISAE, 2024, 12(12s), 280–287 |  281

                                                         Table 1.  Notations used in GNN
                   Notation                           Description
                        G                             A graph
                        V                             The vertex set of G
                        E                             The edge set of G
                        A                             The graph adjacency matrix.
                       AT                             The transpose of the matrix A.
                        v                             The node v ϵ V
                      N (v)                           The adjacent nodes of node v
                        n                             The number of nodes, n = |V|.
                        m                             The number of edges, m = |E|.
                        d                             The dimension of a node feature vector.
                        b                             The dimension of a hidden node feature vector.
                        c                             The dimension of an edge feature vector.
                        hv                            Hidden state of node v
                        ov                            Output of node v
                       Xv                             Features of node v
                      Xco[v]                          Features of its edges
                       hnv                            The states
                       Xnv                            Features of the nodes in the neighbourhood of v
                        k                             The layer index
but  all  the  published  research  used only           14  attributes              1 denotes  heart disease. edges so the machine learning
which  are  listed  in  table  2.  The  predicting  attribute                       algorithms can be benefited from it. significantly more
“target” field refers to heart disease patient and non heart                        effective  because  they  carry  out  graph  classification
disease patient. If the value is 0 then no heart disease and                        directly using the retrieved graph representations [16].
                                              Table 2. Features used for heart disease prediction
Sr. No.            Attribute                               Description
1                  Age                                     Age of patient in year
2                  Sex                                     Gender of patient
                                                           0 = female
                                                           1 = male
3                  cp                                      Chest pain
                                                           0 = typical angina (decreases blood supply to heart)
                                                           1 = not related to heart
                                                           2 =non- heart related
                                                           3 = no sign of disease
International Journal of Intelligent Systems and Applications in Engineering                                       IJISAE, 2024, 12(12s), 280–287 |  282

4                  trestbp                                Resting blood pressure. 120-80 = normal range
5                  chol                                   Serum cholesterol shows the amount of triglycerides present. It should be
                                                          <170mg/dL
6                  fbs                                    Fasting blood sugar. 1 = 120mg/dL
                                                          <100 = normal
                                                          100-125 = prediabetes.
7                  restecg                                Resting electrocardiographic results.
                                                          0 = nothing
                                                          1= Can range from mild to severe
                                                          2= possible left ventricular hypertrophy
8                  thalach                                Maximum heart rate achieved is 220 minus your age
9                  exang                                  Exercise induced angina. Angina is caused due to less blood flow.
10                 oldpeak                                ST depression induced by exercise relative to rest.
11                 slope                                  Slope of the peak exercise ST segment.
                                                          0 = up-sloping. It is uncommon
                                                          1 =indicates healthy heart
                                                          2 = indicates unhealthy heart
12                 ca                                     Colored vessel means the blood passing through. If the blood has movement
                                                          then there is no clots.
                                                          Number of major vessels which varies from 0-3 colored by fluroscopy.
13                 thal                                   Thalassemia stress if 1,3= normal, if value 6: fixed defect, and 7 =
                                                          reversible defect
14                 target                                 Target denotes 0 = no heart disease and 1 = Heart disease presence
It observed that there are more number of male in the age                         used for visualization of data.
group of 52-68 in the dataset. Seaborn python library is
                                                     Fig 2.  Distribution of all 14 attributes
3.1 Methodology
Following are the       steps used for classification of data                     Step 1 :Define GNN Model
into heart disease and no-heart disease.                                          Create a GNN model that consists of several parameters
International Journal of Intelligent Systems and Applications in Engineering                                     IJISAE, 2024, 12(12s), 280–287 |  283

Step 2: Initialization                                                          similar embedding for all nodes.
Set up the initial configuration of the GNN modelwhich
includes specifying the dimensions for input, hidden, and                       3.2 Optimizers
output layers. Also initialize weights and biases for the
connections between these layers.                                               Optimizers are crucial part in neuralnetwork.To choose
Step 3: Node Embedding                                                          the correct optimizer for our application we should know
                                                                                how the optimizers work.
Define a function of message passing between nodes in                           1.   RMSProp  Optimizer:             An  optimization  method
the graph which computes message based on node                                       called RMSprop (Root Mean Square Propagation) is
features and graph structures                                                        used  to   train  neural  networks,  particularly  in  the
Step 4: Aggregation and Transformation                                               context of stochastic gradient descent (SGD) and its
Compute combined message at a node by aggregating                                    variations.  RMSprop's  main  idea  is  to  scale  each
messages of neighboring nodes and apply transformation                               parameter's learning rate inversely proportionate to
which applies weights and biases to this message to                                  the moving average of the squared gradients in order
generate output.                                                                     to adaptively modify it. This means that the learning
                                                                                     rate changes overtime. The value of momentum is
Iterate through step 4 and use backpropagation to update                             denoted by beta which is usually set  to 0.9. The
the weights based on the calculated loss and the all five                            below equation shows the updating rule of RMSprop
optimizers.                                                                          optimizer.
Repeat this process for all batches and continue for the n                           a.   ADAM  optimizer:  ADAM  optimizer  is  First-
number of epochs.                                                                         order-gradient based algorithm. It is known for
1.  Prepare:      the  input  node  representation  will  be                              its   efficiency   in   handling   sparse   gradients,
processed  using  Feed  Forward  network  to  produce  a                                  dealing with noisy or non-stationary problems,
message. Using linear transmission we can simplify the                                    and providing good convergence properties for
process.                                                                                  a  wide  range  of  neural  network  architectures
                                                                                          and problems. The direction of update is given
2.  Aggregate:The  messages  originating  from  every                                     by the first moment normalized by the second
node's neighbors are combined based on edge weights                                       moment.   The   below   equation   shows   the
through various combinations and permutations, such as                                    updating rule of ADAM optimizer.
mean, max, and sum for every node.                                                                                      𝛼
3.  Update:     The  Node-representation  and  aggregated                                         𝜃𝑛+1   =   𝜃𝑛 − √𝜐̂    + ∈  𝑚̂𝑛
                                                                                                                       𝑛
message are combined and processed to produce a new
state of node representation. If the combination-type is a                           b.   Adadelta  optimizer:  Adadelta  is  extension  of
GRU layer then the node representation and aggregated                                     Adagard. In Adadelta instead of summing all
message will be stacked in a queue for the process of                                     past squared gradient we are restricting it to the
GRU  Layer.  Otherwise  the  node  representation  and                                    window size.
aggregated message are added and processed using feed                                c.   Adagard      optimizer:      Adaptive      Gradient
forward neural network.                                                                   Algorithm  is  using  different  learning  rate  for
The GNN Classification model follows the following                                        each parameter based on iterations. The reason
approach:                                                                                 behind this is we need learning rate for sparse
                                                                                          feature parameter need to be higher compared to
a.  To generate initial node representation we will use                                   dense feature learning rate.
    preprocessing on node feature using FFN.                                         d.   SGD:   SGD   stands   for   Stochastic   Gradient
b.  To produce node embedding            use one or more than                             Descent algorithm. The problem with SGD is
    one layer with skip connections.                                                      that we can’t increase its learning rate because
c.  Apply post processing using FFN to generate final                                     of the high oscillation. SGD is slow converge
    node embedding.                                                                       because   it   needs   forward   and   backward
d.  Then feed the node embedding into Softmax layer to                                    propagation for every record.
    predict the node class.                                                     4. Results and Discussions
Every GCL add the information which is captured from                            This  section  shows  the  test  accuracy  for  this  given
the further level of neighbors. Adding more GCL                  can            model. Table shows the accuracy for various optimizers,
result into over smoothing that  means it can produce                           losses and metrics which is used to compile this model.
International Journal of Intelligent Systems and Applications in Engineering                                 IJISAE, 2024, 12(12s), 280–287 |  284

We have used Adam, RMSprop, Adadelta, Adagard and                             Cross-entropy, Mean Absolute Error Loss and Accuracy,
SGD   optimizers.   Binary   Cross-entropy,   Categorical                     Binary Accuracy, Categorical Accuracy Metrics.
                                               Table 3. Comparison of all five optimizers
                                      Optimizer        Binary        Categorical        Mean
                                                       Crossent      Cross-             Absolute Error
                                                       ropy          entropy
                                                                       Binary Accuracy
                                      Adam                85%             87%                  54%
                                      RMSprop             87%             92%                  44%
                                      Adadelta            48%             48%                  58%
                                      Adagrad             48%             48%                  68%
                                      SGD                 48%             84%                  69%
Binary     Cross-Entropy,     also     known     as    Binary                 probability that the ith        sample  belongs to class 1.
Logarithmic Loss or Binary Cross-Entropy Loss, is a loss                      Binary  accuracy  is  a  metric  used  to  evaluate  the
function  used  primarily  in  binary  classification  tasks                  performance of a binary classification model in machine
within   machine   learning.      It   evaluates   how   well   a             learning.  It is calculated using following formula:
classification model predicts the possibility that an input                                                𝑛𝑜.  𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠
will belong to a particular class, usually represented by                        𝐵𝑖𝑛𝑎𝑟𝑦 𝑎𝑐𝑐𝑢𝑎𝑟𝑐𝑦       =     𝑡𝑜𝑡𝑎𝑙 𝑛𝑜.  𝑜𝑓 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠
the  number  1  in  the  model's  output,  which  is  a
probability  value  between  0  and  1.The  formula  for                      Which is mathematically expressed as follows:
Binary Cross-Entropy is as follows:                                                                                    𝑇𝑃  +  𝑇𝑁
𝐵𝑖𝑛𝑎𝑟𝑦 𝐶𝑟𝑜𝑠𝑠      −  𝐸𝑛𝑡𝑟𝑜𝑝𝑦                                                         𝐵𝑖𝑛𝑎𝑟𝑦 𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦        =   𝑇𝑃  +  𝐹𝑃  +  𝑇𝑁   +  𝐹𝑁
                            𝑁                                                 Where  TP :  True  Positive;  TN:  True  Negative;  FP  :
                     =   1 ∑[𝑦     log(𝑝   ) + (1  − 𝑦   ).log (1
                         𝑁        𝑖       𝑖             𝑖                     False Positive; FN: False Negative. From the  above
                           𝑖=1                                                table it is observed that RMSProp is performing better
                     −  𝑝𝑖)]                                                  than other optimizers with accuracy of  92%. Following
Where N is total number of  samples or instances,  𝑦𝑖                         screenshot1shows the confusion matrix and screenshot 2
represents  the truth label of the ith sample belongs to                      shows the overall summary of model with number of
class 1(heart disease) and          𝑝𝑖   represents  predicted                parameters generated during training.
                                                          Fig 3. Confusion Matrix
International Journal of Intelligent Systems and Applications in Engineering                               IJISAE, 2024, 12(12s), 280–287 |  285

                                                            Fig 4. Model Summary
                                 Fig 5. shows the comparison of all optimizers in terms of accuracy.
5. Conclusion                                                                        Prediction,      IEEE    International    conference    on
The  main  objective  of  this  research  was  to  predict                           Bioinformatics and Biomedicine (BIBM) (2019)
presence of    heart disease     or not in a given sample of                    [2]  Tzu-An S., Samadrita C., Fan Y. Heidi J., Georges
attributes using GNN.  For this the dataset which is used                            F.,   Quanzheng   L.,   Keith   J.,   Joyita   D.   Graph
is combination of four different datasets i.e Cleveland,                             Convolutional   neural   network   for   Alzheimer’s
Hungary,  Switzerland,  and  Long  Beach  V  created  in                             disease   classification.      IEEE   16th   international
1988. Various optimizers are used to improve accuracy                                symposium on biomedical imaging (ISBI 2019)
of  model  but  RMS-Prop  perform  better  than  other                          [3]  Sicen L., Tao L., Haoyang D., Buzhou T., Xiaolong
optimizers with accuracy        of 92%. We aim to evaluate                           W., Qingcai C., Jun Y., Yi Z., A hybrid method of
and  contrast  the  performance  of  eight  Graph  Neural                            recurrent neural network and graph neural network
Network   (GNN)   models—AGNN,   ChebNet,   GAT,                                     for next-period prescription prediction, International
GCN, GIN, GraphSAGE, SGC, and TAGCN—using the                                        journal of machine learning and cybernetics (2020).
public health heart disease dataset. Additionally, we plan
to compare their effectiveness among themselves and                             [4]  Amol K. Xue B., Luyang             L., Bryan P., Matt B.,
against traditional machine learning models like decision                            Martin   B.,   Shawn   O.,   Examining   COVID-19
tree,  gradient  boosting,  multi-layer  perceptron,  naive                          Forecasting  using  spatio-temporal  Graph  Neural
Bayes,  and  random  forest,  which  will  serve  as  our                            Network, arXiv:2007.03113v1 (2020)
baseline models for comparison.                                                 [5]  Tao  L.,  Weihua  P.,  Qingcai  C.,  Xioaolong  W.,
References                                                                           Buzhou T., KeoG: a knowledge-aware edge-oriented
                                                                                     graph  neural  network  for  documnet-level  relation
[1]  Bei W. , Xiaoqing L. , Jingwei Q. , Haowen S.,                                  extraction,     IEEE    international    conference    on
     Zehua P., Zhi T., GNDD: A Graph Neural Network-                                 Bioinformatics and biomedicine (2020)
     Based    Method    for    Drug-Disease    Association
International Journal of Intelligent Systems and Applications in Engineering                                  IJISAE, 2024, 12(12s), 280–287 |  286

[6]  Haifeng L., Hongfei L., Chen S., Liang Y., Yuan L.,                        [11]  K, V.; Singaraju, J. Decision Support System for
     Bo  X.,  Zhihao  Y.,  Jian  W.,  Yuanyuan  S.,  Drug                            Congenital Heart Disease Diagnosis based on Signs
     Repositioning  for  SARS-CoV-2  Based  on  Graph                                and   Symptoms   using   Neural   Networks. Int.   J.
     neural network,      IEEE International conference on                           Comput. Appl. 2011, 19, 6–12.
     Bioinformatics and Biomedicine (2020).                                     [12]  Nagavelli U, Samanta D, Chakraborty P. Machine
[7]  Waigi, R.; Choudhary, S.; Fulzele, P.; Mishra, G.                               Learning      Technology-Based      Heart      Disease
     Predicting the risk of heart disease using advanced                             Detection  Models.  J        Healthcare      Eng.  2022  Feb
     machine   learning   approach. Eur.   J.   Mol.   Clin.                         27;2022:7351061.     doi:     10.1155/2022/7351061.
     Med. 2020, 7, 1638–1645.                                                        PMID: 35265303; PMCID: PMC8898839.
[8]  Breiman,          L.          Random          forests. Mach.               [13]  Sogancioglu E., Murphy K., Calli E., Scholten E. T.,
     Learn. 2001, 45, 5–32.                                                          Schalekamp  S.,  Van  Ginneken  B.  Cardiomegaly
[9]  Chen, T.; Guestrin, C. XGBoost: A Scalable Tree                                 detection on chest radiographs: segmentation versus
     Boosting System. In Proceedings of the KDD ’16:                                 classification. IEEE
     22nd    ACM  SIGKDD  International  Conference  on                              Access . 2020;8 doi: 10.1109/access.2020.2995567.9
     Knowledge   Discovery   and   Data   Mining,                San                 4631
     Francisco,    CA,    USA,    13–17    August    2016;                      [14]  Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., &
     Association for Computing Machinery: New York,                                  Philip, S. Y. (2020). A         comprehensive survey on
     NY, USA, 2016; pp. 785–794.                                                     graph neural networks. IEEE transactions on neural
[10]  Gietzelt, M.; Wolf, K.-H.; Marschollek, M.; Haux,                              networks and learning systems, 32(1), 4-24.
     R.    Performance    comparison    of    accelerometer                     [15]  https://blogs.nvidia.com/blog/what-are-graph-
     calibration algorithms based on 3D-ellipsoid fitting                            neural-networks/ accessed on 10th Dec. 2023
     methods. Comput.            Methods            Programs                    [16]  Kriege,  N.  M.,  Johansson,  F.  D.,  &  Morris,  C.
     Biomed. 2013, 111, 62–71.                                                       (2020). A survey on graph kernels. Applied Network
                                                                                     Science, 5(1), 1-42.
International Journal of Intelligent Systems and Applications in Engineering                                  IJISAE, 2024, 12(12s), 280–287 |  287

