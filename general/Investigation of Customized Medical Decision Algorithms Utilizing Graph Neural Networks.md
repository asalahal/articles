         Investigation of Customized Medical Decision
          Algorithms Utilizing Graph Neural Networks
                                          1                 2              3                 4                5                6
                        Yafeng Yan         ,Shuyao He        ,Zhou Yu       ,Jiajie Yuan       ,Ziang Liu      ,Yan Chen
                           1Stevens Institute of Technology,USA,yanyafeng0105@gmail.com
                                2Northeastern University,USA,he.shuyao@northeastern.edu
                              3University of Illinois at Chicago,USA,zyu941112@gmail.com
                                      4Brandeis University,USA,jiajieyuan@brandeis.edu
                               5Carnegie Mellon University,USA,ziangliu@alumni.cmu.edu
                          6Stevens Institute of Technology,USA,yanchen@alumni.stevens.edu
    Abstract—   Aiming at the limitations of traditional medical                                        I.  INTRODUCTION
decision system in processing large-scale heterogeneous medical                    In today's medical and health field, how to effectively use
data and realizing highly personalized recommendation, this                    massive and complex medical data to provide             more accurate
paper   introduces    a  personalized  medical  decision  algorithm            and personalized medical decision support for each patient has
utilizing   graph    neural    network    (GNN).    This    research           become a key issue to be solved. Traditional medical decision           -
innovatively integrates graph neural network technology into                   making  system  often  relies  on  the  doctor's  experience
the medical and health field, aiming to build a high-precision
representation model of patient health status by mining the                    judgment and limited statistical analysis, it is difficult to f      ully
complex association between patients' clinical characteristics,                capture the huge differences among patients and the complex
genetic information, living habits.   In this study, medical data is           correlation among medical data. Therefore,           exploring a new
preprocessed  to  transform  it  into  a  graph  structure,  where             method that can deeply tap the potential value of medical data
nodes represent different data entities (such as patients, diseases,           and realize efficient personalized medical decision         -making is
genes, etc.) and edges represent interactions or relationships                 essential.
between entities.  The core of the algorithm is to design a novel                  Graph  Neural  Networks  [1]  (GNNs),  as  a  powerful
multi-scale fusion mechanism, combining the historical medical
records, physiological indicators and genetic characteristics of               machine learning tool, have received widespread attention for
patients, to dynamically adjust the attention allocation strategy              their excellent performance in processing non         -Euclidean data,
of the graph neural network, so as to achieve highly customized                especially   complex    network    structure    data.   Through
analysis of individual cases.   In the experimental part, this study           information dissemination and agg         regation operations on the
selected   several   publicly   available   medical   data   sets   for        graph structure, GNN can learn the deep feature representation
validation,   and   the   results   showed   that   compared   with            of  nodes  and  their  neighbors,  which  provides a  powerful
traditional machine learning methods and a single graph neural                 means  to  understand  the  complex  interactions  between
network  model,  the  proposed  personalized  medical  decision                various entities in the medical field (such as patients,       diseases,
algorithm showed significantly superior performance in terms                   drugs, genes, etc.)[2-5].
of disease prediction accuracy, treatment effect evaluation and
patient risk stratification.                                                       This paper aims       to explore and realize intelligent and
                                                                               personalized medical decisions through the following core
    Keywords—    graph     neural    network,     multi-scale    fusion        contents: First, we detail how to transform diverse medical
mechanism, medical atlas, attention mechanism                                  data (including but not limited to electronic medical records,
                                                                               genomic data, medical image reports           , lifestyle information,
                                                                               etc.)    into    a    unified    graphical    representation.    This

transformation requires not only the preservation of the rich                     computer         vision[13-15],3d-aware[16-17],         linguistic
information  of  the  original  data,  but  also  the  effective                  system[18-19], etc., but also shown great potential in the field
encoding of multiple associations between entities, such as the                   of medical health, especially in medical image recognition,
association of diseases and symptoms, and the association of                      disease association network analysis, etc. Its powerful graph
gene mutations and disease risk. Secondly, considering the                        structure    representation    learning    ability    brings    new
particularity of medical data, this study will explore and select                 perspectives and tools to medical decision          -making.
GNN models suitable for the characteristics of the medical                            With the popularity of the concept of precision medicine,
field, such as convolutional network (GCN) and Attention                          the research of personalized medical decision algorithms has
network [6-8] (GAT), etc.,          and optimize and adjust them                  become  a  focal  point.  These  algorithms  are  dedicated  to
according to the specific needs of medical decision-making.                       integrating patients' clinical parameters, genetic background,
Emphasis  is  placed  on  designing  model  structures  when                      lifestyle, and other dimensions of information, to tailor the
dealing with highly heterogeneous medical data. By designing                      diagnosis  and  treatment  plan  for  each  patient  through
a novel feature learning mechanism, this study aims to extract                    sophisticated  data  analysis.  Currently,  innovative  methods
high-dimensional  feature  representations  from  the  graph                      such as medical image analysis based on deep learning and
structure that reflect patients' personalized medical needs.                      personalized treatment path planning base           d on reinforcement
Combined  with  the  medical  knowledge  graph  and  deep                         learning  are  propelling  medical  decision  science  to  new
learning technology, algorithms will provide in-depth analysis                    heights. Notably, deep learning techniques have significantly
of  each  patient's  condition,  providing  customized  disease                   enhanced the capabilities of medical image segmentation, as
prediction, treatment recommendations, and risk assessment.                       demonstrated by Zi et al. (2024), who explored the application
    We will conduct extensive experiments based on real-                          of these methods in both segmentation and 3D reconstruction
world  medical  data  sets and  compare  them  with  existing                     of medical images[20].         By deeply integrating graph neural
methods to evaluate performance improvements in disease                           networks  with  personalized  medical  decision  algorithms,
diagnosis accuracy, treatment effect prediction, and patient                      scholars aim to overcome the limitations of current models
risk stratification. At the same time, the application potential                  and achieve fine modeling and prediction of individual health
of the algorithm in practical medical scenarios is discussed,                     conditions. This integration promises to unlock new potentials
including auxiliary clinical decision making and personalized                     in   precision   medicine,   fostering   more   accurate   and
health management plan making. To sum up, this study aims                         personalized healthcare solutions.
to  build  an  efficient  and  accurate  personalized  medical                        Among them, personalized medical decision algorithms
decision support system through the powerful capability of                        are an important research area, aiming to provide customized
graph neural network, promote the development of medical                          diagnosis and treatment plans for each patient based on their
care in a more intelligent and personalized direction, and                        individual characteristics and medical data. In this area, many
provide solid technical support for realizing the vision of                       approaches based       on deep learning have been proposed in
"precision medicine".                                                             recent years, alongside new technologies that utilize graph
                        II.  RELATED WORK                                         neural networks to process medical data. Traditional medical
    In the scientific realm of personalized medical decision               -      decision  algorithms  usually  adopt  statistical  methods  or
making, remarkable progress has been made in recent years,                        machine learning methods. These method              s, while performing
laying a solid foundation for improving the precision and                         well   in   some   situations,   struggle   to   capture   complex
personalized   level   of   health   care.   This   section   will                relationships and individual characteristics in medical data.
comprehensively examine the e           volution of medical decision              Therefore, more and more research has begun to explore new
algorithm,  the  innovation  breakthrough  of  graph  neural                      methods, such as deep learning-based medical image analysis
network  technology,  and  the cutting          -edge  research  results          and personalized treatment decision-making. Notably, Yan et
achieved  by  the  integration  of  the  two  in  the  field  of                  al.  (2024)  have  applied  neural  networks  to  enhance  the
personalized medical decision making.                                             accuracy of survival predictions across diverse cancer types,
                                                                                  illustrating  the  potential  of  these advanced  computational
    Evolution and breakthrough of medical decision algorithm:                     techniques  to  refine  and  personalize  medical  predictions
Medical  decision  algorithm,  as  the  product  of  the  cross                   further[21]. This research not only supports the use of deep
integration of mathematics, statistics and computer science, its                  learning in medical decision algorithms but also highlights its
core lies in the use of advanced algorithms to extract key                        significance in evolving personalized medicine practices by
information  from  the  complicated  medical  data  to  guide                     providing  more  nuanced  and  individualized  insights  into
medical practice[9-11]. In the early days, traditional methods                    patient outcomes.
based on statistics, such as logistic regression and support                          In the processing of medical data, graph neural networks
vector machines[12], played a fundamental role in disease                         have proved to have unique advantages in analyzing graph
prediction and medical resource manag             ement. However, the             structured data. It can effectively capture complex relational
limited ability of these methods to deal with highly complex,                     and structural information in medical data and provide more
non-linear    medical    data    relationships    and    to    mine               accurate  support  for  personalized  medical  decisions.  For
individualized  features  has  led  researchers  to  seek  more                   example,  graph  neural  networks  can  be  used  to  analyze
advanced analytical tools.                                                        medical image data, mining potential patterns and features in
    As  a  revolutionary  technology,  graph  neural  networks                    the images to provide doctors with more accurate diagnoses.
(GNNs) are unique in that they can learn directly on the graph                        To sum up, the research of personalized medical decision
data  structure,  and  achieve  a  deep  understanding  of  the                   algorithm is developing in a more refined and personalized
internal   structure   of   the   data   by   capturing   complex                 direction. As an emerging technology, graph neural network
interactions and information flo       ws between nodes. GNN has                  is expected to bring new breakthroughs and progress to the
not  only  made  remarkable  achievements  in  the  fields  of                    research and practice in this field. Therefore, the combination

of graph neural network and personalized medical decision                        performance of graph convolutional network (GCN) models.
algorithm will be a potential research direction in the future                   Among  them,  the  adaptive  graph  convolutional  network
medical  field.  The  development  of  personalized  medical                     (AGCN)  effectively  captures  and  models  the  underlying
decision algorithm is in a dynamic period, and the introduction                  structural information that was not explicitly expressed in the
of graph neural network brings a new research paradigm for                       graph by introducing the learning distance function and the
this field. The combination of the two not only heralds a move                   residual graph adjacency matrix. At the same time, double
towards a higher level of precision and personalization in                       graph convolutional Network (DGCN) is a new approach, and
medical decision-making.                                                         a unique double graph convolutional architecture is proposed.
                     III. THEORETICAL BASIS                                      DGCN consists of two sets of parallel graph convolution
                                                                                 layers with shared parameters, using a normalized adjacency
A.  Graph neural networks                                                        matrix   and  a   matrix   based   on   positive   point   mutual
    The spectral domain      -based graph neural network employs                 information. The positive point mutual information matrix
a method that utilizes spectral information from graphs for                      captures the co-occurrence information between nodes by
feature extraction and representation learning. It examines the                  means of random walk. The spectral domain-based graph
eigenvalues and eigenvectors of the graph's Laplacian matrix                     neural network utilizes spectral data from graphs for feature
to  understand  its  structure  and  features.  This  technique                  extraction  and  learning  representations.  It  analyzes  the
supports tasks like node and graph classification effectively.                   eigenvalues and eigenvectors of the graph’s Laplacian matrix
By leveraging this approach, it becomes possible to gain a                       to decode its structure and attributes. This method is effective
deeper understanding of the graph's inherent characteristics,                    for tasks such as node classification and graph classification.
which  aids  in  performing  precis       e  classification  tasks  that         Utilizing   this   approach   allows   for   a   more   profound
depend on spectral properties.                                                   comprehension   of   the   graph’s   intrinsic   characteristics,
                                                                                 facilitating accurate classification tasks reliant on spectral
    First with regard to the representation of graphs and the                    features.
Laplacian matrix, we assume that there is a          𝐺  =  (𝑉, 𝐸) where              Although spectral domain graph neural networks have a
𝑉 is  nodes  and     𝐸  is edges.  The  Laplace matrix         𝐿 can  be         solid theoretical foundation and show good performance in
defined as 𝐿   =   𝐷 −   𝐴.                                                      practical tasks, they also expose several significant limitations.
    For Laplacian matrix       𝐿, its eigenvalues and eigenvectors               Firstly, many spectral domain graph neural network methods
can be obtained by spectral decomposition. Let                λ   ≤  λ   ≤       need to decompose         Laplacian matrix to obtain eigenvalues
                                                                1     2          and eigenvectors in the implementation process, which often
⋯  ≤  λ𝑛  be  the  eigenvalue  of       𝐿  ,  and  the  corresponding            brings high computational complexity. Although ChebNet
eigenvectors are 𝑢1,     𝑢2, …  ,𝑢𝑛. These eigenvectors form the                 and GCN simplify this step to some extent, the entire graph is
spectral space of the graph      𝐺.                                              still required to be stored in memory during the calculation
    Graph convolution operations based on spectral domains                       process,  which  undoubtedly  consumes  a  lot  of  memory
can be expressed in the following form:                                          resources. Furthermore, the convolution process in spectral
                                                                                 domain  graph  neural  networks  typically  occurs  on  the
                                                                                 eigenvalue matrix of the Laplacian matrix, indicating that the
                                                                       (1)       convolution kernel parameters do not transfer easily across
                                                                                 different   graphs.  Consequently,   spectral   domain   neural
                                                                                 networks are generally constrained to a single graph, limiting
    𝐻(𝑙)is the nodal eigenmatrix of the        𝑙 layer.                          their ability to learn across multiple graphs and to generalize.
                                                                                 This limitation has led to fewer subsequent studies on spectral
    Interlayer propagation of Graph Convolutional Networks                       domain neural networks compared to those based in the spatial
(GCN) Interlayer propagation of GCN can be              expressed in the         domain.
following form:                                                                      The spatial domain-based image neural network adopts the
                                                                                 principles of traditional convolutional neural networks (CNNs)
                                                                        (2)      used in image processing. It expands the notion of convolution
                                                                                 to accommodate graph data structures and defines the graph
                                  −1     −1                                      convolution operation based on the spatial correlations among
    Among    them:        𝐴̂ =  𝐷   2𝐴𝐷    2  adjacency    matrix    is          nodes within the graph. As the pixels in Figure 1 constitute a
symmetric normalization.        𝐷̂ is the degree matrix of       𝐴̂.GCNs         two-dimensional grid structure, which can be regarded as a
can be trained for graph node classification tasks using the                     special form of topology diagram (see the left side of Figure
cross-entropy loss function      :                                               1). Similarly, when we apply the 3×3 convolution window on
                                                                                 the image, the spatial-space-based graph convolution also
                                                                       (3)       simulates a similar process on the graph data. It integrates the
                                                                                 feature sets of the central node and its neighboring nodes
    Where: 𝑦       is the label of whether the node 𝑖 belongs to                 through convolution. This process is graphically shown on the
                𝑖𝑘                                                               right side of Figure 1. The core principle of spatial graph
the class 𝑘. 𝑦̂𝑖𝑘 is the probability that the model predicts that                convolution  lies  in  the  propagation  of  node  features  and
the node 𝑖 belongs to the class 𝑘.           By optimizing the loss              topological information along the edge structure of the graph,
function, the weight parameters can be used to complete the                      which is similar to the feature extraction and propagation of
task of node classification.                                                     image data by CNN. In other words, spatial graph convolution
    While this is an overview of spectral domain based graph                     can achieve the iterative updating and fusion of graph node
neural  networks,  recent  research  efforts  have  turned  to                   features by simulating convolution behavior on graph data,
exploring  alternative  matrix  structures  to  optimize  the                    thus playing a vital role in the analysis.

                                                                                  network   to   extract   features   of   medical   images.  These
                                                                                  backbone networks extract features with different scales                   at
                                                                                  different levels through convolution operations.
                                                                                       The  middle  stage  is  the  proposal  of  the  side  fusion
                                                                                  framework (2015s). In this stage, Chen et al proposed the side
                                                                                  fusion framework (DSS), which fuses features from deeper
                                                                                  and shallow layers to enable the network to better obtain
    Fig. 1. Comparison of 2D convolution and graph convolution                    multi-scale feature inform       ation. New fusion strategies such as
    Neural Network for Graphs (NN4G) [22] represents the                          pyramid parsing module [25] (PPM) and cross feature module
inaugural  implementation  of  spatial-domain  graph  neural                      [26] (CFM) have been introduced into the multi               -scale fusion
networks in research. It employs a complex neural architecture                    mechanism. PPM achieves multi            -scale feature output through
with distinct parameters to model inter-graph relationships                       pyramid  pooling, while  CFM  provides  cascaded  feedback
and   to   propagate   information.   The   graph   convolution                   through  selective aggregation  of  multi          -layer  features,  thus
technique utilized by NN4G is essentially an aggregation of                       further improving the featur       e characterization capability of the
neighboring node information combined with the use of a                           network.
residual network to preserve the foundational data from the                            The latest stage is the exploration of multi            -level fusion
previous layer of the node. This operation is mathematically                      mode  (2019s till  now).  In  the  latest  research,  researchers
articulated as follows:                                                           began to explore the multi         -level fusion mode, designed the
                                                                                  fusion module to be recycled at each level, and constantly
                                                                         (4)      updated  the  features       to  make  it  gradually  refined[27        -28].
                                                                                  Therefore, the multi      -scale fusion mechanism has undergone
                                                                                  continuous evolution and improvement in the development of
    Where 𝑓(⋅) as the activation function, 𝒉(𝟎)            =  𝟎. From the         personalized medical decision algorithms. From the earliest
                                                      𝒗
point of view of mathematical expression, the whole modeling                      feature extraction based on backbone netwo               rk to the current
process is the same as GCN. NN4G differs from GCN in that                         multi-level  fusion  approach,  new  fusion  strategies  and
it uses an unnormalized adjacency matrix, which can result in                     technologies have been introduced constantly, providing more
very   large   differences   in   the   scale  of   potential   node              accurate and efficient support for medical image analysis and
information. The GraphSage[23] model was proposed to deal                         salient target detection.
with the problem of large number of neighbors of nodes, and                            At   present,   multi-scale   fusion  mechanism   plays   an
adopted the way of downsampling. Figure 2 illustrates how                         important  role  in  solving  personalized  medical  decision               -
the model incorporates a sequence of aggregation functions in                     making problems[29]. This mechanism aims to significance
graph convolution to maintain consistent output regardless of                     target detection by integrating feature information of different
node order. Specifically, GraphSage employs three symmetric                       scales, so as to provide mor      e accurate support for personalized
aggregation functions: mean aggregation, LSTM aggregation,                        medical decision making.
and pooling aggregation. The mathematical formulae utilized
by the GraphSage model are as follows:                                                 However, the features extracted from each convolutional
                                                                         (5)      layers  of  these  backbone  networks  have  different  scale
                                                                                  information.      The  multi-scale  fusion  mechanism  adopts  a
                                                                                  series of fusion strategies, as follows:
    Where 𝒉(𝒌)      =  𝒙   , agg   (⋅) is the aggregation function,
                𝒗        𝒗        𝑘                                                    1. Side Fusion Framework: This method fuses information
𝒮𝒩(𝑣) is 𝑣 neighbor nodes of a random sample. The proposal                        from  deeper  layers  with  shallow  features  through  short
of  GraphSage  has  brought  positive  significance  to  the                      connections, and then performs supervised learning on the
development of graph neural networks. Inductive learning                          merged features. This fusion strategy enables the network to
makes it easier to generalize graph neural networks, while                        obtain multi-scale feature       information better. Mathematically,
neighbor  sampling  leads  the  trend  of             large-scale  graph          fusion operations can be expressed as:
learning.
                                                                                                                                                            (6)
                                                                                       Where     ，   (𝐹fusion)     is  the  fused  feature,     (𝐹deep)    and
                                                                                  (𝐹shallow) are the features from the deeper and shallow layers,
                                                                                  respectively, and (α) is the fused weight.
                                                                                       2. Multilevel Fusion: In this way, the fusion module is
                                                                                  designed to be recycled at each level, and the features are
    Fig. 2. GraphSage sampling and aggregation diagram                            constantly updated iteratively to make them gradually refined.
B.   Classification algorithm                                                          3. Pyramid Parsing Module (PPM): PPM acquires feature
    The multi-scale fusion mechanism has undergone multiple                       outputs at various scales by implementing pyramid pooling at
stages  of  evolution  in  the  development  of  personalized                     multiple levels and subsequently concatenates these channels
medical decision algorithms, and new fusion strategies and                        to   enhance   the   network's   capability   to   gather   global
technologies are constantly introduced. In the early stage,                       information.  This  method  of           pyramidal  feature  fusion  is
multi-scale feature extrac      tion (2010s) is based on backbone                 particularly  effective  in  capturing  multi         -scale  information
network.  Researchers  mainly use  pre           -trained  classification         from images, thereby enhancing the efficacy of medical image
networks  [24]  (such  as  VGG,  ResNet,  etc.)  as  backbone                     analysis.

    4.  Cross  Feature  Module          (CFM)  :  CFM  selectively               mechanism is an important machine learning technique for
aggregates multi-layer features to form a cascade feedback                       models to automatically learn and focus on important parts of
decoder.                                                                         input data.
    The cohesive implementation of such fusion                 strategies
empowers  medical  image  analytics  and  prominent  target
identification models to optimally harness multi-resolution
feature  details,  thereby  furnishing  enhanced  precision  in
supporting tailored medical decision processes.
C.  Attention mechanism
    Attention mechanism [30] is a machine learning technique
used by models to automatically learn and focus on important
parts of input data. The attention mechanism finds broad
utility  across  numerous  domains,  encompassing  natural
language  processing  (NLP)  and  computer  vision, among
others. Within NLP, it assumes a pivotal role in tasks like
machine translation and text summarization. Meanwhile, in                            Fig. 3. Working mechanisms of the attention mechanism.
the field of computer vision, attention plays a significant role
in  processes  such  as  image  categorization  and  object                      IV. PERSONALIZED MEDICAL ALGORITHM BASED ON CNN-
recognition.                                                                                     MULTI-SCALE FUSION MECHANISM
    Fundamentally,  the  attention  mechanism  focuses  on                           In the study documented in this paper, the Multi-Scale
dynamically adjusting the model's emphasis according to the                      Fusion CNN (MSF-CNN) tailored for Personalized Medical
importance of elements in the input data. This allows the                        Decision Making incorporates a graph neural network and a
model to prioritize information that is more pertinent to the                    multi-scale fusion approach. This MSF-CNN model utilizes
specific task. As depicted in Figure 3, this capability enables                  convolutional neural networks at multiple scales to optimize
the model to assign different levels of importance to various                    medical   decision-making   on   an   individual   basis.   The
parts of the input data. Consequently, this attention-based                      integration  of  graph  neural networks with  the  multi-scale
strategy  makes  models  more  flexible  and  accurate  in                       fusion strategy in the MSF-CNN algorithm is intended to
processing    varied    inputs,    thereby    improving              their       enhance the extraction of detailed and comprehensive features
effectiveness and precision.                                                     from    medical   imagery,   facilitating   the   formulation   of
    Specifically,  the  attention  mechanism  is  applied  to                    customized healthcare strategies for patients.
different projection Spaces and different attention results are                      Firstly,   through  preprocessing   steps,  the   MSF-CNN
concatenated or weighted for summation. The mathematical                         algorithm segregates medical imaging data into two distinct
description is as follows:                                                       subsets:   a   training   cohort   and   a   validation   ensemble,
                                                                                 facilitating subsequent model instruction and evaluation., so
                                                                        (7)      as to conduct subsequent model training and evaluation. The
                                                                                 network architecture consists of multiple convolutional layers
                                               𝑄        𝐾       𝑉                and fusion modules, which are used to fuse features from
    which      (head𝑖    =  Attention(𝑄𝑊𝑖        ,𝐾𝑊𝑖    ,𝑉𝑊𝑖    ))     ，        different scales  to obtain  richer  and  more diverse  feature
(𝑊𝑄,𝑊𝐾,      𝑊𝑉) is the projection matrix, (𝑊𝑂) is the output                    representations.
   𝑖     𝑖     𝑖
of matrix projection matrix.                                                         After  the  feature  extraction  stage  is  completed,  the
    The self-attention mechanism is a specialized adaptation                     extracted multi-scale features are input into the graph neural
of  the  attention  concept,  particularly  designed  to  handle                 network for further processing. The purpose of graph neural
relationships within sequential data points. Here, both queries,                 networks is to capture the similarities and differences between
keys, and values originate from a single input sequence. While                   patients by using the relationships between nodes in the graph
maintaining similarity to the standard attention calculation,                    structure, so as to make better personalized medical decisions.
the distinction lies in using the identical sequence for these                   Subsequently, the MSF-CNN architecture undergoes training
three components.                                                                utilizing the allocated training dataset. This process employs
    During the training process, the self-attention component                    a loss function to quantify disparities between predictions and
becomes integral to the model, being optimized end-to-end.                       actual outcomes, with model parameters iteratively refined
By   iteratively   refining   the   model   parameters   via   the               through   optimization   methodologies,   including   gradient
minimization of the loss function, the model achieves peak                       descent. The detailed execution sequence unfolds as follows:
performance tailored to the given task. The backpropagation                          1. Data preprocessing and partitioning: First of all, medical
algorithm is usually used to update the parameters.                              image data is standardized and enhanced to ensure data quality
    The attention mechanism has achieved good performance                        and diversity. Then, the dataset is partitioned into (𝒟𝓉𝓇𝒶𝒾𝓃)
in many tasks, especially in processing sequence data and                        and   (𝒟𝓉ℯ𝓈𝓉)    according  to  the  predetermined  proportion.
image  data.  This  mechanism  adeptly  uncovers  intricate                      Where     (𝒟  =  𝒟𝓉𝓇𝒶𝒾𝓃    ∪  𝒟𝓉ℯ𝓈𝓉)     is   used   for   learning   and
connections within the data, thereby augmenting the model's                      performance verification of the model.
capacity for representation and generalizability. Moreover, the                      2. Multi-scale fusion feature extraction: The core of the
attention   mechanism   boasts   commendable   interpretive                      constructed MSF-CNN lies in its unique multi-scale fusion
qualities, enabling the visualization of attention weights to                    convolution layer structure. Convolution layer functions as
elucidate  the  model's  decision-making  rationale.           Attention         (𝑓(𝑙)(⋅)), of which the first (𝑙) said (𝑙) layer, fusion module

(𝑔(⋅))   can  integrate  the characteristics  of  different scales                      The ISIC dataset contains thousands of high                 -resolution
figure  (𝐹(𝑙))((𝑖    =  1,2,  …  ,𝑛  )，   (𝑛  )𝑎𝑠 𝑡ℎ𝑒 𝑠𝑐𝑎𝑙𝑒 𝑛𝑢𝑚𝑏𝑒𝑟),               dermoscopic  images  covering  multiple  disease  types  and
            𝑖                       𝑠        𝑠                                     different  clinical  situations.  This  allows  us  to  take  full
comprehensive features generated             (𝑭(𝒍+𝟏)   =   𝑔(∑𝑛𝑠     𝑭(𝒍))).
                                                                𝑖=1    𝒊           advantage of this data to train and test our model, thereby
In  this  way,  multi    -level  medical  image  information  from                 improving the generalization and performance of the model.
macro to micro can be captured.
    3. Application of graph neural network to personalized                              Before conducting experiments with ISIC datasets, we
analysis: After feature extraction, graph         neural network (GNN)             performed a series of data pre         -processing steps to ensure the
model, denoted as       (𝐻(⋅)), is used to model the correlation                   quality and consistency of the data, providing a good basis for
between patients. In GNN, nodes represent patients or image                        the training and testing of the model. The dataset utilizes the
regions, edges represent similarities or clinical associations                     Linked  Data[31]  methodology  to  consolidate  various  data
between  them, and  feature  updates  are  made  through  the                      formats,  enhancing  academic  research  by  facilitating  data
information       dissemination       mechanism               (𝐻(𝐴,    𝑋)  =       cross-referencing   and   boosting    interoperability    among
σ(𝐴𝑋𝑊))      .  Where    (𝐴)    is  an  adjacency  matrix,       (𝑋)   is  an      different datasets. This approach is particularly beneficial in
eigenmatrix,    (𝑊) represents a trainable weight matrix, while                    fields like machine learning and arti          ficial intelligence, where
                                                                                   high-quality  data  is  crucial  for  training  accurate  models.
the non  -linear activation function       (σ) is employed to discern              Linked Data also integrates diverse data sources, helping to
intricate and subtle variations among patient data.                                break down data silos and enabling a more comprehensive
    4.       Model       training       and       optimizatio    n:     ℒ  =       analysis approach. This enhanced data connectivity no                  t only
     𝑁                                                                             improves model accuracy but also aids researchers in gaining
− ∑𝑖=1   𝑦𝑖 log(𝑝𝑖)   , (𝑦𝑖 )  are  real  labels,   (𝑝𝑖)   is  a  model  to        deeper insights across multiple disciplines, such as healthcare
predict the probability of, Model parameters            (Θ  ) are adjusted         and social sciences. First of all, the image size is unified. Since
by  backpropagation  and  gradient  descent  (e.g.  Adam)  to                      the image sizes in the ISIC data set may not be consiste                nt, in
minimize losses      (ℒ).                                                          order to ensure that the images input to the model have the
    5. Performance evaluation: Evaluate the performance of                         same size, we adjust all images to a uniform size, usually by
MSF-CNN  on  an  independent  test  set              (𝒟      ),  with  key         cropping or scaling the image to the same size. Second, we
                                                        𝓉ℯ𝓈𝓉                       adjusted  the  brightness  and  contrast  of  the  image.  The
indicators including accuracy, sensitivity, specificity and OC-                    conditions under which dermoscopic images are taken vary,
ROC,  etc.,  by  comparing  the  agreement  of  the  model                         and there may be differences in brightness and contrast.                  We
prediction (𝑦̂   ) with the actual label        (𝑦). Comprehensively               use histogram equalization and other techniques to adjust the
measure  the  effectiveness  of  algorithms  in  personalized                      image, so that the brightness and contrast of the image are
medical decision making.                                                           more consistent. In addition,        the image is denoised. There may
    Ultimately, the efficacy of the trained model is assessed                      be various noises in dermoscopic images, such as background
via examination on a distinct test dataset. Its performance in                     noise introduced by equipment or motion blur during image
tailored medical decision-making is gauged by contrasting the                      acquisition. In order to reduce the interference of noise to the
discrepancies between the model's predicted outcomes and the                       model,  we  use  filtering  technology  to d           enoise  the  image     .
actual   classifications.   Through   mathematically   rigorous                    Finally, the image is normalized. Normalization maps the
framework    design,    MSF-CNN    algorithm    effectively                        pixel values of an image to a fixed range, such as                  [0, 1] or
combines the expressive power of deep learning with the                            [−1,   1], to better match the input requirements of the model.
correlation  analysis  of  graph  models  to  provide  a  more                     This helps to improve model stability and training speed, and
accurate and comprehensive solution for personalized medical                       helps to avoid problems such as gradient explosion or gradient
decision-making, and can fully leverage the benefits of deep                       disappearance. Through the above data preprocessing steps,
learning and graph neural network to extract more accurate                         we can obtain dat      a with higher quality and better consistency,
and  rich  features  and  provide  more  accurate  support  for                    and provide a more reliable basis for the training and testing
personalized medical decision-making.                                              of the model.
                   V.   EXPERIMENTAL ANALYSIS                                      B.   Evaluation indicators
A.  Data set                                                                            In the experiment of this paper, we used ISIC[32] dataset
                                                                                   to evaluate     image-text matching tasks, and adopted several
    We chose ISIC (International Skin Imaging              Collaboration),         common evaluation indicators to measure Recall, Precision
a well-known foreign skin disease detection data set, as our                       and mAP.
experimental data set. The ISIC dataset is a large dataset                              Within  the  context  of  medical  decision             -making,  the
dedicated to skin image analysis, containing thousands of skin                     model's capability to discern positive instances is assessed
microscopy images from around the world, covering many                             using       two       metrics:       Recall,       which       evaluates
different types of skin cases. This dataset has the following                      comprehensiveness in identifying every actual positive case,
advantages: The ISIC dataset covers many different types of                        and Precision, focusing on the proportio              n of true positives
skin cases, including but not limited to melanoma, squamous                        among all instances predicted as positive:
cell carcinoma, basal cell carcinoma, etc. This allows our
model to be tested and evaluated in a wider range of situations;
The  images  in  the  ISIC  dataset  have been annotated  and                                                                                                 (8)
verified by professional doctors, ensuring the accuracy and
credibility  of  the  data.  Each  image  is  accompanied  by  a
detailed    case    description    and    patholo       gical    diagnosis                                                                                    (9)
information, which provides an important reference for the
training and evaluation of the model.

    Correctly identified positive        instances are quantified as             respectively.  Although ResNet  model  also achieved  good
True  Positives,  whereas  False  Negatives  denote  positive                    performance, but compared with our MSF-CNN model, there
scenarios inaccurately classified. Conversely, False Positives                   is a certain performance gap. The Precision, Recall and mAP
represent instances erroneously labeled as positive despite                      indexes of the traditional machine learning algorithm SVM
being negative.                                                                  reach 90.36%, 92.89% and 93.96% respectively. Although the
    Mean Average Precision (mAP): The mAP is utilized as                         SVM model is slightly inferior to the deep learning model in
an  indicator  to  evaluate  the  effectiveness  of  a  model  in                performance, it still shows relatively good performance.
classification or detection tasks spanning various categories.                       In summary, the comparison of experimental outcomes
It integrates aspects of Precision and Recall while addressing                   indicates  that  the  MSF-CNN  model  offers  considerable
the equilibrium among distinct classes. In the context of this                   benefits  in  detecting  skin  diseases.  This  underscores  the
study, we compute the Average Precision for each category                        effectiveness  and  reliability  of our  personalized  medicine
within the model and subsequently average these values to                        approach, which employs a multi-scale fusion convolutional
determine the overall mAP:                                                       neural  network.  Such  results  provide  robust  support  and
                                                                                 direction   for   advancing  personalized   medical  decision-
                                                                      (10)       making strategies in the future.
    Where N is the total number of classes and                      is the        TABLE I.             EXPERIMENTAL RESULTS AT DIFFERENT        BASELINES
Average Precision of the I-th class.                  is the area value            Model              Precision           Recall            mAP
under the accuracy-recall curve for that category. Through the                     MSF-CNN            95.21               96.74             97.29
comprehensive consideration of these evaluation indicators,                        ResNet             91.94               94.12             94.13
                                                                                   SVM                90.36               92.89             93.96
we can comprehensively evaluate the algorithm's decision-
making  ability  on  the  ISIC  data  set,  and  then  verify  its
effectiveness and feasibility.                                                   E.  Ablation experiment
C.  Experimental setup                                                               In   this   study,   we   performed   a   series   of   ablation
    A range of experiments were carried out and validated                        experiments, which included comparing the MSF               -CNN model
using the ISIC dataset. The experimental setup is outlined as                    with the normal CNN model. The following is a detailed
follows: The ISIC dataset is divided into training and testing                   analysis of the ablation experiment: Our MSF           -CNN model has
segments at a ratio of 80:20, where 80% is utilized for training                 achieved   excellent   performance   in   the            dermatological
the model and 20% is set aside for validation purposes. Further,                 detection task. Specifically, the Precision, Recall and mAP
we applied a 5-fold cross-validation method solely within the                    indexes of the model reach 95.21%, 96.74% and 97.29%,
training segment to ensure thorough assessment of the model's                    respectively.   This   shows   that   the   multi         -scale   fusion
performance.  The  architecture  of  the  MSF-CNN  model                         convolutional  neural  network  we  designed  has  significant
includes four convolutional layers and two pooling layers. For                   advantages in de     rmatological detection tasks, recall rate and
integration, the model employs a weighted average fusion                         overall average accuracy. Compared with MSF                -CNN model,
technique, setting the fusion weights to [0.6,0.4].Regarding                     the  performance  of  ordinary CNN  model  is  slightly  less.
training specifics, the initial learning rate is set at 0.001, and               Specifically, the Precision, Recall and mAP indexes of the
is  progressively  decreased  by  a  factor  of  10  (to  0.0001)                model  reach  91.32%,  90.86%  and  88.21%,  r               espectively.
throughout the training phase to facilitate stable convergence.                  Compared to the MSF          -CNN model, the conventional CNN
The  model  undergoes  training  over  100  epochs  using  a                     model  shows  reduced  performance  in  metrics  such  as
stochastic gradient descent (SGD) optimizer with a batch size                    accuracy, recall, and a general decline in mean accuracy. In
of 32. Evaluation of the model is conducted using established                    conclusion, the outcomes of ablation studies reinforce the
metrics such as Precision, Recall, and Map.                                      efficacy    and    advanta     ge    of    our    multi-scale    fusion
                                                                                 convolutional neural network (MSF           -CNN) in tasks related to
    Through   the   above   experimental   Settings,   we   can                  dermatological  detection.  Compared  to  the  normal  CNN
comprehensively  evaluate  the  performance  and  effect  of                     model,   our   MSF-CNN   model  has  achieved  significant
MSF-CNN algorithm in dermatological detection tasks, and                         improvements   in   accuracy,   recall   and   overall   average
provide more accurate and reliable support for personalized                      accuracy, providing more reliable and accurate support for
medical decisions.                                                               personalized medical decisions.
D.  Experimental result                                                                            TABLE II.            ABLATION RESULTS
    In Figure 1, we designed the MSF-CNN algorithm for skin                        Model             Precision            Recall            mAP
disease detection tasks, and compared its performance. The                         MSF-CNN           95.21                96.74             97.29
following   is   a   detailed   analysis   of   the   comparative                  CNN               91.32                90.86             88.21
experimental  results:  Our  MSF-CNN  model has achieved                                                   VI.  CONCLUSION
excellent  performance  in  dermatological  detection  tasks.
Specifically, our model achieves 95.21%, 96.74% and 97.29%                           This paper introduces the MSF          -CNN (Multi-Scale Fusion
in Precision, Recall and mAP (mean precision), respectively.                     Convolutional  Neural  Network  for  Personalized  Medical
This shows that our MSF-CNN model achieves excellent                             Decision  Making)  algorithm,  which  achieves  remarkable
performance  in  both  accuracy  and  recall  rates,  and  also                  advancements in personalized healthcare decision support by
performs well in overall average accuracy. In contrast, the                      ingeniously  integrating  multi       -scale  feature  fusion  within
ResNet model based on deep learning achieved 91.94%, 94.12%                      convolutional  networks  and  the  prowess  of  graph  neural
and   94.13%   in   Precision,   Recall   and   mAP   indexes,                   networks.   The   synergy   of   these   techniques   bolsters

performance significantl. The primary contributions of this                                         multimodal data fusion in the artificial intelligence era. arXiv preprint
study are outlined as follows: multi-scale fusion strategy is                                       arXiv:2404.12278.
used to effectively enhance the ability to capture medical                                   [13]   Atulya Shree, Kai Jia, Zhiyao Xiong, Siu Fai Chow, Raymond Phan,
image features, not only to identify macro structural features,                                     Panfeng Li, & Domenico Curro. (2022). Image analysis.US Patent
but also to      dig deep into micro details, which significantly                                   App. 17/638,773
improves the recognition accuracy of the model for complex                                   [14]   Chen, Y., Ye, X., & Zhang, Q. (2021). Variational model-based deep
                                                                                                    neural networks for image reconstruction. Handbook of Mathematical
pathological changes; The integrated graph neural network                                           Models   and   Algorithms   in   Computer   Vision   and   Imaging:
component  effectively  utilizes  the  intrinsic  relationship                                      Mathematical Imaging and Vision, 1-29.
between patients or cases, and enhances the model's ability to                               [15]   Zhicheng Ding, Panfeng Li, Qikai Yang, Siyang Li, & Qingtian Gong
understand the similarities and differences of cases through                                        (2024).   Regional   Style   and   Color   Transfer.   arXiv   preprint
graph   structure   analysis,   which   is   essential   for   the                                  arXiv:2404.13880.
development  of  highly  personalized  treatment  plans.  The                                [16]   Zhang, Y., Ji, P., Wang, A., Mei, J., Kortylewski, A., & Yuille, A.
experimental results showed that MSF-CNN exceeded the                                               (2023). 3d-aware neural body fitting for occlusion robust 3d human
                                                                                                    pose  estimation.  In  Proceedings  of  the  IEEE/CVF  International
existing methods in many evaluation indicators, especially in                                       Conference on Computer Vision (pp. 9399-9410).
the accuracy, sensitivity and specificity, which verified the                                [17]   Z. Zhao, H. Yu, C. Lyu, P. Ji, X. Yang and W. Yang, ""Cross-Modal
effectiveness and reliability of the model in assisting doctors                                     2D-3D Localization with Single-Modal Query,"" IGARSS 2023 - 2023
to develop personalized medical programs. This study not                                            IEEE  International  Geoscience  and  Remote  Sensing  Symposium,
only promotes the fusion application of deep learning and                                           Pasadena,       CA,       USA,       2023,       pp.       6171-6174,       doi:
graph theory methods in medical image analysis in theory, but                                       10.1109/IGARSS52108.2023.10282358.
also shows strong application potential in practice, laying a                                [18]   Panfeng Li, Qikai Yang, Xieming Geng, Wenjing Zhou, Zhicheng
solid foundation for the future development of personalized                                         Ding,  &  Yi  Nian  (2024).  Exploring  Diverse  Methods  in  Visual
                                                                                                    Question Answering. arXiv preprint arXiv:2404.13565.
medicine. The final conclusion is that MSF-CNN algorithm,                                    [19]   Li, P., Abouelenien, M., & Mihalcea, R. (2023). Deception Detection
with its unique design ideas and excellent performance, has                                         from  Linguistic  and  Physiological  Data  Streams  Using  Bimodal
proved its great value in the field of personalized medical                                         Convolutional Neural Networks. arXiv preprint arXiv:2311.10944.
decision-making.  It  not  only  improves  the  accuracy  of                                 [20]   Zi, Y., Wang, Q., Gao, Z., Cheng, X., & Mei, T. (2024). Research on
diagnosis and treatment recommendations, but also provides                                          the Application of Deep Learning in Medical Image Segmentation and
advanced technical support to promote more personalized,                                            3D Reconstruction. Academic Journal of Science and Technology,
efficient and safe medical care.                                                                    10(2), 8-12.
                                                                                             [21]   Yan, X., Wang, W., Xiao, M., Li, Y., & Gao, M. (2024). Survival
                                  REFERENCES                                                        Prediction  Across  Diverse  Cancer  Types  Using  Neural  Networks.
                                                                                                    arXiv preprint arXiv:2404.08713.
[1]   Scarselli, Franco, et al. "The graph neural network model." IEEE                       [22]   Micheli,   Alessio.   "Neural   network   for   graphs:   A   contextual
      transactions on neural networks 20.1 (2008): 61-80.                                           constructive approach." IEEE Transactions on Neural Networks 20.3
[2]   Wantlin, K., Wu, C., Huang, S. C., Banerjee, O., Dadabhoy, F., Mehta,                         (2009): 498-511.
      V. V., ... & Rajpurkar, P. (2023). Benchmd: A benchmark for modality-                  [23]   Oh, Jihun, Kyunghyun Cho, and Joan Bruna. "Advancing graphsage
      agnostic  learning  on  medical  images  and  sensors.  arXiv  preprint                       with a data-driven node sampling." arXiv preprint arXiv:1904.12935
      arXiv:2304.08486.                                                                             (2019).
[3]   Xiao, M., Li, Y., Yan, X., Gao, M., & Wang, W. (2024). Convolutional                   [24]   Koonce,  Brett,  and  Brett  Koonce.  "Vgg  network."  Convolutional
      neural network classification of cancer cytopathology images: taking                          Neural Networks with Swift for Tensorflow: Image Recognition and
      breast cancer as an example. arXiv preprint arXiv:2404.08279.                                 Dataset Categorization (2021): 35-50.
[4]   Chen, Y., Liu, C., Huang, W., Cheng, S., Arcucci, R., & Xiong, Z.                      [25]   Zhao,   Hengshuang,   et   al.   "Pyramid   scene   parsing   network."
      (2023).  Generative  text-guided  3d  vision-language  pretraining  for                       Proceedings of the IEEE conference on computer vision and pattern
      unified medical image segmentation. arXiv preprint arXiv:2306.04811.                          recognition. 2017.
[5]   Dai, W., Tao, J., Yan, X., Feng, Z., & Chen, J. (2023, November).                      [26]   Wu, Xiao, et al. "Dynamic cross feature fusion for remote sensing
      Addressing Unintended Bias in Toxicity Detection: An LSTM and                                 pansharpening."    Proceedings    of    the    IEEE/CVF    International
      Attention-Based Approach. In 2023 5th International Conference on                             Conference on Computer Vision. 2021.
      Artificial Intelligence and Computer Applications (ICAICA) (pp. 375-                   [27]   Wang, X. S., & Mann, B. P. (2020). Attractor Selection in Nonlinear
      379). IEEE.                                                                                   Energy  Harvesting  Using  Deep  Reinforcement  Learning.  arXiv
[6]   Veličković, Petar, et al. "Graph attention networks." arXiv preprint                          preprint arXiv:2010.01255.
      arXiv:1710.10903 (2017).                                                               [28]   Restrepo, D., Wu, C., Vásquez-Venegas, C., Nakayama, L. F., Celi, L.
[7]   Jin, J., Xu, H., Ji, P., & Leng, B. (2022, October). IMC-NET: Learning                        A., & López, D. M. (2024). DF-DM: A foundational process model for
      Implicit   Field   with   Corner   Attention   Network   for   3D   Shape                     multimodal data fusion in the artificial intelligence era. arXiv preprint
      Reconstruction.  In  2022  IEEE  International         Conference  on  Image                  arXiv:2404.12278.
      Processing (ICIP) (pp. 1591-1595). IEEE."                                              [29]   Wang, X. S., Turner, J. D., & Mann, B. P. (2021). Constrained attractor
[8]   Yao,  J.,  Wu,  T.,  &  Zhang,  X.  (2023).  Improving  depth  gradient                       selection using deep reinforcement learning. Journal of Vibration and
      continuity in transformers: A comparative study on monocular depth                            Control, 27(5-6), 502-514.
      estimation with cnn. arXiv preprint arXiv:2308.08333.                                  [30]   Niu, Zhaoyang, Guoqiang Zhong, and Hui Yu. "A review on the
[9]   Xu, T., Li, I., Zhan, Q., Hu, Y., & Yang, H. (2024). Research on                              attention mechanism of deep learning." Neurocomputing 452 (2021):
      Intelligent   System   of   Multimodal   Deep   Learning   in   Image                         48-62.
      Recognition.  Journal  of  Computing  and  Electronic  Information                     [31]   Li, Y., Yan, X., Xiao, M., Wang, W., & Zhang, F. (2024). Investigation
      Management, 12(3), 79-83.                                                                     of Creating Accessibility Linked Data Based on Publicly Available
[10]  Bian, W., Chen, Y., Ye, X., & Zhang, Q. (2021). An optimization-                              Accessibility Datasets. In Proceedings of the 2023 13th International
      based meta-learning model for mri reconstruction with diverse dataset.                        Conference on Communication and Network Security                  (pp. 77–81).
      Journal of Imaging, 7(11), 231.                                                               Association for Computing Machinery.
[11]  Zhang,  Q.,  Ye,  X.,  &  Chen,  Y.  (2021,  September).  Nonsmooth                    [32]   Cassidy, B., Kendrick, C., Brodzicki, A., Jaworek-Korjakowska, J., &
      nonconvex LDCT image reconstruction via learned descent algorithm.                            Yap, M. H. (2022). Analysis of the ISIC image datasets: Usage,
      In Developments in X-Ray Tomography XIII (Vol. 11840, p. 132).                                benchmarks   and   recommendations. Medical   image   analysis, 75,
      SPIE.                                                                                         102305.
[12]  Restrepo, D., Wu, C., Vásquez-Venegas, C., Nakayama, L. F., Celi, L.
      A., & López, D. M. (2024). DF-DM: A foundational process model for

