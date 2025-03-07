Classi cation of Chronic Kidney Disease Based on
Gfr in Internet of Medical Things Environment Using
Graph Neural Network Based Deep Q Learning
(GNN-DQL)
B Prasad Reddy Tatiparti  (  tatipartibprasadreddy@gmail.com )
 Vellore Institute of Technology
Vydeki Dharmar
 Vellore Institute of Technology
Research Article
Keywords: Deep learning, GNN-DQL, GFR, Classi cation, IoMT, Optimal features, Parameter optimization,
AMO
Posted Date: May 13th, 2022
DOI: https://doi.org/10.21203/rs.3.rs-1625089/v1
License:       This work is licensed under a Creative Commons Attribution 4.0 International License.
Read Full License

      CLASSIFICATION OF CHRONIC KIDNEY DISEASE BASED ON GFR IN
  INTERNET OF MEDICAL THINGS ENVIRONMENT USING GRAPH NEURAL
                   NETWORK BASED DEEP Q LEARNING (GNN-DQL)
                          Tatiparti B Prasad Reddy1*, Vydeki Dharmar2
   1 Research Scholar, School of Electronics Engineering, Vellore Institute of Technology,
                                               Chennai, India
  2 Associate Professor, School of Electronics Engineering, Vellore Institute of Technology
                                               Chennai, India
                                  1*tatipartibprasadreddy@gmail.com
Abstract
The identification of Chronic kidney (CK) disease in the medical field is still acts as a
challenging  scenario  in  the  recent  years.  Precise  recognition  of  CK  disease  possess  a
significant  aspect  in  rendering  effective  treatment  to  the  patients.  Various  forms  of
approaches have been developed for the exact classification of CK disease, but still there
emerges certain forms of demerits including, improper selection of features, necessity of high
storage space, requirement of effective learning model, less accurate, high complexities with
respect to time and cost. The presence of these drawbacks adversely decreases the overall
model performance. Hence to overcome these complexities, a graph neural network based
deep Q learning (GNN-DQL) approach is proposed for the effective classification of five
different stages like normal, mild, moderate, severe and end. Initially, the data are gathered
from different people with the help of biomedical sensors through Internet of medical things
(IoMT). The  data are  pre-processed through Handling  Missing  Values,  Categorical Data
Encoding, Data Transformation and Outlier Detection to eradicate the unwanted distortions.
The Glomerular Filtration rate (GFR) is calculated with respect to age and serum creatinine
level. Then, GNN-DQL technique is adopted for enhancing the classification accuracy. The
parameters  are  optimized  through  Adaptive  Mayfly  Optimization  (AMO)  method.  The
classification performance is analysed with respect to accuracy, precision, recall, specificity,
F1 score, confusion matrix and so on using PYTHON simulation tool. The classification
accuracy  of 99.93% is attained in  the CK  disease  classification of  five  different  stages
regarding the collected data.
Keywords:  Deep  learning,  GNN-DQL,  GFR,  Classification,  IoMT,  Optimal  features,
Parameter optimization, AMO.
1.  Introduction
    The CK disease is a severe illness all over the world, mainly in developing countries [1].
Timely identification of chronic diseases and also tracking of risk factors slow down the
continuation of diseases and may exclude dangerous events in the everyday patient’s life. An
unidentified  CK   disease  results  in   various   problems  that  put   patients  in   high-risk
circumstances [2]. A quick creation of renal disease in patients with hypertension may cause

worse events. CK disease is a heterogeneous breakdown that mainly affects kidney function
and structure. The major reasons of the CK disease are heart disease, diabetes, and HB (high
blood) pressure [3]. Diabetes is also known as blood sugar that can harm the blood vessels in
human kidney. HB pressure also can harm the blood vessels in human kidney [4]. Heart
disease is also related with kidney disease which is severe than all other disease.
         Symptoms of the CK disease are chest pain, feeling tired, loss of appetite, shortness of
breath, weight loss, muscle cramps, trouble concentrating and sleep problems [5]. Human
body temperature placed a vital role to the well-being of the patients. The patient’s health can
be routinely monitored to reduce the risk of their life. Hence, biomedical sensors have the
capability to handle medicine. The usage of the biomedical sensors for detecting the early-
stages of CK disease was identified by comparing breath samples of patients to analyse the
breath samples of healthy controls [6]. A typical BM (Biomedical sensor) device consists of
small  battery-operated  board  with  a  memory,  a  microprocessor,  and  a  radio  frequency
transceiver. DM (data mining) is a process by combining information from biomedical sensor
devices, storing information, and successfully send information to the patients [7].
         Some of the common features of DM process are flexibility, robustness, trade-off
energy efficiency, etc.  The  simple reasoning of health identifying device in biomedical
sensor readings like calculating the sleep hours or the number of processes each day to the
better levels of information processing in order to provide correct data to the patients [8].
Nowadays, healthcare services have mainly focused on deeper DM tasks to provide better
services for the well beings of human [9]. The DM approach consists model data learning,
extraction and identification, and information pre-processing [10]. Various features including
meta-data and expert knowledge to identify the operations such as detection, prediction, and
DM (decision making). Pre-processing techniques helps to filter unusual information from
the input data to remove high frequency noise [11].
         The main purpose of extraction is to identify the features of an information set which
are  characteristics  of  the  original  information.  ML  (machine  learning)  models  can  be
classified into unsupervised and supervised methods, also called descriptive and predictive,
respectively [12,13]. Supervised learning methods, including face recognizers over images,
medical diagnosis systems and spam classifiers of e-mail for patients, where the training
information can be taken the collection of (u, v) pairs, the prediction “u” and a query “v”
[14]. Several supervised methods include support vector machines, kernel machines, decision
trees, logistic regression, decision forests, Bayesian classifiers and neural networks. ML can
achieve significant task, but it still falls short of replicating HI (human intelligence) [15].
These drawbacks can be rectified by using DL (deep learning) which is a subcategory of ML.
         DL is evaluated by the way of classifying, clustering, and predicting things by using a
NN (neural network) that has been trained on huge amounts of information [16]. It has its
roots in NN that consists of algorithms, modelled loosely after the brain of human, that are
mainly designed to find patterns. The main aim of DL models is to bring together impactful
and  novel  research  work  on  deep  learning  for  medicine  based  on  the  IoMT,  thereby
expediting research in this field [17,18]. BMs are used all over the world, which used to
measure the  blood level and  help patients  cope  with diabetes [19,20].  Such  sensors are
specifically implanted under the skin. It offers continuous monitoring and measurement of

blood levels in patients.  In order to overcome the above-mentioned ML problems, a new
deep learning technique is proposed in this research.
A novel methodology is adopted in the proposed research work for the precise classification
of five various stages of CK disease. Some of the prominent contributions for enhancing the
performance of classification accuracy are given as follows.
       To generate the CK disease data throughBMs on several people and thereby gathering
         the data using IoMT.
       To  initiate  data  pre-processing  by  Handling  Missing  Values,  Categorical  Data
         Encoding, Data Transformation and Outlier Detection for eradicating the unwanted
         distortions.
       To  classify  different  stages  of  CK  disease  through  the  evaluation  of  glomerular
         filtration rate based on age and serum creatinine level.
       The generated data are precisely classified using Graph neural network with deep Q-
         learning technique (GNN- DQL) classifier.
       The   parameters  are   optimized   through   Adaptive   mayfly   optimization  (AMO)
         approach for precise classification outcome.
The structure of the proposed research work is organized in to various sections. Section 2
describes the literature survey of CK disease classification done by various researchers in
learning methodologies. Section 3relates with the description of proposed methodology in
CK disease classification. The simulations implemented for determining the performance
outcome of proposed method are discussed in Section 4 using PYTHON simulation tool.
Finally, the conclusion and future workof the proposed research is delivered under Section 5.
2.  Related works
Based upon the CK disease classification, most of the researchers have undertaken several
techniquesto  achieve  precise  outcomes.  Some of  the  significant  classification  techniques
adopted by several authors are surveyed below.
Singh et al. [21] performed a deep NN for early prediction and detection of CK disease. HB
(high blood) pressure and diabetes are the common reasons of CK disease. The patients with
CK disease have a higher possibility of dying young age. This project presented a novel DL
approach for the early prediction and detection of CK disease. This model created a deep NN
and compared its operation to that of other ML models. The NN optimum devices were
attached by running multiple trials and building the parameters. The characteristics were
determined by RF (recursive feature) elimination process. Specific gravity, cell count of red
blood,  Haemoglobin,   serum  creatinine,   packed   cell   volume,  and  hypertension   were
considered as important structures in the RF elimination. The deep NN model outperformed
the  other classifiers Logistic  regression,  KN (K-nearest) neighbour,  SV (support  vector)
machine, NB (naive bayes) and Random Forest classifier to achieve the 100% precision. It
used for nephrologists in detecting CK disease. However, there is limitations on how much
information required to train the correct model to calculate about deep structures.
Liao  et  al.  [22]   identified    data  augmentation  with  generative  adversarial  networks  by
improving stage classification of CK Disease. To identify the stage classification of CK

disease, this model provides an auxiliary diagnosis system with DL technique for renal ultra-
sound images.  It used the MobileNetV2 pre-training model and ACWGAN-GP model. The
generated images by the original images and the ACWGAN-GP technique were concurrently
input to the MobileNetV2 pre training technique for better training. The technique evaluated
a precision of 81.9% in the four phases of CK disease classification. The forecast outputs
permitted a bigger stage acceptance, then the accuracy could be achieved by up to 90.1%. DL
method solved the problem of inadequate data samples and imbalance during training model
for  an  automatic  process.  This  model  enhanced  the  prediction  of  CK  disease  diagnosis
process. This  technique  requires a  hugeamount of training information  to accelerate the
forecastaccuracy-based classifier.
Sabanayagam  et  al.  [23]  implemented  a  DL  technique  to  identify  CK  disease  from
photographs  of retina in  community-based  persons. The  information related  from  cross-
sectional, multi-ethnic studies in China and Singapore. SEED (Singapore-epidemiology-eye-
diseases) process worked to progress (5188 persons)and validate (1297 persons) using DL
technique. There are three models were trained: Image DL technique, RF (risk factor) and
hybrid DL technique combined with RF and image. DL technique were evaluated usingthe
receiver   characteristics.   It   used   CNN   (convolutional-neural-network)   based   DL   for
CKdisease  from  retinal  images.  It  has  many  challenges  that  it  does  nothave  data  on
albuminuria  for  every  patient,  so  it  cannot  perform  an  albuminuria  usingprediction  DL
approach. It does not know what characteristics were used bythe DL technique to obtain CK
disease because of heat-maps specified changes of vessel andun-usual lesions.
Navaneeth et al. [24] developed a dynamic pooling using CNN model to detect CK disease.
This model analysed        the attention of urea in the saliva sample to identify the disease of
patients. Novel discovery and DL model were used to find the disease using saliva samples
from patients. Hybrid DL model consists of SVM (support vector classifier) and CNN which
helped to rectify the difficulties obtained by the CDC (convolutional data classification)
technique. It analysed the urea concentration in the sample of saliva to identify the illness of
patients. This technique obtained sensitivity (97.5%), specificity (97.83%) and an accuracy
(97.67%). The drawback in this model that the evaluation was continued ten more times until
the total information set was calculated.
Kriplani et al. [25] analysed the prediction of CK diseases using deep artificial NN model.
This model analysed        224 documents of CK illness maintained on the UCI (University of
California, Irvine) ML repository named CK diseases dating back to 2015. The deep NN
predicted the appearance (or) non-appearance of CK disease with a precision of 97%. The
automatic CK disease process helped to minimize the damage of kidney, but for the detection
of CK disease at starting stage was essential to attain the better output. Deep NN was made of
number of layers that were built with number of neurons. The cost function is identified using
convex shape. The chronic and non-chronic diseases were obtained and compared with other
classification models. Deep NN achieved 97.7679 % accuracy that was the perfect prediction
of the CK  disease. The limitation  was  that, Deep NN  model required large amount  of
information to analyse the better output. Table 1 signifies the analysis of various existing
approaches with its respective merits and demerits.

                       Table 1: Analysis of existing classification approaches
 Author         Technique         Objective        Merits       Demerits       Performance         Dataset
name and             used                                                            (%)             used
Reference
 Singh et        Deep NN            Early         Improve         Lower           Precision-         UCI-
  al. [21]                        prediction          d           model              100             CKD
                                      and         efficienc     performan
                                  detection          y in        ce due to
                                    of CK         selection       testing
                                   disease            of        over small
                                                  features       datasets
Liao et al.     MobileNet             To         Imbalanc        Requires        Accuracy-          Kidney
    [22]           V2 and        provide an         e and          huge              90.1          ultrasou
                ACWGAN-           auxiliary      insufficie     amount of         Precision-           nd
                     GP           prediction       nt data       training            81.9           image
                                    model        problems          data                             (KUI)
                                   with DL           are
                                                   solved
Sabanayag       CNN-based             To            High       Specified         AUC- 93.8          SEED,
 am et al.           DL           recognize      potentiali    variations          (SEED)          SP2 and
    [23]                              CK         ty can be     of    vessel      AUC- 81.0           BES
                                   disease        attained     and      un-         (SP2)
                                     from          in CK       usual             AUC- 85.8
                                    retinal        disease     lesions              (BES)
                                   images        identifica
                                                     tion
Navaneeth       Hybrid DL        To analyse         High       Evaluation        Accuracy-           Real
et al. [24]     (SVM+CN            the urea       robustne      processing          97.67            time
                     N)          concentrat      ss can be        time is        Specificity-       dataset
                                  ion in the      attained         high             97.83
                                    saliva                                       Sensitivity-
                                   sample                                            97.5
Kriplani et      Deep NN              To          Overfitti      Requires        Accuracy-           UCI-
  al. [25]                         classify      ng issues         large            97.76            CKD
                                   the CK          can be       amount of
                                  and non-         solved       informatio
                                      CK                           n for
                                   disease                        output
                                  efficiently                   prediction
When  undergoing  survey  over  the  existing  approaches  with  respect  to  CK  disease
classification, there emerges certain drawbacks which widely affects the performance of the

overall system. The limitations like lower performance model due to the utilization of smaller
datasets,  requires  larger  amount  of  training  data  for  promoting  effective  classification
process,Quantified variations of vessel and un-usual lesions degrades the output performance.
Also, high processing time is needed for the performance estimation. Due to the existence of
these complexities, accurate classification CK disease stages cannot be obtained. In order to
conquer these limitations and promote the classification accuracy, an effective deep learning
technique is proposed in this research work.
3.  Proposed methodology
CK disease is found to be highly threatening as it adversely affects the working conditions of
kidney. When it is not detected in the early stages, the affected people may enter severe
conditions. Most  of the patients are left in to  critical  stages due  to improper  or  wrong
prediction of diseases. Even though so many CK disease classification techniques are in
practice, precise results cannot be attained. Hence in the proposed research work, GNN-DQL
model is adopted for the precise classification stages of CK disease. The overall architecture
for accurate CK disease classification with different stages is illustrated in Figure 1.
                          Collection of
    Biomedical sensors     IoMT data                                                 Pre-processing of data
                                               Handling          Categorical      Transformation         Outlier
                                               of missing         encoding            of data           detection
                                                 values            of data
                      Normal
                       Mild                  Optimization of             Hybrid approach                 Evaluation
                                               parameters                                                 of GFR
    CK disease       Moderate                                          DQL                                  Age
   classification                                 AMO
                      Severe                                                         GNN                    SC
                        End
                      Figure 1: Overall architecture of CK disease classification
Initially,  the  BMs  are  used  to  evaluate  different  factors  of  CK  disease  such  as  serum
creatinine, sugar, red blood cells (RBC), white blood cells (WBC), potassium and so on. The
gathered data from the BM sensor are collected by the data centre through IoMT. To improve
the quality of data, pre-processing is the first step undertaken to eradicate the unwanted
distortions present in the data. The glomerular filtration rate (GFR) is evaluated, GNN- DQL
model is utilized to classify and predict several stages of CK disease including normal, mild,
moderate, severe and end stage. The parameters of the neural network are then optimized
through AMO approach. The steps involved in the proposed research work are described as
follows.

3.1 Data pre-processing
The evaluation of missing values and the eradication of noises including outliers as well as
the  validation  and  normalization  of  instable  data  are  the  pre-processing  stages  that  are
undertaken.  During  the  patient  assessment,  some  of  the  estimations  are  found  to  be
incomplete or missing. To compensate that, several pre-processing steps are carried out in the
proposed research work which are labelled as follows.
3.1.1   Handling of missing values
The modest method to work with the missing values are neglecting the records but however it
is not possible among the smaller datasets. During the process of data generation, the dataset
is examined to confirm whether any attribute values are missing. Through the adoption of
statistical  method  of  mean  imputation,  the  missing  values  for  numerical  structures  are
evaluated. The mode method is used for missing value replacement of insignificant features.
3.1.2 Categorical encoding of data
As  most  of  the  deep  learning  procedures  only  consider  numerical  values  as  input,  the
category values must be encoded in to arithmetical values. The characteristics of categories
including yes and no are represented by the binary values 0 and 1.
3.1.3 Transformation of data
    The process of converting numbers over the small scale so that the domination of one
variable over the others does not happen is called to be data transformation. Or else, the
learning  approaches  observe  bigger  values  as  advanced  and  smaller  values  as  lesser
regardless of the weight unit. The data alterations modify the values in a dataset and so they
can be treated further. To enhance the accuracy of deep learning approaches, this research
undergoes a data normalization method. The data is converted between the ranges from -1
and +1 where as the transformed data possess the standard deviation as 1 and mean as 0.
            The standardization can be stated as,
                                                s           vv      (1)
                                                      
From the above equation,        s denotes the standardized score, observed value is represented as                v
, mean is denoted as     v and    signifies the standard deviation.
3.1.4 Outlier detection
    Outliers are considered to be the observation facts which are inaccessible from the rest of
the data. An outlier can be created in the experiment by variability estimation or the signal
error. The learning process of deep learning algorithm can be distorted and mislead by the
outliers. The presence of outliers directs the process to longer training time, poorer result
generation and less model accuracy. Before the data transferring to the learning algorithm,
this research utilizes Interquartile range (IQR) foundedmethod to eliminate the outliers.

            The IQR is the evaluation of variability on the basis of dividing dataset in to quartiles.
The values that divide every part are termed to be first, second and third quartiles that are
denoted as        1V,   2Vand     3V . The formula to calculate IQR is given as follows.
                              IQR      V3   V1                                                             (2)
where,       1V denotes the middle value in the first half of the ordered data set whereas                                              3V
denotes the second half.              2V denotes the median value in the dataset.
3.2 Estimation of Glomerular filtration rate
The GFR is highly crucial to most of the characteristics including public health, medical care
and research. The  clinical laboratories render a  significant role in GFR assessment and
chronic kidney  disease  diagnosis. In the GFR  evaluation,  serum  creatinine  measurement
along with the estimated GFR is recommended as the initial step. From the gathered data, the
GFR is estimated to classify the five different stages of CK disease and here the filtration rate
is estimated on the basis of age and serum creatinine (SC) level. The equation for estimating
the GFR for age greater than or equal to 18 years can be mathematically expressed as,
GFR                                                                                                                            Age142                                                                                    200min                                                       012SCk1,/b                   kmaxSC1,/.1                     .09938.1                      (3)
From the above equation,              k  = 0.7 for females and 0.9 for males,                  b  = -0.241 in case of females
and -0.302 for males.             SC   denotes the SC level in mg/dL and Ageis represented in years.
Table 2 shows the equation to estimate                   GFRfrom SC level.
                                             Table 2: GFR estimation values
       Age                     Gender                  SC mg/dL                                          GFR
  18                    Female                       0.70 or <0.71                                                                                age142                               241                                     012SC7.0/.0                    .09938.1
                                                    > 0.70                                                                                          age142                                200                                   012SC7.0/.1                    .09938.1
  18                    Male                         0.90 or <0.91                                                 142                                  302                                          ageSC                         99389.0/.0    .0
                                                    > 0.90                                                           142                                  200                                          ageSC                         99389.0/.1    .0
3.3 Graph neural network with deep Q learning technique
     GNNs are the framework to gather the node dependency in graphs through passing of
messages between the nodes. The GNN performs on the graph to describe the data from its
neighbourhood with random stages. This creates GNN as an appropriate tool to utilize for the
wireless networks that holds compound features which cannot be taken in a closed form. In
the proposed research work, the GNN based approach in accordance with the relationship of
cell and entities between the nodes.

            Two adjacent matrices are defined for the given network comprising a set of                                                             P  cells
andQ      entities. The graph between cells is represented as                                              1,0      PPand the graph between
                                                                                                    clR                       
entities and cells are denoted as                             1,0      QP. The mathematical expression can be given as,
                                                        eR                      
                                                                 1ife,e    cl
                           R          u,                              v          Otherwisefclufclv                                          (4)
                              cl
                                               0
                                                              1ife,e     e
                          R         u,                              v          Otherwisefclufclv                                              (5)
                             e
                                             0
A L-layer GNN is considered which  calculates on the graph and the fundamental nodal
characteristics   of   the   cells   and   the   entities   are   defined   as                                               )0(,          )0(and            )0(
                                                                                                                             1,clY        2,clY           eY
correspondingly.
            The  initial  nodal  characteristics  are  the  functions  of  cell  data  rates  and  the
reportednetwork capacities and entities. The channel capacity matrix                                                  C            PZQis defined with
                                                                                                                           fc      ,clfe
the elements                  efc      ,cland user rate matrix           Z            PZQwith elements               u      v   for an assumed
                            u   fv                                                                                    C  f cl
                                                                                                                            u
cell-entity connectivity graph. The input features can be calculated as,
                                               Y     )0(                                                 Q                         2RZ1                             Z               (6)
                                                  cl                              Z1,                    1clPP
                                                  Y     )0(                                                   Q                          2RZT1                             C        (7)
                                                    cl                               Z2,                      1eQP
                                                   Y     )0(                    1                                              P                      2CT1                              ZT        (8)
                                                     e                            ZQQ
From the above equation, the vector concatenation operator is denoted as                                                      ... All-ones vector
of size      P1 and     Q1 is denoted as          P  ,Q    . Each of the latent features gather either the sum rate of
nearby cells or node or channel capacity in case of entities. These are chosen as the features
as they gather relevant data regarding to make better decisions.
            At each layer, the GNN evaluates a                           d  dimensional latent feature vector for every node
 f            ecl,fVin  the  graphG            .  At   L  layer,  the  later  feature  estimation  can  be  expressed  as
  u       v
follows.
                    H                                                                                                              L                                                                2                          YLwL                       dYLwLP                                               (9)
                       cl                                                  Zcl1,           1cl2,
                       H                                                                   L                       3                     dYLwLQ                                                            (10)
                           e                          Ze
                           Y                                                 (L                 )1                     )RH(LPd                                                        (11)
                             cl                         Z1,clcl
                                Y                                             (L                )1                    )RTH(LQd                                               (12)
                                  e                        Zecl

                         Y                                                (L                 )1                    )RH(LPd                                                   (13)
                           cl                        Z2,ee
From  the  above  equations,  the  neural  network  weights  are  represented  as                               w                              d0    2and
                                                                                                                   k     Z
w                               dLdfor L                   ,3,2,1,0   ketc, the layer index of GNN is represented as Land.denotes
  k     Z
the non-linear activation function. The sum of hidden features of cell to cell and cell to entity
graph connectivity is represented by the auxiliary matrices                           (L) and      (L). The   L  layer in GNN
                                                                                      clH          eH
effectively replicates the above estimation for                     L            l1,0        ,....1. By this, the nodal features are
directed to other nodes and will get combined at distant nodes. Every feature comprises of
data regarding      l hop neighbours whereas the embedding is undertaken                          L  times.
            In the last layer of GNN, the feature vectors are integrated to attain a scalar valued
score for    G  . The output layer of GNN is combined over cells, the score estimation                                         ( l)1
                                                                                                                               clH
invariant to nodes before transforming to the single fully connected neural network layer. The
network score of the graph            G  is expressed as follows.
                   S                                    l(                              wG)             51TH()1w                                                              (14)
                                  P    cl      4
All-ones vector of size           P  is denoted as        T , the weight matrix of the fully connected neural
                                                          P1
network is represented as           w             dZdand the vector to combine the output of neural network is
                                       4
represented as       w      dZ1.  Once the evaluations of GNN are over, the scores of                           G   ,S(G)    will
                       5
be adopted to choose the best connection graph. The optimal weights of GNN are learned by
the deep Q learning algorithm.
The Q-function is learned  from the cell and  entity  placement instances through deep Q
learning approach. The major merit of Q- function is to establish GNN scalable over various
sizes that can gather limited network features with different number of cells and entities. To
generate the optimal selection, the right Q function has to be learned. When the Q function is
gathered through GNN, this renders to learn the GNN parameters which is done by sequential
accumulation of new cell entity connections over partly connected graph. The state, action
and the reward in deep Q learning approach are provided as follows.
The  state        TSis  defined  as  the  present  graph               TGholding  the  cells  and  linked  entities  at
iteration and also the input features of corresponding nodes                             )0(and      )0(. The beginning state
                                                                                      clY          eY
can  be  contemplated as the  partlylinked network  with  linked and unlinked  entities. The
ending  stateis  attained  when  the  entire  network  entities  are  associated.    The  action
A                                 ,T             eGTfclu    fevin     stepTis to link a separate  entity to  one  of the  cells. The  reward
R(             TST  A,)atTSstate after choosing the action        TAcan be expressed as,
                     R(                                                    ST                           G,AT)                                  TU           1GTU                                                   (15)
The reward is described as the variation in the network utility function after linking a new
entity. The deterministic greedy policy can be expressed as                            (          A                                                             T         ,T                                      AST)ArgmaxAQ      STT

with   greedy examination during the training process. Here                    Q     ,      ST  ATis denoted in equation
(14) with    G                                 ,T             eSTfclu    fev.
At first, the parameters are initialized in deep Q learning approach that are defined for every
deployment. At everyT           step, one entity      A                  ,T    efclu    fevis linked by pursuing the greedy policy
 (             TAT S)where the exploration rate is denoted as      . The number of stagesT           is provided by
the end state     TS. The graph       TGis updated and so the following step                1TSis attained. Every time
when the graph is being efficient, the new input features called                            )0(and     )0(are estimated.
                                                                                         clY         eY
For every chosen action, the reward               R(             TST  A,)is evaluated and thellayer GNN evaluation
renders the score for every action and state pair. To enhance the classification accuracy, the
GNN with DQL parameters are optimized through the adoption of AMO approach.
            The may flies that are separated to male and female would update the velocities
randomly. The individual velocities are updates from the weighted present velocities with
some other weighted distance among them and the global finest individuals. The weighted
distance of either parts can be found through the following expression.
                     J                                       n                           2Ker         U                                                       (16)
                       o                          Umnm
When      nU is far away from         mU, the velocities are updated with a lower amplitude. When they
are near, the velocities are updated with a higher amplitude. But these situations cannot be
acceptable probably because when the individuals are distant away, the velocities should be
reorganized with larger rates and should attain lower rates when they are nearby. Hence the
equation (16) can be updated to optimize the parameters of GNN-DQL as,
                                 r
                    J                                    n      o                        UKme         Unm                                                            (17)
Where      oJdenotes the composited velocity,               mKand    are constants,        nUdenotes the male fly,
  mUrepresents   the   female   fly   and           nrdescribes   the   Cartesian   distance.   Through   the
implementation of the proposed research, the classification accuracy can be greatly improved.
The evaluation time for conducting this research is low and the overall system performance is
enhanced.
4.  Results and discussion
The  performance  outcomes  of  the  proposed  method  are  conferred  in  this  section.  The
experiments  are  analysed  and  implemented  using  the  PYTHON  simulation  tool.  The
performance outcomes  of  the  proposed technique  are  compared  with  the recent existing
techniques. The performance metrics including accuracy, F1 score, recall and precision of the
proposed method are compared with the existing approaches like Linear Regression (LR),
Nearest Neighbour (KNN), Support Vector Machine (SVM), Decision Tree (DT) and Naive
Bayes (NB). The metrics like specificity, Mathew’s correlation coefficient (MCC), Kappa,
Balanced score (BS) and AUC are compared with the existing techniques like DT, SVM,

KNN,  LR,  Adaboost (ADB), Stochastic Gradient Descent (SGD),  Multilayer Perceptron
(MLP) and Gaussian Naive Bayes (GNB). In accordance to this, some of the methods such as
SVM, Multi-Kernel Support  Vector Machine (MKSVM), Hybrid  Kernel Support  Vector
Machine  (HKSVM),  and  Fuzzy  Min-Max  GSO  Neural  Network  (FMMGNN)  are  also
adopted for comparing Positive predictive value (PPV), Negative predictive value (NPV),
False positive rate (FPR) and False negative rate (FNR). The mean absolute error (MAE)
performance is compared with the techniques like Random Forest (RF), NB, SVM, neural
network (NN), DL, KNN, DT and Auto-MLP. The error rate (ER) performance is compared
with NB, SVM, Artificial neural network (ANN), NB-Hybrid Filter Wrapper Embedded-
feature selection (NB-HFWE-FS), ANN-HFWE-FS and SVM-HFWE-FS.
4.1 Dataset description
The proposed CK disease classification approach is performed by utilizing the gathered CK
disease data. This dataset comprises of 400 instances, 76 parameters and 25 attributes. Butthe
data  may  subject  to  noisy  data  and  numerical  missing  values  that  has  been  retrieved
systematically through pre-processing. For analysing the results, the dataset has been splitted
into training and testing sets as 80% and 20%. The download link of the gathered dataset is
https://www.kaggle.com/mansoordaku/ckdisease/activity.Moreover,     different     kinds     of
features include age, anemia, bacteria, albumin, appetite, blood urea, blood pressure, blood
glucose random, diabetes mellitus, coronary artery disease, hypertension, haemoglobin, pus
cell clumbs, pus cell, packed cell volume, potassium, RBC, pedal edema, serum creatinine,
WBC count, specific gravity, sodium, sugar and RBC count.
4.2 Performance metrics
The description of each performance metric considered for the performance evaluation of the
proposed method and its mathematical expression are explained as follows.
(a) Accuracy
The overall count of precise predictions over the whole amount of predictions is termed as
accuracy. The accuracy can be mathematical expressed as,
                       Accu                  DE                                               (18)
                                D   E   F    G
Where    D  signifies true positive,      E denotes true negative,       F  defines false positive and        G
signifies false negative.
(b) F1 score
The harmonic means of PPV and Recall or TPR (True positive rate) is termed as F1 score. It
can be mathematically represented as,
                    F                  1S  2PPVTPR                                                        (19)
                               PPV      TPR
(c) Recall

The measure of positive outcomes over the entire count of samples that are actually positive
and is also called as Recall. Recall can be mathematically expressed as,
                              R      D                                                            (20)
                                   D     G
(d) Precision
Precision is represented as the availability of predicted positive that are actually positive. The
mathematical expression of precision can be denoted as,
                               P      D                                                          (21)
                                    D    E
(e) Specificity
The number of negative outcomes over the entire number of samples that are truly negative.
The specificity rate can be mathematically represented as,
                              S        E                                                          (22)
                                     E    F
(f) MCC
MCC is described as the combination between true and projected decisions by undertaking
the correlation coefficient evaluation formula that can be expressed as,
                MCC                                             (D*E)(F*G)                                        (23)
                             (E    G)(E     F )(D    G)(D      F)
(g) Kappa
The  steadiness  of  prediction  and  employment  of  probabilistic  evaluations  amongst  the
predictable scores in case of disagreement and agreement is determined in Cohen’s Kappa
Score (CKS). It can be expressed as,
                             K            10f                                                      (24)
                                          f
From the above equation,               0 represents the score agreement between predicted and actual
values and        f describes the score disagreement between actual and predicted ones.
(h) Balanced score
The arithmetic mean of sensitivity and true negative rate is termed as the balanced score that
can be mathematically expressed as,
                        BS    2                           F1DE                                                (25)
                                 D    E     E      

(i)  AUC
The capability of the technique to differentiate between the aimed classes is signified by
AUC. It is also termed to be area underneath the receiver operating curve. The performance
of AUC is evaluated by mapping the graph for true positive rate (TPR) over FPR.
(j) Positive predictive value
The probability that intends with a positive screening test showing the presence of actual
disease is analysed in PPV that can be expressed as,
                            PPV       D                                                         (26)
                                      D    E
(k) Negative predictive value
The amount of the cases providing negative test outcomes that are really positive is analysed
in NPV that can be expressed as,
                            NPV        E                                                        (27)
                                      E    G
(l)  False positive rate
The ratio between the quantity of negative results that are wrongly categorized as positive is
analysed in FPR which can be represented as,
                           FPR       F                                                            (28)
                                     F    E
(m) False negative rate
FNR refers to the proportion of important tests which failed to eradicate the null hypothesis
when it is indeed false. It can be mathematically expressed as,
                           FNR        G                                                           (29)
                                     D    G
(n) Mean Absolute error
The prediction error between the predicted and actual values is termed as MAE. High values
of error tend to minimize the CK disease classification accuracy. The formulation of MAE
can be expressed as,
                                     m
                                    v          y      xvv
                        MAE               1m                                                         (30)
From the above equation,           x  indicates the predicted value,           y  represents the actual value and
m    denotes the total amount of data samples.

(o) Error rate
The proportion of the number of erroneous data units over the entire amount of data units
transmitted in a process is termed as error rate. It can be expressed as,
                        Err        VaVe                                                         (31)
                                   Ve
where     eVdenotes the expected value,         aVrepresents the attained actual value and         Err   denotes
the error percent.
4.3 Performance analysis and comparison
The significant performance metrics adopted for estimating the comparison of proposed and
existing approaches including accuracy, F1 score, recall, precision, MCC, Kappa, BS, AUC,
PPV, NPV, FPR and FNR are analysed with its description and graphical representation that
are  explained  as  follows.Table  3  describes  the  proposed  results  in  terms  of  various
performance metrics.
                           Table 3: Performance analysis of proposed work
               Technique             Performance metrics               Performance outcomes
                                             Accuracy                              99.93
                                             Precision                            99.861
                                            Sensitivity                            99.86
                                             F-measure                            99.869
                                                MCC                               99.901
                                            Specificity                           99.911
                Proposed                         BS                                99.88
              (GNN-DQL)                         AUC                                99.89
                                               Kappa                               99.72
                                                FPR                                0.011
                                                FNR                                0.013
                                                NPV                                99.3
                                                PPV                                99.20
                                             Error rate                            0.115
                                                MAE                                0.86
Table 4 represents the performance outcomes of proposed and existing methods [26] in terms
of accuracy, F1 score, recall and precision.
               Table 4: Performance outcomes of proposed and existing techniques
                    Techniques                  Performance outcomes (%)
                                      Accuracy       F1 score       Recall      Precision

                   LR               99            99           100         98
                   KNN              92            92           88          98
                   NB               95            95           92          100
                   SVM              92            92           87          96
                   DT               97            97           95          100
                   Proposed         99.93         99.86        99.86       99.86
Table 5 signifies the performance results of proposed and existing techniques [27] in terms of
specificity, MCC, Kappa, BS and AUC.
                    Table 5: Result analysis of proposed and existing methods
                Techniques                   Performance outcomes (%)
                                 Specificity      MCC        kappa       BS        AUC
                DT               93               88         87          94        94
                SVM              95               91         91          97        96
                ADB              95               93         93          97        97
                KNN              85               76         76          89        89
                GNB              91               88         87          95        95
                SGD              84               81         79          92        92
                MLP              98               95         94          97        97
                LR               99               96         96          98        98
                Proposed         99.91            99.90      99.72       99.88     99.89
Table 6 demonstrates the performance comparison of proposed and existing techniques [28]
in terms of PPV, NPV, FPR and FNR.
                 Table 6: Performance comparison of PPV, NPV, FPR and FNR
                   Techniques                   Performance outcomes

                                      PPV          NPV          FPR          FNR
                   HKSVM           98.49         95.99        0.029       0.030
                   SVM             96.70         82.37        0.050       0.160
                   FMMGNN          89.94         85.49        0.062       0.040
                   MKSVM           99.00         96.30        -           0.032
                   Proposed        0.992         99.3         0.011       0.013
The performance comparison of MAE and error rate is described in Table 7.
                    Table 7: Performance comparison of MAE and error rate
                                                  Techniques used
Performance        RF     NB     SVM       NN      DL     KNN        DT     Auto-MLP       Proposed
    MAE           0.91    3.8     6.31     3.48   1.96    28.44     4.58        3.78          0.86
                                                  Techniques used
Performance        ANN-          NB-         SVM-          NB         ANN        SVM       Proposed
                  HFWE-        HFWE-        HFWE-
                     FS           FS           FS
 Error rate        13.33        14.77         6.67        33.33      30.00       26.67        0.115
Confusion matrix
The significance of the proposed CK disease model in classifying the five different stages
including normal, mild, moderate, severe and end. Figure 2 describes the confusion matrix
using training data for the proposed model of CK disease classification.

                                       Figure 2: Confusion matrix
Here the data collected from different people for classifying the five stages of CK disease are
considered. From the figure, it can be clearly analysed that the proposed model accurately
classifies the stages of CK disease with improved accuracy which are represented in the
diagonal format. The remaining values represent the number of wrong predictions made with
respect to each stage. For example, 1 normal person is wrongly predicted as mild stage. The
mild, moderate, severe and end stages are precisely classified with no error and hence the
accuracy of the proposed model is widely enhanced.

                                    Figure 3: Accuracy comparison
Figure 3 provides the  graphical representation  of the  performance  measures in terms of
accuracy. From the figure, it is clear that the accuracy attained by the proposed model is
found to be highly accurate when compared to the existing models like LR, KNN, NB, SVM
and DT. The overall accuracy of the proposed model is attained to be 99.93%. the existing
models obtained lower accuracy due to larger accumulation of datasets, degraded system
performance and increased complexities. Higher rate of accuracy insists that the technique
accomplishes better classification performance.
                           Figure 4: Performance comparison of F1 score
Figure 4 illustrates the graphical representation of F1 score in terms of proposed and existing
techniques. It is made clear that the proposed method attains more capability to classify the
CK disease depending on the input parameters when compared to the existing techniques.
The value of F1 measure is attained to be 99.86% in the proposed method whereas the
existing approaches like LR obtained 99% of F1 score, KNN as 92%, NB as 95%, SVM as
92% and DT as 97%in classification performance. In the proposed method, F1  measure
shows better results in classifying the different stages.

                            Figure 5: Performance comparison of Recall
The graphical representation of recall on the basis of proposed and existing approaches is
presented in Figure 5. 99.86% of recall is obtained while assessing the performance of the
proposed technique in contrast to the existing approaches. The existing learning algorithms
has accomplished100%, 88%, 92%, 87%, 95% with respect to LR, KNN, NB, SVM and DT.
Due to high complexities of time and storage, the existing approaches tends to offer lower
performance other than LR approach when compared to the proposed method.

                                    Figure 6: Precision performance
The graphical representation of precision in terms of proposed and existing approaches are
shown in Figure 6. Precision is one of the prominent metrics to be measured for gathering the
effectiveness of outcomes. The proposed GNN-DQL method has attained 98% of precision
and showed a better result in reducing the false detection rate. While the existing learning
methodologies other than NB and DT has achieved lower results when compared to the
proposed approaches. Finally, from the figure, it is estimated that the proposed technique
outperforms well than the existing methods due to higher ability in data handling process.
                                   Figure 7: Specificity performance
Figure  7  presents  the  performance  values  obtained  over  the  experiments  in  terms  of
specificity for proposed and existing approaches. The results illustrates that the proposed
model performed better in classifying the various stages of CK disease effectively than the
other compared models. The overall specificity performance of the proposedmethod is found
to  be  99.91%.  The  proposed  method  has  improved  the  classification  performance  high
specificity. The selection of optimal features helped in optimally predicting the outcomes
based on the input attributes. Whereas the performance attained in the existing methods like
DT, SVM, ADB, KNN, GNB, SGD, MLP and LR are 93%, 95%, 95%, 85%, 91%, 84%,
98% and 99% respectively.

                                     Figure 8: MCC performance
The performance of MCC for the proposed technique is analysed with existing methods and
the attained outcomes are portrayed in Figure 8. In the figure, it is clearly shown that the
proposed technique has gained improved correlation between true and predicted decisions
than the existing methods. This obviously exposed that the proposed method has established
minimum false rate and superior to CK disease classification stages. MCC attained by the
proposed method is 99.90% whereas, the existing methods like DT, SVM, ADB, KNN, GNB,
SGD, MLP and LR are88%, 91%, 93%, 76%, 88%, 81%, 95%and 96% respectively.
                                     Figure 9: Kappa performance

Figure  9  represents  the  graphical  illustration  of  kappa  performance  in  CK  disease
classification. High kappa values are obtained in the proposed approach as 99.72% which
shows better outcomes when compared to the existing approaches. When compared to the
existing  methods  of  DT,  SVM,  ADB,  KNN,  GNB,  SGD,  MLP  and  LR,  the  kappa
performance  of the  proposed  method tends to  be  highly  superior  in  classifying the CK
disease. Efficient performance can be attained in testing the data reliability gathered for CK
disease classification. The performance of kappa among the existing method are found to be
low because of larger accumulation of information from the datasets.
                                     Figure 10:Performance of BS
Figure 10 signifies the BS performance of proposed and existing methods. The proposed
model has attained 99.88% because of limited redundant features and false rates. The result
of BS is analysed with other existing methods like DT, SVM, ADB, KNN, GNB, SGD, MLP
and LR which have obtained 94%, 97%, 97%, 89%, 95%, 92%, 97% and 98% respectively.
On this performance evaluation, it is found that the proposed method is highly efficient in CK
disease classification.

                                         Figure 11:AUC analysis
Figure 11represents the AUC comparisonin terms of proposed and existing approaches. The
proposed method possesses better ability to differentiate between the target classes. The
graph has plotted between FPR and TPR to establish the AUC value. The proposed method
has attainedthe AUC value of 99.89%, that is superior than the other state-of-art techniques
since,  it  has  improved  the  capabilityin  CK  disease  classification  on  the  basis  of  input
parameters.

                                     Figure 12: PPV performance
Figure  12  establishes  the  comparison  of  PPV  with  different  methods  including  SVM,
MKSVM, HKSVM, and FMMGNN. From the graph, it can be analysed that the proposed
technique  has  accomplished  better  result  when  compared  to  the  other  approaches.  The
proposed method attains maximum PPV of 0.992 so the false detection rate is found to be
very low. When comparing with the other techniques, FMMGNN has attained very low PPV
of 0.89 whereas the MKSVMhas reached better value of 0.99, which is inferior than the
proposed  technique.  Overall,  it  is  shown  that  the  proposed  model  outperformed  when
compared to the existing methods considerably.
                                     Figure 13: NPV performance
The performance evaluation of NPV in terms of proposed and existing methods are illustrated
in Figure 13. The existing methods such as MKSVM, HKSVM, SVM and FMMGNN are
adopted to analyse the NPV performance in comparison with the proposed model to classify
CK  disease. From the  figure,  it  is  perceptibly analysed that  the proposed approach has
attained maximum NPV than recent existing approaches. The NPV value obtained by the
proposed model is99.3 whereas, the existing methods have secured95.99, 82.37, 85.49 and
96.30 for HKSVM, SVM, FMMGNN, and MKSVM considerably.

                                      Figure 14: FPR performance
FPR of the proposed method is compared with the existing approaches and the attained
outcomes are demonstrated in Figure 14. From the graph, it can be analysed that the proposed
technique  has accomplished less FPR  than  the  existing techniques. This  shows  that the
proposed method is superior in classifying the different stages of CK disease. The FPR value
achieved by the proposed method is 0.011 whereas, the FPR values of the existing methods
including HKSVM, SVM, FMMGNN are 0.029, 0.050 and 0.062 respectively.

                                     Figure 15: FNR performance
FNR of the proposed method is compared with the existing methodologies and the obtained
outcomes are presented in Figure 15. For an efficient system, the FNR performance value has
to be comparatively low. It is clearly observed from the figure that the proposed technique
has attained less FNR than the existing techniques. This indicates that the proposed method
has accomplished less false rate and appropriate for effective classification. The FNR value
obtained by the proposed method is 0.013 whereas, the FNR values of the compared methods
like   HKSVM,   SVM,   FMMGNN,   MKSVM   are   0.030,   0.160,   0.040   and   0.032
correspondingly.
                                     Figure 16: MAE performance
Figure 16 presents the MAE performance of the proposed modelanalysed in CK disease
classification. Minimized error rates  enhance the classification performance with reduced
wrong predictions. The MAE performance is comparatively low for the proposed model
compared to the existing approaches like RF, NB, SVM. NN, DL, KNN, DT and Auto-MLP
[29].Therefore,  the  existing  approaches  are  found  to  be  inappropriate  for  CK  disease
classification process. The MAE is obtained to be 0.86 which is adversely low than the
existing methods because of effective learning algorithm.

                                  Figure 17: Performance of error rate
Figure 17 presents the error rate outcomes of the proposed and existing models obtained
during CK disease classification. The figure shows that the error performance of the proposed
modelis found to be more optimal than the other existing methods. The error rate of hybrid
model is found to be 0.115 and the error rates of existing approaches like NB-HFWE-FS,
ANN-HFWE-FS, SVM-HFWE-FS, NB, ANN and SVM [30] are found to be 14.77, 13.33
6.67,  33.33,  30.00  and  26.67  respectively.  The  classification  of  the  proposed  model
isobtained to provide fruitful results than the other models.
Training and testing measures for model accuracy and loss
The loss and accuracy of the proposed approach is analysed with training and testing data. In
the proposed research, 80% of data is adopted to train the model and 20% of data is adopted
to test the model. The training and testing model accuracy and loss of the proposed approach
is depicted in Figure 18 (a) and (b).
                         (a)                                                     (b)

                            Figure 18: (a) Model Accuracy (b) Model loss
By varying the epoch size, all the accuracies and losses of the proposed model are evaluated.
The accuracy for the two cases is similar. While observing the training and testing accuracy,
only slight variations are captured with high accuracy. The increase in accuracy may occur
due to the increased epoch size. If the epoch size is 40, the proposed model procures a
training and testing accuracy in the range of 80 to 85% consecutively. If the epoch size is 60,
the model retains accuracy in the range of 85 to 90% and if the epoch size is increased to80,
the proposed model generates an accuracy in the range of 90 to 100%. And from the figure, it
is  obvious that the proposed  approach achieves  maximum accuracy and the accuracy is
almost similar for training and testing data samples.
            The training and testing loss is procured for the proposed model and the network has
been trained for 100 epoch size. The increase in epoch size decreases the loss. If the epoch
size is 40, the model attains a training and testing loss in the range between 0.15-0.20. If the
epoch size is 60, the model acquires a loss in the range of 0.10 to 0.15. For epoch size 80, the
value ranges between 0.10 to 0.5. The model achieved minimal error value due to optimal
training through the adoption of MAO approach for optimizing.
5.  Conclusion
CK disease is one of most threatening diseases in the recent years and exact diagnosis is most
challenging. In the proposed research work, precise classification of five stages of CK disease
like normal, mild, moderate, severe and end is attained through GNN-DQL approach. The
unwanted distortions are eradication through data pre-processing which includes Handling
Missing Values, Categorical Data Encoding, Data Transformation and Outlier Detection.
The GFR rate is evaluated with respect to age and SC level for obtaining the enhanced result.
Then, GNN-DQL technique is carried out for improving the classification accuracy. The
parameters are optimized through  AMO  method and  the  five  stages  of CK  disease are
precisely classified. The classification performance is analysed  with respect to accuracy,
precision, recall, specificity, F1 score, confusion matrix and so on. The proposed method is
implementedand  the  performances  are  analysed  using  PYTHON  simulation  tool.  The
classification  accuracy  of  99.93%  is  observed  in  the  CK  disease  classification  of  five
different stages. The MAE and the error rate are attained to be 0.86 and 0.115 which are
comparatively less than the other existing approaches. The proposed model is tested with
smaller datasets and to enhance the system performance, significant volumes of data will be
gathered  in  the  future  for  better  results.  In  addition  to  this,  valuable  features  will  be
implemented to attain a wider perception over the enlightening parameters regarding CK
disease.
Abbreviations
CK:Chronic  Kidney  ,GNN-DQL  :graph  neural  network  based  deep  Q  learning,  IoMT:
Internet  of  medical  things,  GFR:  Glomerular  Filtration  rate  ,AMO:  Adaptive  Mayfly
Optimization,   HB:High   Blood,   SEED:   Singapore-epidemiology-eye-diseases,   KNN:K-
Nearest  Neighbour,LR:  Linear  Regression,  DT:Decision  Tree,  NB:Naive  Bayes,  MCC:

Mathew’s correlation coefficient, BS: Balanced score, MLP: Multilayer Perceptron, GNB:
Gaussian Naive Bayes, FMMGNN: Fuzzy Min-Max GSO Neural Network,;
Acknowledgements
Not applicable.
Authors’ contributions
TBPR  has found the proposed algorithms and obtained the datasets for the research and
explored different methods discussed. and contributed to the modification of study objectives
and framework. The rich experience was instrumental in improving our work. VD has done
the literature survey of the paper and contributed writing the paper. All authors contributed to
the editing and proofreading. All authors read and approved the final manuscript.
Funding
Authors did not receive any funding for this study.
Availability of data and materials
This dataset comprises of 400 instances, 76 parameters and 25 attributes. But the data may
subject to noisy data and numerical missing values that has been retrieved systematically
through pre-processing. For analysing the results, the dataset has been splitted into training
and  testing  sets  as  80%  and  20%.  The  download  link  of  the  gathered  dataset  is
https://www.kaggle.com/mansoordaku/ckdisease/activity
Declarations
Ethics approval and consent to participate
Not applicable
Consent for publication
Not applicable.
Competing interests
The authors declare that they have no Competing interests.
Author details
 1 Research Scholar, School of Electronics Engineering, Vellore Institute of Technology,
Chennai, India
 2 Associate Professor, School of Electronics  Engineering, Vellore Institute of Technology
Chennai, India

References
[1] Romagnani, Paola, Giuseppe Remuzzi, Richard Glassock, Adeera Levin, Kitty J. Jager,
Marcello  Tonelli,  Ziad  Massy,  Christoph  Wanner,  and  Hans-Joachim  Anders.  "Chronic
kidney disease." Nature reviews Disease primers 3, no. 1 (2017): 1-24.
[2] Sharma, Kanishka, Christian Rupprecht, Anna Caroli, Maria Carolina Aparicio, Andrea
Remuzzi, Maximilian Baust, and Nassir Navab. "Automatic segmentation of kidneys using
deep  learning  for  total  kidney  volume  quantification  in  autosomal  dominant  polycystic
kidney disease." Scientific reports 7, no. 1 (2017): 1-10.
[3] Norouzi, Jamshid, Ali Yadollahpour, Seyed Ahmad Mirbagheri, Mitra Mahdavi Mazdeh,
and Seyed Ahmad Hosseini. "Predicting renal failure progression in chronic kidney disease
using integrated intelligent fuzzy expert system." Computational and mathematical methods
in medicine 2016 (2016).
[4] Kaur, Guneet, and  Ajay Sharma.  "Predict  chronic kidney  disease using data mining
algorithms  in  hadoop."  In 2017  International  Conference  on  Inventive  Computing  and
Informatics (ICICI), pp. 973-979. IEEE, 2017.
[5] Dulhare, Uma N., and Mohammad Ayesha. "Extraction of action rules for chronic kidney
disease   using   Naïve   bayes   classifier."   In 2016   IEEE   International   Conference   on
Computational Intelligence and Computing Research (ICCIC), pp. 1-5. IEEE, 2016.
[6] Tricoli, Antonio, and Giovanni Neri. "Miniaturized bio-and chemical-sensors for point-of-
care monitoring of chronic kidney diseases." Sensors 18, no. 4 (2018): 942.
[7] Lan, Kun, Dan-tong Wang, Simon Fong, Lian-sheng Liu, Kelvin KL Wong, and Nilanjan
Dey. "A survey of data mining and deep learning in bioinformatics." Journal of medical
systems 42, no. 8 (2018): 1-20.
[8]  Saidi,  Tarik,  Omar  Zaim,  Mohammed  Moufid,  Nezha  El  Bari,  Radu  Ionescu,  and
Benachir    Bouchikhi.    "Exhaled    breath    analysis    using    electronic    nose    and    gas
chromatography– mass spectrometry for non-invasive diagnosis of chronic kidney disease,
diabetes mellitus and healthy subjects." Sensors and actuators B: chemical 257 (2018): 178-
188.
[9] Wu, Jiandong, Dumitru Tomsa, Michael Zhang, Paul Komenda, Navdeep Tangri, Claudio
Rigatto, and Francis Lin. "A passive mixing microfluidic urinary albumin chip for chronic
kidney disease assessment." ACS sensors 3, no. 10 (2018): 2191-2197.
[10] Wibawa, Made Satria, I. Made DendiMaysanjaya, and I. Made AgusWirahadi Putra.
"Boosted classifier and features selection for enhancing chronic kidney disease diagnose."
In 2017 5th international conference on cyber and IT service management (CITSM), pp. 1-6.
IEEE, 2017.
[11]  Bressendorff,  Iain,  Ditte  Hansen,  Morten  Schou,  Charlotte  Kragelund,  and  Lisbet
Brandi.  "The  effect  of  magnesium  supplementation  on  vascular  calcification  in  chronic

kidney disease—A randomised clinical trial (MAGiCAL-CKD): Essential study design and
rationale." BMJ open 7, no. 6 (2017): e016795.
[12] Chan, T.C., Zhang, Z., Lin, B.C., Lin, C., Deng, H.B., Chuang, Y.C., Chan, J.W., Jiang,
W.K., Tam, T., Chang,  L.Y. and Hoek, G., 2018. Long-term  exposure to ambient  fine
particulate  matter  and  chronic  kidney  disease:  a  cohort  study. Environmental  health
perspectives, 126(10), p.107002.
[13] Ledbetter, David, Long Ho, and Kevin V. Lemley. "Prediction of kidney function from
biopsy        images        using        convolutional        neural        networks." arXiv        preprint
arXiv:1702.01816 (2017).
[14] Sharma, Kanishka, Christian Rupprecht, Anna Caroli, Maria Carolina Aparicio, Andrea
Remuzzi, Maximilian Baust, and Nassir Navab. "Automatic segmentation of kidneys using
deep  learning  for  total  kidney  volume  quantification  in  autosomal  dominant  polycystic
kidney disease." Scientific reports 7, no. 1 (2017): 1-10.
[15] Keshwani, Deepak, Yoshiro Kitamura, and Yuanzhong Li. "Computation of total kidney
volume from CT images in autosomal dominant polycystic kidney disease using multi-task
3D  convolutional  neural  networks."  In International  Workshop  on  Machine  Learning  in
Medical Imaging, pp. 380-388. Springer, Cham, 2018.
[16] Shankar, K., P. Manickam, G. Devika, and M. Ilayaraja. "Optimal feature selection for
chronic   kidney   disease   classification   using   deep   learning   classifier."   In 2018   IEEE
international conference on computational intelligence and computing research (ICCIC), pp.
1-5. IEEE, 2018.
[17]  Bevilacqua,   Vitoantonio,   Antonio  Brunetti,  Giacomo  Donato  Cascarano,   Flavio
Palmieri,  Andrea  Guerriero,  and  Marco  Moschetta.  "A  deep  learning  approach  for  the
automatic detection and segmentation in autosomal dominant polycystic kidney disease based
on magnetic resonance images." In International Conference on Intelligent Computing, pp.
643-649. Springer, Cham, 2018.
[18] Zhang, Jinghe, Jiaqi Gong, and Laura Barnes. "HCNN: Heterogeneous convolutional
neural  networks  for  comorbid  risk  prediction  with  electronic  health  records."  In 2017
IEEE/ACM International
[19] Zheng, Q., Tastan, G., & Fan, Y. (2018, April). Transfer learning for diagnosis of
congenital abnormalities of the kidney and urinary tract in children based on ultrasound
imaging data. In 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI
2018) (pp. 1487-1490). IEEE.
[20] Wang, Bohan, Hsing-Wen Wang, Hengchang Guo, Erik Anderson, Qinggong Tang,
Tong Tong Wu, Reuben Falola, Tikina Smith, Peter M. Andrews, and Yu Chen. "Optical
coherence tomography and computer-aided diagnosis of a murine model of chronic kidney
disease." Journal of Biomedical Optics 22, no. 12 (2017): 121706.

 [21]  Singh,  Vijendra,  Vijayan  K.  Asari,  and  Rajkumar  Rajasekaran.  "A  Deep  Neural
Network for Early Detection and Prediction of Chronic Kidney Disease." Diagnostics 12, no.
1 (2022): 116.
[22] Liao, Yun-Te, Chien-Hung Lee, Kuo-Su Chen, Chie-Pein Chen, and Tun-Wen Pai.
"Data   Augmentation   Based   on   Generative   Adversarial   Networks   to   Improve   Stage
Classification of Chronic Kidney Disease." Applied Sciences 12, no. 1 (2022): 352.
[23] Sabanayagam, Charumathi, Dejiang Xu, Daniel SW Ting, Simon Nusinovici, Riswana
Banu, Haslina Hamzah, Cynthia Lim et al. "A deep learning algorithm to detect chronic
kidney  disease  from  retinal  photographs  in  community-based  populations." The  Lancet
Digital Health 2, no. 6 (2020): e295-e302.
[24] Navaneeth, Bhaskar, and M. Suchetha. "A dynamic pooling based convolutional neural
network  approach  to  detect  chronic  kidney  disease." Biomedical  Signal  Processing  and
Control 62 (2020): 102068.
[25] Kriplani,  Himanshu, Bhumi  Patel, and Sudipta  Roy.  "Prediction  of  chronic kidney
diseases using deep artificial neural network technique." In Computer aided intervention and
diagnostics in clinical and medical images, pp. 179-187. Springer, Cham, 2019.
[26] Singh, Vijendra, Vijayan K. Asari, and Rajkumar Rajasekaran. "A Deep Neural Network
for Early Detection and Prediction of Chronic Kidney Disease." Diagnostics 12, no. 1 (2022):
116.
[27] Rafy, M. F. "Multivariate Statistical Analysis and Detection of Chronic Kidney Disease
Using Supervised Machine Learning Algorithms."
[28] Jerlin  Rubini, L., and Eswaran  Perumal.  "Efficient  classification  of chronic kidney
disease   by   using   multi‐kernel   support   vector   machine   and   fruit   fly   opt imization
algorithm." International Journal of Imaging Systems and Technology 30, no. 3 (2020): 660-
673.
[29]   Rezayi,   Sorayya,   KeivanMaghooli,   and   SoheilaSaeedi.   "Applying   Data   Mining
Approaches  for  Chronic  Kidney  Disease  Diagnosis." International  Journal  of  Intelligent
Systems and Applications in Engineering 9, no. 4 (2021): 198-204.
[30]  Parthiban,  R.,  S.  Usharani,  D.  Saravanan,  D.  Jayakumar,  Dr  U.  Palani,  Dr  D.
StalinDavid, and D. Raghuraman. "Prognosis of chronic kidney disease (CKD) using hybrid
filter  wrapper  embedded  feature  selection  method." European  Journal  of  Molecular  &
Clinical Medicine 7, no. 9 (2021): 2511-30.

