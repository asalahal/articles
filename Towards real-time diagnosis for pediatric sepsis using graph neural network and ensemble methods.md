European Rev                                                     iew for Med                       ical and Pharmacol                                ogical Sci          ences                                                                                                                                                                                              2021; 25: 4693-4701
Towards real -time diagnosis for pediatric
sepsis using graph neural net work and
ensemble methods
X . CH EN1, 2, R. ZH ANG1, 2, X.-Y. TANG1, 2
1Universit y of Chinese Academy of Sciences, Beijing, China
2Shanghai Institute of Technical Physics of the Chinese Academy of Sciences, Key L aborator y of
Infrared Detection and Imaging Technology, Shanghai, China
A bst ract .      –     OBJECTIVE : The rapid onset of                 es of data are obtained as input, the model ac -
pediatric sepsis and the short optimal time for                        curacy is close to using complete test data,
resuscitation    pose    a    severe    threat    to    children’s     which can help compress the time to diagnosis
health in the ICU. Timely diagnosis and inter-                         to   about   an   hour   af ter   the   test   and   significantly
vention are essential to curing sepsis, but there                      reduce waiting time.
is a lack of research on the prediction of sepsis                      Key Wo rds:
at shorter time intervals. This study proposes a                           Pediatric sepsis, Real-time diagnosis, Graph neural
predictive model towards real-time diagnosis of                        net work, Tree - based model, Ensemble method.
sepsis     to     help     reduce     the     time     to     first     antibiot-
ic treatment.
   PATIENTS AND METHODS : The dataset used
in this paper was obtained from the pediatric in -
tensive      care      unit      of      Shanghai      Children’s      Medi-                 Introduction
cal Center and consisted of the initial examina-
tion records of patients admitted to the hospi-                            Sepsis is a common complication of infection
tal. The data included six groups of laborato -                        in the pediatric intensive care unit (PICU) and
ry tests: medical history, physical examination,                       is also a leading cause of death1. Studies have
blood gas analysis, routine blood tests, sero -                        shown that the optimal resuscitation time for pa-
logical tests, and coagulation tests. We divid   -                     tients with sepsis is within 6 hou rs of the onset of
ed the admission examination into three stag -
es and proposed a sepsis prediction model to -                         the disease. Mor talit y may f u r ther increase if pa-
wards real-time diagnosis based on local infor-                        tients with severe sepsis are not promptly t reated
mation to shorten waiting time for treatment.                          with antibiotics. The mor tality rate increases by
The model extracts homogeneous features from                           6% for each one-hour delay in receiving antibiot-
patient groups in real-time using a graph neu-                         ic therapy af ter the onset of septic shock hy poten-
ral network and uses the deep forest to learn                          sion 2. The real-time prediction of pediat r ic sepsis
from homogeneous features and laboratory da   -
ta to give a comprehensive prediction at the                           is thus cr ucial to improving sur vival rates.
current stage. Discriminative features of each                             With the development of electronic health re-
stage are use d as augmente d information for the                      cords   (EHR)   and   ar tificial   intelligence,   research-
next phase, finally achieving self-optimization of                     ers have proposed methods that can provide accu-
global judgment, assisting in pre -allocation of                       rate prediction of sepsis at an early stage. Zhang
medical resources and providing timely medical                         et al3 used Lasso to constr uct a sepsis prediction
assistance to sepsis patients.                                         model,                while                Wang                et                al4 proposed a kernel
   R  E S U  LT  S  :        Based        on        the        first        stage,        second
stage, and full test, the AUCs of our model were                       extreme learning machine (KELM) to predict
93.63%, 96.73%, and 97.58%, respectively, and                          disease probabilities. Masino et al5 used the Ada-
the F1-scores were 77.35% , 85.71% , and 86.48% ,                      Boost algor ith m to predict patient data and found
respectively. The models gave relatively accu-                         that the performance of machine learning was
rate predictions at each stage.                                        not inferior to bacterial cult ure detection in pre-
   CONCLUSIONS : The prediction model toward                           diction. Le et al6 used g radient-en hanced decision
a real-time diagnosis of sepsis shows more ac -
curate predictions at each stage compared to                           trees (GBDT) to provide early warning of sep-
other  control  methods.  When  the  first  two  stag-                 sis in children. Recurrent neural network-based
Corresponding Author: Xiao Chen, MD; e - mail: nalanyu2000@163.com                                                               4693

approaches have also been proposed to process                                   tients to receive antibiotic treatment earlier. The
clinical data for sepsis with a temporal str uct ure                            prediction results g radually improve with g radual
(e.g.,      hear t      rate,      respiration,      blood      pressu re,      and dat a  refi nement.
blood oxygen). Futoma et al7   proposed Gaussian
process recu r rent neu ral net works to model phys-
iological      data,      while      Fagerst röm      et      al8 proposed                      Patients and Methods
a LiSep long shor t-ter m memor y (LSTM) model
for early sepsis detection. Bedoya et al9 devel-                                Study Population
oped the MGP-RN N model to verify that deep                                         The dataset for this paper was collected from
learning models can detect sepsis earlier and                                   the PICU at Shanghai Child ren’s Medical Center;
more accu rately than baseline methods. Although                                it contains the clinical records f rom 2010 to 2017
some   of   the   above   methods   use   the   patient’s   first               of   the   patients’   first   examination   af ter   ad mission
ad mission   examination   records   to   predict   sepsis,                     and the diag nostic results.
the lengthy complete examination can delay the                                      General diagnosis criteria for sepsis include
t reat ment of cr itically ill patients.                                        Sequential Organ Failure Assessment score (SO -
   Patients are usually given a thorough examina-                               FA) and Quick Sequential Organ Failure As-
tion upon admission to the ICU. The initial ICU                                 sessment          (qSOFA).          In          clinical          diag nosis,          organ
admission examination usually consists of six                                   dysfunction can be indicated by a SOFA with
tests: anam nesis, physical examination, blood gas                              an increase of more than or equal to 2 points.
analysis,                blood                routine                examination,                serolog y, The  higher  the  SOFA  score  gets,  the  more  severe
and coag ulation tests. It takes 3- 4 hou rs to obtain                          the patient’s condition is. Some indicators of the
the results for all of these tests. If the diag nosis is                        SOFA            scores            require            blood            test            results,            which
not made u ntil all of the test results are available,                          poses a bar rier to the employment of SOFA for
the best opportunity for the patient’s treatment                                patient’s assessment in clinical practice. Regard   -
may be missed.                                                                  ing        the        practicalit y,        exper ts        have        selected        th ree
   In       this       paper,       we       propose       a       sepsis       prediction clinically available indicators to const r uct a quick
model towards a real-time diagnosis that may                                    and   simplified   version   of   the   SOFA,   called   qSO-
predict sepsis infection based on par tial laborato-                            FA10. qSOFA scores include altered mental stat us,
r y data. The model aims to help shor ten the time                              systolic blood pressure less than or equal to 100
to       first       antibiotic       t reat ment       and       to       avoid       failu re m m Hg,         and         respirator y         f requency         less         than         22
to provide life-saving treatment to patients due                                breaths/min. qSOFA allows for rapid screening of
to waiting for time-consuming lab results. We                                   septic patients outside the ICU.
divide each test into three different groups ac-                                    Consider ing      that      this      is      a      ret rospective      st udy,
cording   to   the   time   to   get   results.   The   first   stage           the       final       diag nosis       of       sepsis       patients       was       deter-
includes            anam nesis,            physical            examination,            and mined by clinicians f rom a combination of qSO-
blood       gas       analysis,       which       allows       for       im mediate FA,  SOFA,  and  bacilli  cult u re  results.
results. The second stage compr ises routine blood                                  The inclusion criteria for patients in the data-
examination,      such      as      routine      blood      and      reactive   set were (a) age <18 years and (b) a PICU stay
proteins,        which        can        be        obtained        within        half        an greater than or equal to 2 days. The exclusion
hour to one hour. The last stage includes sero-                                 criteria were (a) propor tion of missing values in
logical and coag ulation tests, the results of which                            data <60% and (b) incomplete elect ronic medical
take at least 3- 4 hours to obtain.                                             records due to abandoned t reat ment or t ransfer.
   The main contributions of this paper can be                                      Af ter removing a large sample of missing val-
summarized as follows: (1) We propose a Balan-                                  ues, the dataset included a total of 3,298 patients,
ceEnsemble hybr id integ ration model that out per-                             including 445 cases of sepsis. Based on clinical
for ms comparable baseline methods; (2) we pro-                                 exper ience           and           previous           literat u re,           the           clinical
pose a multi-activations autoregressive moving                                  laborator y categories selected for this paper in-
average (MA-AR MA) graph neural network for                                     cluded                 medical                 histor y,                 physical                 examination,
patient homogeneity feature lear ning extraction;                               blood                gas                analysis,                routine                blood,                serological
and (3) we propose a phased sepsis prediction                                   tests,  and  coag ulation  tests.
model oriented toward real-time prediction. The                                     According to clinical experience and the ex-
proposed model can output accurate sepsis pre-                                  isting  literat u re,  we  chose  six  kinds  of  indexes —
diction           at           an           early           stage,           shor ten           the           time           re-anam nesis, physical examination, blood gas, rou-
quired  for  initial  diag nosis,  and  enable  sepsis  pa-                     tine      blood      examination,      ser um,      and      coag ulation
4694

Table I. Character istics of PICU patients.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            Total                                                                                                                                                                                     Sepsis
   Characteristics                                                                                                                                                                                                                                                                                                                                                                           Count                                                                                                                                                                                             (%)                                                                                                                                                                             Count                                                                                                                                                                         (%)
   Gender   Male  1282  38.8 274 61.6
           Female 2015            61.2  171  38.3
   Age     0-1   1264  38.3 162 36.4
          2-5   1197  36.3 146 32.2
                                                                                                                                                                                                          6-12589                                                                                                                                          17.989                                                                                                                                        20
                                                                                                                                                                                                          13-172477.548                                                                                                                                  10.7
tests —to         const r uct         the         model.         The         selection         of               has essentially the same str ucture and consists
these tests is based on diagnostic criteria such as                                                             of a g raph neu ral net work and t ree-based model.
SOFA,     qSOFA     and     practical     clinical     exper ience.                                             The details of the model are shown in Figure
The physical examination contains the items re-                                                                 1.            The            inputs            for            stage            1            include            anam nesis,
quired for the qSOFA score and is even more                                                                     physical          examination,          and          blood          gas          analysis.
comprehensive. Laboratory tests such as rou-                                                                    We also modeled the patients and their medical
tine      blood      examination,      ser um,      and      coag ulation                                       records as a graph network. In a graph network
have more comprehensive indicators than those                                                                   st r uct u re,     neighbor     nodes     with     similar     feat u res
involved in SOFA and can provide more valid                                                                     usually      have      the      same      label,      which      means      that
infor mation for the prediction                                                                                 the neighbors of sepsis patients are also likely
     According            to            their            consuming            time,            the            six to be sepsis patients. It is well k nown that sim        -
group examinations are divided into three stag-                                                                 ilar people share similar characteristics; this is
es.             The             first             stage             compr ises             medical             histor y, k nown as homogeneity. We therefore apply the
physical           examination,           blood           gas           analysis,           such                proposed MA-AR MA graph convolutional net    -
as              fever              histor y,              t umor,              poor              spir its,              cyanosis work (GCN) to extract homogeneous features
of          the          lip,          cyanosis,          t r i-ret raction          sig n,          pulmo-     for each data stage, and the deep forest is used to
nar y            rales,            temperat u re,            hear t            rate,            respirator y    learn high-dimensional discriminative features
rate,      SPO2,      blood      pressu re,      Lac,      PaCO2,      PaO2,                                    for the learned isomorphic features and input
O2SAT,                 PH,                 FreeCa,                 K+,                 Na+,                 CL-,                 HCO3-, data,      making      it      possible      to      provide      predictions
capillar y    refill    time    (CRT).    The    second    stage    in-                                         for the cu r rent st age. T he discr iminative feat u res
cludes     routine     blood     examination,     such     as     white                                         lear ned by each stage module are propagated to
cell cou nt, hemoglobin deter mination, hematocr it                                                             the next stage as augmented information for
deter mination,  platelet  cou nt,  neut rophil  number,                                                        further learning and providing discriminative
neut rophil   ratio,   monocy te   number,   ly mphocy tes                                                      infor mation  for  prediction.  That  is,  the  modules
number,             Ly mphocy te             ratio,             C-reactive             protein.                 in different stages are cascaded through the
The    third    stage    contains    ser um,    and    coag ulation                                             discr iminative feat u res propagated f rom the pre       -
test,      including      creatinine      (CR EA),      Total      Protein                                      vious stage. The input of the second stage is the
(TP),     total     bilir ubin     (TBIL),     albumin     (A LB),     as-                                      fusion vector of the routine blood examination
par tate            aminot ransferase            (AST),            U R EA,            Gam-                      and the discriminative features learned from
ma-glutamyl                   t ransferase                   (GGT),                   u nconjugated             the         first         stage.         The         final-stage         input         f uses         the
bilir ubin          ( U NCONJ.BILI),          conjugated          bilir ubin                                    second-stage discriminative features with sero      -
(CONJ.BILI),       proth rombin       time       (PT),       Activated                                          logical examination and coagulation tests. The
Par tial  Th romboplastin  Time  (A PTT),  fibr inogen                                                          model str ucture of the latter two stages is also
(FIB),      D -Dimer      (D -D),      th rombin      time      (TT),      in-                                  identical  to  that  of  the  first  stage.
ter national nor malized ratio (I N R).
                                                                                                                MA - ARMA Graph Convolutional
Predictive Models Towards                                                                                       Neural Network
Real-Time Diagnosis                                                                                                   Patients in the PICU can be divided into sepsis
     In       this       paper,       we       propose       a       th ree-stage       cas        -            and    non-sepsis    g roups,    and    in    this    st udy,    patient
cade model for real-time sepsis diagnosis that                                                                  clinical data are considered as graph data with
out puts predictions based on par tial data at each                                                             a bipartite structure. Each patient constitutes
stage. The cor responding module for each stage                                                                 a         node         in         the         g raph         net work,         and         the         patient’s
                                                                                                                                                                                                           4695

F i g  u r  e  1. Schematic diag ram of model st r uct u re: th ree identical cascade modules that make predictions based on the assay
results of the cor responding stage.
clinical records are used as at t r ibutes of the node.                       MA layer allows neurons to choose different
We computed the neighbor nodes for each pa-                                   activation functions to obtain different high-di-
tient node using the k-nearest neighbor algor ith m,                          mensional mapping vectors. The str uct ure of the
where k=5. We connected each patient with the                                 M A-A R M A GCN is show n in Fig ure 2.
five most similar patients. The homogeneous fea    -
t u res can illu minate infor mation beyond the indi-                         ARMA Convolutional Layers
vidual  patient  for  classification.                                            GCNs based on autoreg ressive moving average
   We propose an MA-ARMA GCN to extract                                       (A R M A)     filters     have     g reater     robust ness     and     can
homogeneous features among patients. The                                      approximate         a         var iet y         of         different         filter         shapes
MA-AR MA GCN is a multi-activation function                                   to provide more graph frequency responses. The
convolution     net work     based     on     the     A R M A     filter.     expression  of  the  K-order  A R M A  filter  is11,
Compared       with       com monly       used       poly nomial       fil-
ters,      A R M A      filters      can      fit      more      complex      sig nal
responses  and  are  more  robust,  thus  reducing  the                                                                                          (1)
dependence on the graph st r uct ure. An MA-AR-
MA network provides different activation func-                                   where L is the st andard Laplacian mat r ix of the
tions       for       each       layer       of       neu rons,       thus       generating g raph,  λ  is  the  eigenvalue  of  L.
different distributions of response signals. The                                 The output signal of the ARMA convolution
diverse         response         sig nals         increase         the         flexibilit y can be w r it ten as
and        diversit y        of        the        neu ral        net work,        which        can
effectively suppress the over smoothing problem
and improve the generalization ability of the                                                                                                    (2)
model.
   The MA-ARMA network consists of AR-                                           where X is the initial node feat ure.
MA convolutional layers (ARMA-Conv) and
multi-activations layers (M A). The A R M A- Conv                             MA-ARMA Convolution Layer
is a graph convolution layer composed of AR-                                     Traditional neural networks use uniform acti-
M A         g raph         filters11;         it         can         simulate         more         filters vation    f u nctions    for   each   layer,    such    as    all    relu   or
with better spectral processing capabilities. The                             leak y relu, so each net work layer can only fit a sin-
4696

                                                                               map low-dimensional data to a high-dimensional
                                                                               space. The different ker nel f unctions represent a
                                                                               diversity of var ying data mappings and high-di-
                                                                               mensional         projections,         and         the         increased         diver-
                                                                               sit y         also         means         a         g reater         possibilit y         of         finding
                                                                               hyper planes to par tition the data. A mathemati-
                                                                               cal expression for the out put X         –    (t +1)     of     the     (t+1)-th
                                                                               A R M A- Conv layer with a multivar iate activation
                                                                               f unction is as follows:
                                                                                                                                                  (5  )
                                                                               Hybrid Ensemble Model Based on
                                                                               BalanceEnsemble Method
                                                                                  There is a severe categor y imbalance problem
                                                                               in     the     dataset,     which     would     lead     to     poor     perfor-
                                                                               mance if used directly as input to the GCNs. We
                                                                               therefore proposed a BalanceEnsemble method
                                                                               and const r ucted a hybr id ensemble model (HEM)
                                                                               with it. The minority class sample is the sample
                                                                               of septic patients, also called the positive sample.
                                                                               The core idea of the BalanceEnsemble is to retain
                                                                               all positive samples and match different negative
                                                                               samples to for m a new t raining set for the model.
Figure 2. Schematic diag ram of the M A-A R M A GCN.                           The process of BalanceEnsemble is as follows:
                                                                               1.  Select all positive sample data.
                                                                               2. Randomly select a group of negative samples
gle data dist r ibution. We proposed an M A-A R M A                               whose total number is equal to the positive
GCN in which each activation layer in the net work                                samples and repeat the operation n times to get
consists          of          different          ker nel          activation          f u nctions, n groups of negative samples.
and neu rons can cor respond to different activation                           3. Combine all of the positive samples and dif-
f u nctions. M As can increase the flexibilit y and di-                           ferent groups of negative samples to obtain n
versit y of a neural net work and to a cer tain extent                            different t raining sets.
can alleviate the problem of over smoothing in                                 4. Train the model with n t raining sets and weigh
G C Ns12.  The  M A  f u nction  is  defined  as  follows:                        the  results  to  obtain  the  final  results.
                                                                   (3)            We built the HEM based on the BalanceEn-
                                                                               semble st rateg y, which consists of several parallel
                                                                   (4 )        feat u re-decision               branches,               each               of               consists               of
                                                                               two modules in series: a feature extractor and
                                                                               a       decision       classifier.       The       feat u re       ext ractor       is       an
   where D is a hyperparameter indicating the                                  M A-A R M A               net work,               which               ext racts               patients’
total number of different ker nel activation func-                             homogeneous     feat u res.     The     decision     classifier     is
tions in the cur rent M A layer;  denotes the adap-                            a deep forest, which predicts patients based on the
tive weighting coef ficients;  is a linear expression,                         input infor mation. The HEM acts as an infor ma-
which       can       be       inter preted       as       w xi+b       (we       use       the tion-processing      module      at      each      stage,      accepting
letter  to represent the single input of the func-                             the cor responding clinical data infor mation and
t  i o n);   k (…) denotes the one- dimensional Gaussian                       yielding the prediction. The embedding vectors
ker nel; and  denotes the dictionar y element. The                             learned in each stage of the HEM are used as
dictionary of the kernel function is determined                                augmentation feat ures to be f used into the input
by even steps of x with mean 0. γ є R denotes the                              of         the         next         stage.         The         specific         st r uct u re         of         the
bandwidth of the kernel. The kernel functions                                  HEM is show n in Fig ure 3.
                                                                                                                                              4697

                                                                                                                               Figure 3. Schematic diagram
                                                                                                                               of the hybrid ensemble model
                                                                                                                               f ramework.
    We    used    the    random    forest,    g radient    boosting                       to 2016 were used as the training set. K-fold
decision            t ree            (GBDT),            suppor t            vector            machine cross-validation (K=4) was used to verify algo-
(SV M), residual neu ral net work (ResNet), LSTM,                                         r ith m per for  mance. T he dat a f rom 2017 were used
and GCN as the baseline models for the experi-                                            as the test set to evaluate model per for mance. T he
ment. The experimental data were divided into                                             results are show n in Table II, Table III, and Table
a training set and a test set. The data from 2010                                         IV. The AUCs are show n in Fig ure 4.
Table II. Performance results for each model at stage 1.
  Name                                                             ACC                                                          AUC                                                        F1                                                Sensitivity                                  Specificity
  GCN13   0.8046  0.7595 0.3707          0.3704  0.8843
  LST M14  0.8103  0.8475 0.5285  0.6851  0.8333
  Logist ic15 0.8506  0.8850 0.5928  0.7037  0.8776
  ResNet16  0.8506  0.8644 0.5937           0.7037  0.8775
  MLP17   0.8591  0.8413 0.5950           0.667  0.8945
  SV M18   0.8621  0.8838 0.6190           0.7222  0.8878
  RF19    0.8908  0.9235 0.6935                                0 .7 9 6 3 0.9081
  GBDT20  0.9023  0.9289 0.7119           0.7778  0.9251
  DF21    0.9281  0.9227 0.7524           0.7037                                  0.9693
  Ours                                                                                                                                                                                                                                                                       0.9331                                                                                                                                                                                                                     0.9363                                                                                                                                                                                            0 .7 7 3 5 0.7592          0.9626
Table III. Performance results for each model at stage 2.
   Name                                                             ACC                                                          AUC                                                        F1                                                Sensitivity                                  Specificity
   GCN13   0.8218  0.7504 0.4364         0.4444  0.8911
   LST M14  0.8390  0.8799 0.5942           0.7592  0.8537
   ResNet16  0.8621  0.9196 0.6307          0.7592  0.8809
   MLP17   0.8793  0.8852 0.6315           0.6667  0.9183
   Logist ic15 0.8707  0.9363 0.6512           0.7778  0.8878
   SV M18   0.8879  0.9498 0.6609          0.8518  0.8911
   RF19    0.9080  0.9457 0.7333          0.8148  0.9251
   GBDT20  0.9367  0.9527 0.78                                                                   0. 8722                                                                                                                                                                                                                             0 . 9 761
   DF21    0.9396  0.9567 0.8037          0.7963  0.9659
   Ours                                                                                                                                                                                                                                                                      0.9568                                                                                                                                                                                                                    0.9673                                                                                                                                                                                            0. 8571 0.8333         0.9759
4698

Table IV. Performance results for each model at stage 3.
  Name                                                             ACC                                                          AUC                                                        F1                                                Sensitivity                                  Specificity
  GCN13   0.7816  0.8167 0.4412          0.5556  0.8231
  LST M14  0.8793  0.9132 0.6250          0.6481  0.9217
  ResNet16  0.8793  0.9110 0.6557          0.7407  0.9047
  MLP17   0.8851  0.9124 0.6875          0.8148  0.8979
  SV M18   0.8851  0.9505 0.7015          0.8703  0.8877
  Logist ic15 0.8965  0.9566 0.7049          0.7962  0.9149
  GBDT20  0.9109  0.9684 0.7596                             0 . 9  0 74 0.9115
  RF19    0.9195  0.9698 0.7704          0.8703  0.9285
  DF21    0.9511  0.9639 0.8316         0.7778                              0.9829
  Ours                                                                                                                                                                                                                                                                       0.9569                                                                                                                                                                                                                    0.9 75 8                                                                                                                                                                                           0.8648 0.8889         0.9693
                                Results                                                 and 0.8648 for the three stages. The next best
                                                                                        perfor mance      was       that      of      the      deep      forest      model,
    According to the performance tables for the                                         with  F1-scores  of  0.7524,  0.8037,  and  0.8316.  The
th ree         stages,         the         model         proposed         in         this         paper convolutional neural network-based models and
achieved better results than baseline on the pre-                                       the recur rent neural network-based models were
diction              task              with              F1-scores              of              0.7735,              0.8571, unable to achieve desirable performance. The
Figure 4. AUC for dataset; (                        a) AUC for phase 1; (b) AUC for phase 2; (c) AUC for stage 3; (d) AUC for stage 1 ablation
exper iment.
                                                                                                                                                              4699

F1-scores for the three stages for LSTM were                                                 model        achieves        a        more        sig nificant        perfor mance
0.5285,    0.5942,    and    0.6250,    respectively,    and    the                          improvement   compared   with   the   first   stage,   with
F1-scores           for           ResNet           were           0.5937,           0.6307,           and the AUC and F1-score improving by 3.31% and
0.6557,  re s p e ct ively.                                                                  10.8%,             respectively.             From             stage             2             to             stage             3,
    To verify the soundness and validity of the                                              the performance improvement of the model is
model    st r uct u re,    ablation    exper iments    were    con-                          smaller     compared     to     stage     2,     with     the     AUC     and
ducted to validate the performance of the var-                                               F1-score improving by 0.87% and 0.89%, respec-
ious modules. The models in the ablation ex-                                                 tively. The prediction from stage 2 can thus be
periments are referred to as the Model without                                               approximated     as     the     model’s     final     prediction     for
GN N   (M WG),   Model   without   GN N   and   Ensem-                                       patients. Considering that it takes only 1 hour to
ble   Lear ning   (M WGE),   and   A R M A-based   GN N                                      obtain     the     laborator y     results     for     stage     2,     we     can
(AGN N). We evaluated the ablation experiments                                               thus reduce the minimum time for patients to
on               the               first-stage               dataset,               and               the               results               are receive antibiotic t reat ment to 1 hour.
show n in Table V.                                                                                The perfor mance g radually decreased with the
                                                                                             removal of sub-modules from the cur rent mod-
                                                                                             el.        From        the        entire        model        to        the        M WG        model,
                               Discussion                                                    sensitivity decreased from 0.77778 to 0.7407.
                                                                                             According           to           the           equation           defining           sensitivit y,
    Combining the per for mance tables for all th ree                                        a decrease in sensitivity means reducing the
stages,       the       perfor mance       of       the       deep       neu ral       net-  number  of  cor rectly  identified  septic  patients  (t p)
work was far inferior to that of the tree-based                                              and         an         increase         of         misclassified         septic         patients
models. The main reason for this may be that the                                             (fn). The model became less sensitive to sepsis
tabular clinical laborator y data lack clear spatial                                         and tilts toward classif ying all patients as routine
structures or temporal correlations. Traditional                                             patients.      Specificit y      measu res      the      abilit y      to      cor-
machine         lear ning         models,         especially         t ree-based             rectly  identif y  reg ular  patients  (negative  sample),
models,       may       be       more       advantageous       when       deal-              so the negative bias of the model led to a slight
ing with tabular data because they divide the                                                r ise  in  specificit y.
high-dimensional space based on data attributes                                                   The Ensemble method was good at improving
to obtain a series of subspaces. The splitting                                               generalization and avoiding over-fit ting. The rep-
nodes in the t ree-based models are highly consis-                                           resentation lear ned from the GCN also brought
tent   with   the   st r uct u re   of   the   tabular   data,   which                       in   different   infor mation,   thus   cont r ibuting   to   the
is an impor tant reason to integ rate the t ree-based                                        accurate prediction of the total model from a
model    into    the    desig n.    As    show n    in    Table    V,    the                 new perspective. GN N extracted the individual
model proposed in this paper outperformed the                                                patient data feat ures and the homogeneity of the
other  models,  obtaining  notably  bet ter  results  for                                    neighborhood     data,     thus     en hancing     the     diversit y
the AUC and F1-score. Higher AUC and F1-score                                                of feat ures and improving model perfor mance.
mean more accu rate sepsis identification and low-
er false negatives without a sig nificant increase in
false p osit ive s.                                                                                                        Conclusions
    In terms of changes in model performance
at     each     stage,     the     AUC     and     F1-score     of     the     first              In       this       paper,       we       proposed       a       real-time       sepsis
stage     model     are     0.9363     and     0.7735,     respectively,                     diagnosis prediction model to decrease the time
which means that the model can im mediately ob -                                             to   first   antibiotic   t reat ment   for   patients   with   sep-
tain     a     relatively     accu rate     judg ment     based     on     the               sis. The model divides the six individual tests for
first-stage data alone. From stage 1 to stage 2, the                                         sepsis diagnosis into three stages according to
Table V. Results of ablation experiments.
   Name                                                             ACC                                                          AUC                                                        F1                                                Sensitivity                                  Specificity
   AGNN   0.8965  0.9082 0.6896 0.7407  0.9659
   M  WG E                                                                                                                                                                                                                                     0.9281 0.9227 0.7524 0.7037                               0.9693
   M  WG                                                                                                                                                                                                                                                           0.9281 0.9332 0.7619 0.7407  0.9625
   Ours 0.9252          0.9337                                                                                                                                                                                              0 .76 3 6                                                                                                                                                                                                                  0.7778 0.9523
4700

the time required for each test. Based on the data                                     6)                           Le    S,    Hof fman    J,    Bar ton    C,    Fitzgerald    JC,    Allen
available   at   different   stages,   ou r   model   predicted                      A, Pellegrini E, Calver t J, Das R. Pediatric Severe
the likelihood of sepsis in patients so that patients                                Sepsis Prediction Using Machine Learning. Front
with high prevalence could be t reated with timely                                   Pediatr 2019; 7: 413-428.
intervention. We validated the performance of                                         7)                       Futoma J, Hariharan S, Heller K. Learning to De-
                                                                                     tect Sepsis with a Multitask Gaussian Process
the model on a dataset collected from Shanghai                                       RNN Classifier 2017; 70: 1174-1182.
Children’s Medical Center. The results indicated                                8)             Fagerström J, Bång M, Wilhelms D, Chew MS.
that the cascaded model proposed in this paper                                       LiSep LSTM: A Machine Learning Algorithm for
achieved more accurate prediction in all cases                                       Early Detection of Septic Shock. Sci Rep 2019; 9:
compared to the baseline model. The cont r ibution                                   15132.
of each submodule was also demonstrated in the                                 9)           Bedoya AD, Futoma J, Clement ME, Corey K,
ablation st udy. Compared to previous st udies, ou r                                 Brajer N, Lin A, Simons MG, Gao M, Nichols M,
study provides a new perspective for predicting                                      Balu  S,  Heller  K,  Sendak  M,  O’Brien  C.  Machine
sepsis dynamically and in real time using data                                       learning for early detection of sepsis: an internal
                                                                                     and temporal validation study. JAMIA Open 2020;
f rom different stages, allowing patients to receive                                 3: 252-260.
timely             t reat ment.             In             the             f ut u re,             we             hope             that 10)               Seymour CW, Liu VX, Iwashyna TJ, Brunkhorst
our approach can be extended to the predictive                                       FM, Rea TD, Scherag A, Rubenfeld G, Kahn JM,
analysis           of           other           similar           conditions,           especially Shankar-Hari M, Singer M, Deutschman CS, Es -
those requiring timely diagnosis and therapeutic                                     cobar GJ, Angus DC. Assessment of Clinical Cri-
i nter  vent ion.                                                                    teria for Sepsis: For the Third International Con-
                                                                                     sensus     Definitions     for     Sepsis     and     Septic     Shock
                                                                                     (Sepsis-3). JAMA 2016; 315: 762-774.
                                                                              11)                   Bianchi FM, Grattarola D, Livi L, Alippi C. Graph
Conflict of Interest                                                                 Neural Networks with Convolutional ARMA Fil   -
The Authors declare that they have no con flict of interests.                        ters. IEEE Trans Pattern Anal Mach Intell 2021;
                                                                                     PP. doi: 10.1109/TPAMI.2021.3054830. Epub
                                                                                     ahead of print. PMID: 33497331.
                         References                                           12)          Scardapane S, Van Vaerenbergh S, Totaro S,
                                                                                     Uncini A. Kafnets: Kernel-based non-parametric
      1)                      Singer M, Deutschman CS, Seymour CW, Shan-             activation functions for neural networks. Neural
      kar-Hari M, Annane D, Bauer M, Bellomo R, Ber-                                 Netw 2019; 110: 19-32.
      nard GR, Chiche JD, Coopersmith CM, Hotchkiss                           13)                             Kipf T, Welling M. Semi-Super vised Classification
      RS, Lev y MM, Marshall JC, Mar tin GS, Opal SM,                                with Graph Convolutional Networks. ArXiv 2017;
      Rubenfeld GD, van der Poll T, Vincent JL, Angus                                abs/1609.02907.
      DC.         The         Third         International         Consensus         Defini-14)             Hochreiter S, Schmidhuber J. Long Short-Term
      tions for Sepsis and Septic Shock (Sepsis-3). JA-                              Memor y. Neural Computation 1997; 9: 1735 -1780.
      MA 2016; 315: 801-810.                                                  15)                Cramer, J. S. The origins of logistic regression.
   2)               Kumar A, Roberts D, Wood KE, Light B, Parril-                    2002.
      lo JE, Sharma S, Suppes R, Feinstein D, Zanotti                         16)                         Kaiming H, Zhang XY, Ren SQ, Sun J. Deep Re-
      S, Taiberg L, Gurka D, Kumar A, Cheang M. Du-                                  sidual Learning for Image Recognition. Computer
      ration of hypotension before initiation of effective                           Vision and Pattern Recognition 2016: 770 -778.
      antimicrobial therapy is the critical determinant                       17)             Rumelhart DE, Hinton GE, Williams RJ. Learn-
      of sur vival in human septic shock. Crit Care Med                              ing internal representations by error propagation.
      2006; 34: 1589-1596.                                                           Parallel distributed processing: explorations in
      3)                    Zhang Z, Hong Y. Development of a novel score            the microstructure of cognition 1988; 1: 318-362.
      for the prediction of hospital mortality in patients                    18)       Corinna C, Vapnik V. Support-Vector Networks.
      with severe sepsis: the use of electronic health-                              Machine Learning 1995; 20: 273-297.
      care records with LASSO regression. Oncotarget
      2017; 8: 49637-49645.                                                   19)         Breiman L. Random forest. Machine Learning
         4)                           Wang XC, Wang ZY, Weng J, Wen CC, Chen HL,     2001; 45: 5-32.
      Wang XQ. A New Effective Machine Learning                               20)                     Chen TQ, Guestrin C. XGBoost: A Scalable Tree
      Framework for Sepsis Diagnosis. IEEE Access                                    Boosting System. Knowledge Discovery and Da-
      2018; 6: 4 8 3 0 0 - 4 8 310.                                                  ta Mining. Proceedings of the 22nd ACM SIGKDD
     5)                   Masino AJ, Harris MC, Forsyth D, Ostapenko S,              International Conference on Knowledge Discov-
      Srinivasan  L,  Bonafide  CP,  Balamuth  F,  Schmatz                           er y and Data Mining 2016: 785-794.
      M, Grundmeier RW. Machine learning models for                           21)                              Zhou ZZ, Feng J. Deep Forest: Towards An Alter-
      early sepsis recognition in the neonatal intensive                             native to Deep Neural Networks. Proceedings of
      care unit using readily available electronic health                            the Twenty-Sixth International Joint Conference
      record data. PLoS One 2019; 14: e0212665.                                      on Artificial Intelligence 2017: 3553-3559.
                                                                                                                                              4701

