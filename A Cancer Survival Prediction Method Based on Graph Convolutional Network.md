This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                            Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                            1
                 A Cancer Survival Prediction Method Based on
                                           Graph Convolutional Network
                  Chunyu Wang1, Junling Guo1, Ning Zhao1, Yang Liu1, Xiaoyan Liu1, Guojun Liu1, Maozu Guo2,3,*
                  1. School of Computer Science and Technology, Harbin Institute of Technology, Harbin, 150001, China
                  2. School of Electrical and Information Engineering, Beijing University of Civil Engineering and Architecture,
                  Beijing 100044, China
                  3. Beijing Key Laboratory of Intelligent Processing for Building Big Data, Beijing 100044, China
                  *Correspondence: guomaozu@bucea.edu.cn
       ABSTRACT                                                                             numbers reached 14.10 million and 8.20 million respectively in
       Background and obejective: Cancer, as the most challenging part in the               2012. Morbidity and mortality rates of lung cancer are the
       human disease history, has always been one of the main threats to                    highest and those of gastric cancer, oesophageal cancer and
       human life and health. The high mortality of cancer is largely due to                hepatic cancer grow with each passing day. Meanwhile, cancers
       the complexity of cancer and the significant differences in clinical                 have become common among young people in recent years.
       outcomes. Therefore, it will be significant to improve accuracy of                   Cancer morbidity and mortality rates in the whole world have
       cancer survival prediction, which has become one of the main fields of               been rising. Cancer has become the main death cause since
       cancer  research.  Many  calculation  models  for  cancer  survival
       prediction have been proposed at present, but most of them generate                  2010 and it becomes one of the main public health problems in
       prediction models only by using single genomic data or clinical data.                the whole world[1]. Therefore, it’s urgent to design a more
       Multiple genomic data and clinical data have not been integrated yet                 accurate  calculation  method  for  cancer  survival  prediction,
       to take a comprehensive consideration of cancers            and predict their        which  can  contribute  to  development  of  individual-based
       survival.                                                                            treatment and management. Hence, this will also be conductive
       Method:  In  order  to  effectively  integrate  multiple  genomic  data              to lowering total death rate of cancers and further improving
       (including   genetic   expression,   copy   number   alteration,   DNA               living quality of cancer patients.
       methylation and exon expression) and clinical data and apply them to
       predictive studies on cancer survival, similar network fusion algorithm                 The heterogenous disease, cancer, has different molecular
       (SNF) was proposed in this paper to integrate multiple genomic data                  features,  clinical  behaviors,  morphological  appearances  and
       and clinical data so as to generate sample similarity matrix, min-                   different reactions to therapies[2-5]. In addition, complexity of
       redundancy  and  max-relevance  algorithm  (mRMR)  was  used  to                     invasive   cancers   and   their   clinical   outcomes   presenting
       conduct feature selection of multiple genomic data and clinical data of              significant changes result in extreme difficulties in prediction
       cancer samples and generate sample feature matrix, and finally two                   and treatment[6, 7]. Therefore, a more accurate prediction of
       matrixes  were  used  for  semi-supervised  training  through  graph
       convolutional  network  (GCN)  so  as  to  obtain  a  cancer  survival               cancer  prognosis  can  not  only  help  cancer  patients  to
       prediction method integrating multiple genomic data and clinical data                understand their expected life and can also help clinicians to
       based on graph convolutional network (GCGCN).                                        make wise decisions and give proper therapeutic guidance.
       Result: Performance indexes of GCGCN model indicate that both                        Meanwhile, prognosis plays a significant role in clinical work
       multiple genomic data and clinical data play significant roles in the                of  all  clinicians,  especially  those  clinicians  working  with
       accurate survival time prediction of cancer patients. It is compared                 patients with short survival. Being able to estimate prognosis
       with existing survival prediction methods, and results show that cancer              reasonably and accurately, clinicians generally make clinical
       survival   prediction   method   GCGCN   which   integrates   multiple
       genomic data and clinical data has obviously superior prediction effect              decisions with the help of prognostic prediction knowledge[8],
       than existing survival prediction methods.                                           confirm that patients accept therapeutic schemes[9] and design
       Conclusion: All study results in this paper have verified effectiveness              and analyze qualifications for clinical test. In addition, when a
       and superiority of GCGCN in the aspect of cancer survival prediction.                patient is predicted as short-survival patient, the clinician can
                                                                                            provide him/her with an opportunity to consider whether he/she
       Index Terms—       graph convolution network;multiple genomic data;                  wants to be cared, take time to take actual measures and make
       clinical    data;similarity    network    fusion;minimum    redundancy               preparations for the death[10].
       maximum relevance feature selection;cancer survival prediction;
                                                                                               In order to realize the goal, many researches have adopted
                                                                                            microarray technique to study genetic expression profiling of
       1. INTRODUCTION                                                                      cancers in the past years, but only some of them have displayed
             S morbidity and mortality rates gradually rise, cancer is                      definite prognostic significance[11, 12]. For instance, Van’s Veer
       A becoming the main death cause in the globe and one of                              et al., used DNA microarray to analyze 117 primary breast
       important  public  health  problems.  According  to  the  global                     cancer patients and used supervised classification method to
       cancer report, additional 12.7 million cancer cases occurred in                      identify prognostic features of 70 genes[13]. Moreover, they
       the globe with death toll reaching 7.6 million in 2008; the two                      tested these previously applicable prognostic markers among
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                           Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                          2
       295 breast cancer patients, and results indicated that prognostic                  prognostic prediction of GBM[28]. In consideration of multiple
       features of 70 genes had great significances[14]. Wang et al.,[15]                 different kinds of data features and when these features are
       revealed prognostic features of 76 genes and clustered genetic                     extensively applied to prognostic prediction of cancers, it’s not
       expression profiles and associated them with prognostic values,                    surprising that favorable effect is achieved. However, most of
       which could accurately predict tumor recurrence in the later                       these methods directly integrate different types of data into
       phase.  Microarray  markers  used,  some  machine  learning                        model generation while neglecting that features from different
       classification   methods   such   as   support   vector             machine        patterns (like genomic signature and clinic) may have different
       (SVM)[16-18], Naive Bayesian Classifier (NB)[19]               and random          representations. With the latest progress of the new generation
       forest (RF)[20]    are also used      to predict cancer survival. For              of  sequencing  technique,  multi-omics  cancer  diagnosis  and
       instance, Nguyen et al.,[21] proposed a breast cancer diagnosis                    prognosis   based   on   genetic   expression   profile,   clinical
       and prediction method based on random forest classifier and                        information and DNA copy number alteration have enjoyed
       feature  selection  technique,  and  the  result  was  superior  to                broad  development[29-31].  Therefore,  based  on  accelerated
       previously reported results.                                                       development of multi-omics data, it’s urgent to develop an
          In view of complexity and heterogeneity of cancer survival                      effective   computing   method   to   accurate   predict   cancer
       prediction, Brenton et al.,[21]        put forward a more pragmatic                prognosis.
       strategy and used clinical data and genetic prediction markers                         In  order  to  solve  these  problems  and  enlightened  by
       which  might  contain  some  supplementary  information.  In                       successful application of the present deep learning methods and
       addition, with rapid development of new technologies in the                        great contributions made by multi-dimensional data to cancer
       medical field, a large number of clinical cancer data have been                    prognosis  prediction,  a  cancer  survival  prediction  method
       generated and collected. Clinical data and microarray markers                      integrating multiple genomic data (including gene expression,
       combined, different computing methods have been developed                          copy number alteration, DNA methylation and exon expression)
       to  accurately  predict  cancer  survival[22,   23].  For  example,                and clinical data based on graph convolutional network namely
       Gevaert  et  al.,  developed  Bayesian  network[14],  integrated                   GCGCN was proposed in this study. The method considered
       clinical data and information of 70 genes through three different                  heterogeneity between different data types while taking full
       strategies  (including  complete,  decision-making  or  partial                    advantages of abstract high-level representations of different
       integration) and proved that combination of clinical data and                      data  sources.  In  order  to  verify  effectiveness  of             multiple
       microarray data had better or considerable performance than                        genomic     data and clinical data, GCGCN was compared with
       single use of clinical data or microarray data. Khademi et al.,[24]                different   independent   models   which   only   used               multiple
       put forward an interesting strategy, reduced dimensionalities of                   genomic      data  or  clinical  data.     Results  indicated  that  both
       microarray  data  using  manifold  learning  and  deep  belief                     multiple genomic         data and clinical data could improve the
       network and integrated clinical data through the probabilistic                     prediction performance of cancer survival, and this meant that
       graph  model.  Through  a  large  quantity  of  experiments,                       both multiple genomic data and clinical data reflected cancer
       compared with traditional classification methods, this method                      survival  from  different        aspects.  In  addition,  the  proposed
       has more excellent effects.          Besides microarray and clinical               GCGCN   method   was   compared   with   present   popular
       information, human reference protein interaction network has                       classification  methods.  In  the  aspect  of  cancer  survival
       also been used to predict survival of breast cancer. Das et al.,[25]               prediction, GCGCN achieved the optimal effect in multiple
       designed a method based on elastic network and named it                            evaluation indexes, thus proving the feasibility of integrating
       Encapp,  and  then  combined  protein  network  and  genetic                       multiple  genomic         data  and  clinical  data  as  well  as  the
       expression dataset in order to accurately predict survival time                    significance of GCGCN to cancer survival prediction.
       of human breast cancer. However, limitation of Encapp lies in
       that accuracy of survival prediction highly depends on quality                     2. MATERIALS AND METHODS
       of genetic expression datasets. As cancer is a very complicated                        Fig.  1  shows  experimental  framework  of  the  proposed
       disease, combination of genetic expression profiling data and                      GCGCN,  which  is  divided  into  three  steps:  (1)  generating
       clinical data may improve accuracy of prognosis and diagnosis                      sample similarity matrix; (2) generating sample feature matrix;
       by the prediction model[26]. Based on 70 genetic expression                        (3)  obtaining  cancer  survival  classifier  through  training.
       features, Sun et al., further identified a hybrid feature through                  Specifically speaking, the whole process is divided into three
       prediction of genetic features of breast cancer prognosis and                      steps. First of all, similarity network fusion algorithm (SNF)
       combination  of  clinical  markers[8].  Therefore,  it  is  still  of              was used to integrate multiple genomic data and clinical data to
       considerable  space  to  improve  prediction  performance  of                      obtain a sample similarity matrix A, and then min-redundancy
       cancer survival by combining more cancer-related information.                      max-relevance feature selection algorithm (mRMR) was used
          Besides  successes  achieved  through  the  abovementioned                      to  conduct  feature  selection  of  multiple  genomic  data  and
       methods, some new methods are proposed, Multi                  -dimensional        clinical  data  to  obtain  the  optimal  feature  combination.
       data  are  integrated  and  applied  to  human  cancer-related                     According to these optimal features, a sample feature matrix X
       prediction fields. Hayes et al., determined related microRNA                       could  be  established.  X  and  A  were  placed  in  the  graph
       and mRNA features of patients with high and low risks of                           convolutional  network  (GCN)  for  classified  training  and
       glioblastoma (GBM)[27]. Zhang et al., put forward a multi-core                     prediction, and finally a cancer survival prediction model was
       machine learning method integrating different types of data for
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                              Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                                 3
                                                                  Fig. 1 GCGCN experimental framework[32,37]
       built based on graph convolutional network               .                             where Fig.2-a is mRNA expression and DNA methylation of
       2.1 Experimental Data                                                                  patients of the same type; Fig.2-a             is patient-patient similarity
          Breast  cancer  (BRCA)  and  Lung  squamous  cell  cancer                           matrix of each data type; Fig.2-c             is patient-patient similarity
       (LUSC) were taken as study               objects, where cancer genomic                 network, node represents patient, and edge represents similarity
       data and clinical data mainly came from TCGA database. As                              between two patients; Fig.2-d is network fusion process. Each
       the  largest  cancer  genetic  information  database  at  present,                     network is upgraded iteratively through information from other
       TCGA is comprehensive in aspect of numerous cancer types but                           networks using SNF algorithm so that each step is more similar;
       more in aspect of genomic data record with high confidence                             e is continuous iteration and fusion of networks until the final
       level, including gene expression, copy number alteration, DNA                          fusion network is obtained through convergence, and edge color
       methylation, exon expression, clinical information, etc. For                           expresses   that   data   type   has   contributed   to   the   given
       patient samples used in this chapter, their multiple genomic data
       and clinical data were obtained from TCGA library. According
       to ID matching of patient samples, 249 BRCA patient samples
       and 220 LUSC patient samples were finally obtained, and each
       sample covered detailed information of four types of genomic
       data  and  clinical  data.  5  years  and  3  years  were  taken  as
       thresholds to divide two types of patients with two cancer types.
       Samples were divided into long-survival and short-survival
       patients   according   to   the   thresholds,   and   meanwhile,                                        Fig. 2 Similarity network fusion algorithm[32]
       classification label of short-survival samples was set as 1 and                        similarity[32].
       that  of  long-survival  samples  was  set  as  0,  and  concrete                         For SNF model, assuming that there are m different data
       information is seen in Tab. 1.                                                         types. For the v(th) data type                            , similarity matrix
                                                                                                     of all patients and K-nearest similarity matrix                         are
               Tab. 1 Information summary of BRCA and LUSC patients                           respectively  calculated,the  calculated  formula  is  shown  as
                   Cancer type                    BRCA                  LUSC                  follows. Two data types are taken as the example namely m =
            Total number of samples                249                   220                  2.
                Threshold (year)                    5                      3
             Long-survival patients                 70                   109                  similarity matrix:                                                             (1)
             Short-survival patients               179                   111
            Average age (years old)               57.19                 68.29
                                                                                              K-nearest similarity matrix:                                                  (2)
       2.2 Similarity Network Fusion
          The flow of similarity network fusion is shown in                      Fig. 2,
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                             Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                              4
       Where               is a scaled exponential similarity matrix.                        classification   variables.   However,   a   single   good   feature
       Step1:Calculate                and           of each data type                        combination can’t improve the classifier performance in feature
             Calculate          and         through equation (1), and calculate              selection,  because       features  may  be  highly  relevant,  which
                                                                                             causes   redundancy   of   feature   variables.   Hence,   min-
             and   through        equation      (2),so   that                       and      redundancy max-relevance feature selection algorithm comes
                               represent  two  initial  state  matrixes  under               into being, namely maximizing relevance between features and
             t=0(t is time).                                                                 classification variables while minimizing relevance between
                                                                                             features, and this is core idea of mRMR[33], which is a common
       Step2:Update the similarity matrix                                                    dimension      reduction      algorithm      enjoying      extensive
              Similarity matrix of each data type is iteratively updated                     application[34-36]. Therefore, mRMR feature selection method
          as         follows:                                                                was   used   to   select   features   from   the   original   dataset,
                                                                                    (3)      dimensionality of the dataset was reduced while significant
                                                                                             information loss was not caused, optimal combination features
                                                                                    (4)      in each genomic part were selected, and feature dimensionality
                                                                                             was reduced, thus improving generalization ability of the model.
       Step3:Standardization                                                                    In the specific experimental steps, we need to iterate to find
                      and           obtained  in  Step2  are  substituted  into              the optimal feature, so this method mainly uses incremental
             formula (1) for standardization.                                                search method to select features. For example, when we have
       Step4:Result output                                                                   obtained the optimal feature subset                , the next goal is to find
             After  t  steps,  the  output  overall  state  matrix  can  be                  the         feature from the remaining feature set                        , and
             calculated as:                                                                  maximize             by selecting features. The process of feature
          As a general rule, when m>2, the following formula (5) can                         selection  is  shown  in  Figure  3-4.  The  incremental  search
       be obtained according to formulas (3) and (4)                                         algorithm finds the optimal feature by optimizing the following
                                                                                             conditions:
                                                                                    (5)                                                                                   (6)
          Through the above method, a patient network graph structure
       integrating     multiple  genomic        data  and  clinical  data  can  be           2.4 Graph convolutional network
       finally   obtained.   SNF   algorithm   captures   shared   and                          Since 2012, deep learning has achieved enormous success in
       supplementary   information   from   different   data   sources.                      two fields—computer vision and natural language processing.
       Information quantity of similarities observed between samples                         However, its study object is still restricted to Euclid field data.
       of each data type should be deeply understood. As it is based on                      In the real world, many important datasets exist in forms of
       sample network, useful information can be obtained even from                          graph or     grids. Graph is a data format. Graph convolutional
       a small quantity of samples, and meanwhile, it is of robustness                       network algorithm model on the graph structure was considered
       to noise and data heterogeneity.                                                      in this paper.
                                                                                                Spectral convolution of graph is defined[37] as the product of
       2.3 Min-redundancy max-relevance feature selection                                    signal             and filter                   :
          A common problem which will be encountered when high-
       throughput sequencing dataset is used to predict survival time                                                                                                          (7)
       of human cancers is so-called “curse of dimensionality”. In our                          Where U is a matrix consisting of eigenvectors of normalized
       study, the sample size was limited while sample features were                         graph   Laplacian   matrix.   Graph   Laplacian   (matrix)   is
       obtained   by   combining   gene   expression,   copy   number                                                          ,     is diagonal matrix consisting of
       alteration, DNA methylation, exon expression and clinical data,
       so   dimensionality   of   sample   features   was   considerably                     eigenvalues of L and                is graph Fourier transform of               .
       enormous,  and  then  sample  feature  dimensionality  was  far                       According to related literatures, Chebyshev polynomial
       larger than sample size, which brought about certain difficulties                     can be used for approaching, and the following formula is
       to model learning and prediction. Because high dimensionality                         obtained:
       of  dataset  and  small  sample  size  can  easily  cause  model
       overfitting and the model fits in with the training set too much,                                                                                                  (8)
       prediction effects are very poor for untrained validation set and
       test  set.  Hence,  for  problems  involving  a  large  number  of                    Where                        .
       features, feature selection plays a critical role in success of a
       learning algorithm.                                                                      Graph   convolutional   neural   network   model   can   be
          A  common  feature  selection  method  is  maximizing  the                         constituted by stacking multiple convolutional layers in the
       relevance between features and classification variables, namely                       form of formula (8). Hereby number of convolutional layers is
       selecting the first k variables with the highest relevance with
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                            Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                             5
       restricted as k=1, and meanwhile, approximate                          is taken,     takes this sample network as the basis for integrating multi-
       and then:                                                                            omics  data,  aiming  at  utilizing  complementation  in  data.
                                                                                            Relative to analytical methods established for single data, SNF
                                                                                      (9)   algorithm has been verified having considerable advantages in
          Furthermore,  overfitting  can  be  avoided  by  constraining                     aspects of identifying tumor subtype and prediction. Hence,
       number of parameters while operation frequency at each layer                         multiple  genomic  data  and  clinical  data  can  be  integrated
       is minimized, and the following formula is obtained:                                 through SNF algorithm to finally obtain a sample similarity
                                                                                            matrix. As multiple genomic data and clinical data have high
                                                                                       (10) feature dimensionalities, mRMR algorithm was used for feature
                            is  set  in  (9).  But  eigenvalue  interval  of                selection, and some optimal features were taken respectively for
                                                                                            splicing to obtain a sample feature matrix. Sample similarity
                          is  [0,2].  In  the  deep  neural  network  model,                matrix and sample feature matrix integrating multiple genomic
       repetitive operation of this operation may result in numerical                       data  and  clinical  data  were  used  to  establish  the  graph
       instability and gradient explosion/vanishing. In order to solve                      convolution-based cancer survival prediction model GCGCN.
       this       problem,       “re-normalization”       is       introduced:              2.6 Comparison with other cancer survival prediction
                                            , where                                    .    methods
                                                                                               In  order  to  verify  effectiveness  of  the  proposed               graph
          This definition is promoted to signal                        (N is number         convolutional   network-based   cancer   survival   prediction
       of nodes and C is dimensionality of node eigenvector) and F                          method integrating multiple genomic data and clinical data,
       filters or feature map with C channels:                                              GCGCN was compared with other six models namely CGCN
                                                                                            (clinical  data),  GGCN  (multiple  genomic  data  integrated),
                                                                                        (11)GeneExpr (gene expression), DNAmethy (DNA methylation),
          Where                    is filter parameter matrix and                           CNA   (copy   number   alteration)   and   ExonExpr   (exon
       is signal matrix after convolution. Complexity of this filter                        expression). Main differences of these methods from GCGCN
                                                                                            lie in that they have not completely integrated multiple genomic
       operation is                     .        can be regarded as the product             data  and  clinical  data,  but  instead,  they  only  use  multiple
       of a sparse matrix and a dense matrix.                                               genomic data or data of other single types.
       2.5 Cancer survival prediction model GCGCN                                              In  order  to  verify  effectiveness  of  the  proposed  cancer
          Two problems exist in the cancer survival prediction namely                       survival  prediction  method  GCGCN,  five  commonly  used
       too small sample size and high sample feature dimensionality.                        classification   methods   were   adopted   for   a   comparison,
       When the sample size is too small, different training samples                        respectively being naïve Bayesian classification (NB)[38, 39],
                                                                                            K-nearest neighbor classification (KNN), logic regression (LR),
       will result in great differences in trained classification models                    decision     tree  (DT)  and  support  vector  machine  (SVM)[40].
       and then give rise to a large variance in the model classification                   Meanwhile, in order to verify effectiveness of the proposed
       result. How to reasonably utilize all samples and guarantee a                        method integrating        multiple genomic         data and clinical data,
       certain stability of the trained classification model has become                     dataset was divided into three groups—multiple genomic data,
       the   difficulty   in   the   cancer   survival   prediction.   Graph                clinical data and multiple genomic data + clinical data—in this
       convolutional network not only considers features and labels of                      contrast experiment.
       training  samples  but  also  will  continuously  acquire  related
       features  from  adjacent  points  (namely  samples)  through                                 Tab. 2 Data division of training set, validation set and test set
       adjacent matrixes or similarity matrixes of samples, so it will                           Cancer            Data            Training      Validation        Test
       take full consideration features of all samples and labels of                                          Long-survival          119             20            40
       training samples in the training process, thus ensuring stability                          BRCA        Short-survival          40             10            20
       of  the  trained  graph  convolutional  model  under  different                                        Total sample           159             30            60
       training  samples,  and  meanwhile,  the  graph  convolutional                                         Long-survival           79             10            20
       model can obtain considerable or better effect under a small                               LUSC        Short-survival          81             10            20
                                                                                                              Total sample           160             20            40
       sample size.
          Therefore, graph convolutional network was selected in this
       study  to  establish  a  cancer  survival  prediction  model  and                    3. EXPERIMENT
       establish sample similarity matrix or sample adjacent matrix                         3.1 Experimental setting
       and sample feature matrix needed by the graph convolutional                             In    this    paper,    the    total    dataset        (available    from
       model.  It’s  necessary  to  integrate  multiple  genomic  data                      https://xenabrowser.net)          was randomly divided into training
       (including gene expression data, copy number alteration, DNA                         set, Validation set and test set according to the proportion 7:1:2,
       methylation and exon expression) and clinical data in this study,                    and  concrete  data  division  is         seen  in  Tab.  2.  In  order  to
       but similarity network fusion (SNF) establishes the network of                       guarantee fairness and robustness of research methods, datasets
       samples of each available data type (like patient or gene) and                       were  randomly  divided  for  5  times,  experiment  on  each
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                                 Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                                       6
                                           Tab. 3 Numbers of optimally selected different features of two cancers—BRCA and LUSC
                                           Cancer type                                 BRCA                                         LUSC
                                             Data type                num of original        num of optimal        num of original       num of optimal
                                                                          features               features              features              features
                                         Gene expression                   20531                    50                  20531                   50
                                     Copy number alteration                24777                    50                  24777                   50
                                        DNA methylation                   364737                    50                 365863                   50
                                         Exon expression                  239323                    50                 239323                   50
                                       Clinical information                  26                     24                    22                    20
                                  Fig. 3 BRCA SNF converging curve                                                     Fig. 4 LUSC SNF converging curve
                      Fig. 5 Performance comparison of BRCA models                                             Fig. 6 Performance comparison of LUSC models
       research  method  was  carried  out  for  5  times,  and  final                            information are deleted, residual features are retained, and all
       evaluation indexes were average values of 5 experiments                       .            of these features are combined to obtain the sample feature
           In this experiment, hyper-parameters k=20 and μ=0.5 are                                matrix X. Concrete results are seen in Tab. 3.
       taken in SNF algorithm and threshold value is taken as , namely                            3.2 Experimental results
       when  relative  error  in  the  iteration  process  satisfies,  the                           As shown in Fig. 3 and Fig. 4, in the similarity network
       algorithm can stop the iteration. In the GCN algorithm model,                              fusion  process  of  two  cancers,  rapid  convergence  can  be
       a three-layer graph convolutional network is uniformly adopted,                            realized,  but  in  order  to  satisfy  the  condition  for  iterating
       respectively being input layer, hidden layer containing 40 nodes                           termination, 1,500 times of iterations are needed.
       and  output  layer.  In  mRMR  feature  selection,  50  optimal                               GCGCN was compared with other six models respectively,
       features of gene expression profile, copy number alteration,                               namely CGCN (clinical data), GGCN(multiple genomic data
       DNA methylation and exon expression are respectively selected,                             integrated),GeneExpr  (gene             expression),  DNAmethy  (DNA
       two  features—survival  time  and  survival  state—in  clinical                            methylation), CNA           (copy number         alteration) and ExonExpr
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                            Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                             7
      Tab. 4 Comparison chart of multiple performance evaluation indexes of seven Tab. 5 Comparison chart of multiple performance evaluation indexes of seven
                                        BRCA models                                                                            LUSC models
              BRCA           Precison        Accuracy          Recall          AUC                  LUSC           Precison         Accuracy          Recall         AUC
             GCGCN            0.7698           0.7933          0.9850         0.9280               GCGCN            0.5539           0.7200          0.8400         0.8050
              CGCN            0.7577           0.7500          0.9200         0.8350                CGCN            0.4800           0.6534          0.6800         0.7350
              GGCN            0.6807           0.6833          0.9900         0.7805                GGCN            0.5149           0.6867          0.7200         0.7840
            GeneExpr          0.6841           0.6867          0.9850         0.5849              GeneExpr          0.3506           0.5600          0.3600         0.5750
           DNAMethy           0.6667           0.6667          1.0000         0.6537             DNAMethy           0.4746           0.6400          0.6000         0.6950
               CNA            0.6689           0.6667          0.9900         0.5555                 CNA            0.3516           0.4733          0.6400         0.5200
            ExonExpr          0.6667           0.6667          1.0000         0.6180              ExonExpr          0.4518           0.5867          0.5600         0.6480
           Fig. 9 Comparison histogram of multiple average performances of                  Fig. 10 Comparison histogram of multiple average performances of
                           seven different BRCA models                                                                            seven different LUSC models
       (exon expression). Results corresponding to these models were                        Recall and AUC—were respectively calculated for each model,
       compared using ROC curve and AUC value. Fig. 5 and Fig. 6                            where Precion, Accuracy and Recall were all measured under
       show average ROC curve chart and AUC mean value when the                             the threshold value 0.5.
       experiment   was   randomly   repeated   for   5   times.   When                        As shown in Tab. 4, Tab. 5, Fig. 9 and Fig. 10, the proposed
       multiplegenomic data and clinical data were integrated in the                        method integrating multiple genomic data and clinical data has
       model,  the  obtained  prediction  performance  was  obviously                       higher prediction performance than other comparative models
       superior to any other model. For BRCA, the AUC value of the                          in all indexes. For instance, for BRCA, Precion, accuracy, recall
       used model integrating multiple genomic data and clinical data                       and AUC value of the proposed method are 0.7698, 0.7933,
       was  0.93,  while  the  value  was  0.58,  0.65,             0.56  and  0.62         0.9850 and 0.9280 respectively with Precion, accuracy and
       respectively  in      gene    expression  model,  DNA  methylation                   AUC value higher than those of the method only using multiple
       model,  copy number           alteration   model  and  exon expression               genomic data by 0.0891, 0.1100 and 0.1475 respectively, and
       model. Meanwhile, AUC value when only clinical data were                             meanwhile        their  difference  in  recall  is  minor).  Precion,
       used  was  0.84.  When  the  method  integrated  four  types  of                     accuracy, recall and AUC value are higher than those of the
       genomic data, model AUC value could reach 0.78 which was                             method only using clinical data by 0.0121, 0.0433, 0.0650 and
       obviously superior to the model established based on single                          0.0930 respectively. For LUSC, Precion, accuracy, recall and
       genomic data. As for LUSC, AUC value of the proposed model                           AUC value of the proposed method are 0.5539, 0.7200, 0.8400
       integrating multiple genomic data and clinical data was 0.81,                        and 0.8050 respectively, which are higher than those of the
       while the value was 0.57, 0.69, 0.52 and 0.65 respectively in                        method only using multiple genomic data by 0.0390, 0.0333,
       gene expression model, DNA methylation model, copy number                            0.1200 and 0.0210 respectively and higher than those of the
       alteration model and exon expression model. Meanwhile, AUC                           method only using clinical data by 0.0739, 0.0666, 0.1600 and
       value when only clinical data were used was 0.73. When the                           0.0700 respectively. To sum up, the proposed model integrating
       method integrated four types of genomic data, model AUC                              multiple genomic data and clinical data has obviously superior
       value could reach 0.78 which was obviously superior to the                           prediction performance indexes than prediction methods only
       model established based on single genomic data. The above two                        using single genomic data or clinical data in the aspect of cancer
       results  indicated  that  the  proposed  integration  of  multiple                   survival  prediction,  indicating  that  both  clinical  data  and
       genomic data and clinical data reflected survival time of cancer                     multiple genomic data can reflect their influences on concrete
       patients from different aspects, thus providing a certain help for                   cancer survival time from different aspects.
       medical personnel to formulate concrete medical methods.                                Survival analysis expresses a statistical method considers
          In  addition,  in  order  to  further  verify  reliability  of  the               both result and survival time. Moreover, it can give full play to
       proposed model integrating multiple genomic data and clinical                        incomplete information provided by censored data to describe
       data. Four performance evaluation indexes—               Precion, Accuracy,          distribution  features  of  survival  time  and  analyze  main
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                                Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                                      8
             Tab. 6 Comparison of BRCA AUC value of GCGCN with existing                               Tab. 7 Comparison of LUSC AUC value of GCGCN with existing
                                      classification methods                                                                   classification methods
            BRCA       Multiple genomic           Clinical data         Multiple genomic            LUSC        Multiple genomic           Clinical data        Multiple genomic
                               data                                    data + clinical data                             data                                    data + clinical data
             NB         0.7593±    0.5318       0.7675±    0.0494       0.8683±    0.4549             NB         0.6885±    0.0796       0.7212±   0.0622        0.7831±    0.0797
            KNN         0.6210±    0.0904       0.7383±    0.0660       0.7038±    0.0848            KNN         0.6015±    0.0900       0.5325±   0.0797        0.6355±    0.0409
             LR         0.7663±    0.0392       0.8065±    0.0626       0.9080±    0.0319             LR         0.6985±    0.1263       0.7330±   0.0295        0.7660±    0.0411
             DT         0.6300±    0.0509       0.6650±    0.0365       0.7500±    0.0622             DT         0.5475±    0.0406       0.5750±   0.1140        0.5950±    0.0400
            SVM         0.7735±    0.0503       0.8055±    0.0628       0.8855±    0.0226            SVM         0.7145±    0.0236       0.7012±   0.0764        0.7560±    0.0471
            DNN         0.7765±    0.1021       0.8145±    0.0920       0.9080±    0.0510            DNN         0.7265±    0.1302       0.7122±   0.0902        0.7721±    0.0890
          GCGCN         0.7805±    0.0553       0.8350±    0.0520       0.9280±    0.0260          GCGCN         0.7840±    0.0491       0.7350±   0.0574        0.8050±    0.0301
                      Fig. 11 KM survival curves of BRCA prediction category                 Fig. 12 KM survival curves of LUSC prediction category
                         Fig. 13 KM survival curves of BRCA original category                 Fig. 14 KM survival curves of LUSC original category
       influence factors of survival time. Fig. 11 and Fig. 12 shows test                        GCGCN method have significant differences.
        sets  of  two  cancers,  which  are  divided  into  two  groups                             In order to verify favorable effect of the proposed method in
       according to classification results obtained                 through GCGCN,               cancer survival prediction, this method is hereby compared with
       KM  curves  are  drawn,  and  their  P  values  are  calculated                           five common classification methods. Concrete classification
       according to the curves, where left picture is BRCA and right                             effects of the models are seen in Tab. 6 and Tab. 7. For BRCA,
       picture  is  LUSC.  Both  P  values  are  smaller  than  0.05.                    In      when multiple genomic data + clinical data are used, average
       addition,     as shown in Fig. 13 and Fig. 14,                 we list the KM             AUC  value  of  GCGCN  is  0.9280  while  those  of  five
       survival curves and P values of the original groups of two kinds                          classification methods—          NB, KNN, LR, DT, SVM and DNN—
       cancer data to compare the results above. Therefore, the cancer                           are  0.8683,  0.7083,  0.9080,  0.7500,                 0.8855      and    0.9080
       survival classification results of test sets using the proposed                           respectively. Meanwhile, when only multiple genomic data or
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                          Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                          9
       clinical data are used, average AUC value obtained through                         still has certain limitations in the aspect of cancer survival. For
       GCGCN is 0.7805 and 0.8350 respectively, which are also                            example,   this   research   work   can   realize   extension   and
       obviously higher than those of other methods, and meanwhile,                       verification based on a larger sample size (cancer patients). The
       the variance is small. For LUSC, when multiple genomic data                        present sample size is restricted by availability of multiple
       + clinical data are      used, average AUC value of GCGCN is                       genomic data and clinical data. It’s estimated that when more
       0.8050, and similarly, the proposed GCGCN method has higher                        cancer samples are available in the future, performance of this
       average  AUC  value  and  smaller  AUC  variance  than  other                      method can be significantly improved. In            addition, it’s believed
       methods. In a similar way, it can be seen from Tab. 6 and Tab.                     that if cancers are classified by their subtypes and GCGCN
       7 that for most classification methods, prediction performance                     model  is  established  for  each  subtype,  this  will  be  more
       based on integrated multiple genomic data and clinical data has                    significant for cancer researchers and its performance may be
       been improved to a certain degree. For instance, for BRCA,                         further improved, because cancer survival prediction                is greatly
       when multiple genomic data + clinical data are used by NB                          influenced by cancer subtypes           [41-43]. Unfortunately, for each
       algorithm, the obtained average AUC value increases by 0.1090                      subtype of cancer patients, available data size is very small.
       and 0.1008 respectively when compared with methods only                            Therefore, when there are more available samples in the future,
       using  multiple  genomic  data  or  clinical  data.  When  LR                      the concrete analysis of cancer subtypes will be our extension
       algorithm uses multiple genomic data + clinical data, average                      direction. Another future research direction is t          o integrate more
       AUC value increases by 0.1417 and 0.1015 respectively when                         genomic data like protein expression, miRNA expression                   [44-47]
       compared with methods only using multiple genomic data or                          and other genomic data[48-50]. Meanwhile, pathological image
       clinical data. It can also be found that the proposed GCGCN has                    features of cancer patients may be considered in the future
       overall small variance, because in the training process of graph                   research work. The final goal is to establish a multitask learning
       convolutional network model, the model considers both global                       system   for   different   cancer   researches,   including   cancer
       information  and  sample  relevance,  and  then  a  more  stable                   susceptibility prediction, cancer recurrence and cancer therapy.
       prediction result can be obtained.
                                                                                          AUTHOR CONTRIBUTIONS
       4. DISCUSSION AND CONCLUSIONS                                                      Funding by C.W., M.G., Y.L. and X.L.; Ideas initiated by M.G.
          As cancer is one of the most common and malignant diseases                      and Y.L.; Manuscript drafted by C.W. and J.G.; Manuscript
       in the world, this study was dedicated to improving prediction                     review and editing by X.L., Y.L.; Experiments implemented by
       performance   of   cancer   survival.   A   graph   convolutional                  J.G., N.Z. and G.L.; Paper fin         alized by C.W.; and all authors
       network-based  cancer  survival  prediction  method  GCGCN                         have read and approved the final manuscript.
       integrating  multiple  genomic  data  and  clinical  data  was
       proposed in this paper, where multiple genomic data included                       ACKNOWLEDGMENTS
       gene expression, copy number alteration, DNA methylation and                       The work was supported Natural Science Foundation of China
       exon expression. First of all, multiple genomic data and clinical                  (No. 91735306, 61872114, 61671189, 61671188, 61571163
       data  were  integrated  using  the  similarity  network  fusion                    and   61871020),   and   the   National   Key   Research   and
       algorithm,  sample  similarity  matrix  was  obtained,  cancer                     Development Plan Task of China (No. 2016YFC0901902). The
       survival-related features were extracted using min-redundancy                      funders had no role in study design, data collection and analysis,
       max-relevance   mRMR   feature   selection   algorithm,   the                      decision to publish, or preparation of the manuscript.
       influence of useless features was mitigated, and classification                    Conflicts of interest: The authors declare no conflict of interest.
       training  and  prediction  were  conducted  through  the  graph
       convolutional network. In order to explore into effectiveness of                   REFERENCES
       multiple  genomic  data  and  clinical  data  in           cancer  survival        1.         Torre, L.A., et al., Global cancer statistics, 2012. 2015.
       prediction, compared with existing cancer survival prediction                                  65(2): p. 87-108.
       methods only using multiple genomic data or clinical data,                         2.          Rakha,   E.A.,   et   al.,      Breast   cancer   prognostic
       indexes—ROC curve, AUC, Recall, Accuracy, Precion, etc.—                                       classification  in  the  molecular  era:  the  role  of
       were compared in this paper, and results indicated that both                                   histological grade. 2010. 12(4): p. 207.
       multiple genomic data and clinical data had a bearing on cancer                    3.          Balacescu,     O.,     et     al.,     Blood     genome-wide
       survival in different aspects. Therefore, integration of multiple                              transcriptional  profiles  of  HER2  negative  breast
       genomic data and clinical data could improve prediction effect                                 cancers patients. 2016. 2016.
       on  cancer  survival.  Meanwhile,  five  common  classification                    4.          Liao,  Z.,  et  al.,     Cancer  diagnosis  from  isomiR
       methods—NB,  KNN,  LR,  DT  and  SVM                   —were  applied  to                      expression  with  machine  learning  method.              Current
       cancer survival prediction. Through a comparison, the proposed                                 Bioinformatics, 2018. 13(1): p. 57-63.
       GCGCN method had higher average AUC value and lower                                5.          Yu,  L.,  et  al.,   Inferring  drug-disease  associations
       AUC standard deviation, so GCGCN showed obviously more                                         based  on  known  protein  complexes.            Bmc  Medical
       excellent   classification   effect   relative   to   o     ther   common                      Genomics, 2015. 8: p. 13.
       classification methods.                                                            6.         Martin, L.R., et al., The challenge of patient adherence.
          Even though GCGCN has excellent classification effect, it                                   2005. 1(3): p. 189.
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                          Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                        10
       7.         Yu, L., et al., Prediction of new drug indications based                           cancer   diagnosis   and   prognostic.           J.   Biomedical
                  on clinical data and network modularity.                Scientific                 Science and Engineering, , 2013. 6: p. 551-560
                  Reports, 2016. 6.                                                       21.        Brenton,  J.D.,  et  al.,     Molecular  classification  and
       8.         Sun,  Y.,  et  al.,   Improved  breast  cancer  prognosis                          molecular  forecasting  of  breast  cancer:  ready  for
                  through  the  combination  of  clinical  and  genetic                              clinical application?       J Clin Oncol, 2005. 23(29): p.
                  markers. 2006. 23(1): p. 30-37.                                                    7350-60.
       9.         Xu,  X.,  et  al.   A  gene  signature  for breast  cancer              22.        Y. Sun, S.G., J. Li, L. Liu, W.Farmerie,                Improved
                  prognosis using support vector machine. in 2012 5th                                breast cancer prognosis through the combination of
                  International Conference on BioMedical Engineering                                 clinical and genetic markers. Bioinformatics, 2006. 23:
                  and Informatics. 2012. IEEE.                                                       p. 30–37.
       10.        Stone, P. and S.J.A.o.o. Lund, Predicting prognosis in                  23.        A.-L.  Boulesteix,  C.P.,  M.  Daumer,  ,           Microarray-
                  patients with advanced cancer. 2006. 18(6): p. 971-                                based   classification   and   clinical   predictors:   on
                  976.                                                                               combined classifiers and additional predictive valu.
       11.        L.J. Van’t Veer, H.D., M.J. Van De Vijver, Y.D. He,                                Bioinformatics, 2008. 24: p. 1698–1706.
                  A.A. Hart, M. Mao, et al., Gene expression profiling                    24.        M. Khademi, N.S.N., Probabilistic graphical models
                  predicts clinical outcome of breast cancer.               Nature,                  and  deep  belief  networks  for  prognosis  of  breast
                  2002. 415: p. 530-536.                                                             cancer, , in in: Machine Learning and Applications
       12.        Yu, L., J. Zhao, and L. Gao,             Predicting Potential                      (ICMLA), 2015 IEEE 14th International Conference
                  Drugs for Breast Cancer based on miRNA and Tissue                                  on, 2015. p. 727–732.
                  Specificity.     International    Journal    of    Biological           25.        J. Das, K.M.G., F. Bunea, M.H. Wegkamp,,                   H. Yu,
                  Sciences, 2018. 14(8): p. 971-980.                                                 ENCAPP:elastic-net-based prognosis prediction and
       13.        D.M. Abd El-Rehim, G.B., S.E. Pinder, E. Rakha, C.                                 biomarker   discovery   for   human   cancers,.              BMC
                  Paish, J.F. Robertson, et al., High-throughput protein                             genomics, 2015. 16 p. 263.
                  expression    analysis    using     tissue     microarray               26.        M. Khademi, a.N.S.N., "." pp. 727-732., Probabilistic
                  technology   of   a   large   well-characterised   series                          Graphical  Models  and  Deep  Belief  Networks  for
                  identifies biologically distinct classes of breast cancer                          Prognosis  of  Breast  Cancer,  in           2015  IEEE  14th
                  confirming recent cDNA expression analyses. Int. J.                                International Conference on Machine Learning and
                  Cancer, 2005. 116: p. 340–350.                                                     Applications (ICMLA)2015.
       14.        M.J. Van De Vijver, Y.D.H., L.J. Van’t Veer, H. Dai,                    27.        J. Hayes, H.T., C. Tumilson, A. Droop, M. Boissinot,
                  A.A.  Hart,  D.W.  Voskuil,et  al,          A  gene-expression                     T. A. Hughes, D. Westhead, J. E. Alder, L. Shaw, and
                  signature as a predictor of survival in breast cancer.                             S.   C.  Short,     Prediction  of  clinical  outcome  in
                  N. Engl. J. Med, 2002. 347: p. 1999–2009.                                          glioblastoma     using     a     biologically     relevant
       15.        Y. Wang, J.G.K., Y. Zhang, A.M. Sieuwerts, M.P.                                    ninemicroRNA signature. Molecular oncology, 2015.
                  Look,  F.  Yang,  et  al,      Gene-expression  profiles  to                       9(3): p. 704-714.
                  predict    distant  metastasis  of  lymph-node-negative                 28.        Y. Zhang, A.L.,  C. Peng, and M. Wang,                   Improve
                  primary breast cancer. Lancet North Am. Ed, 2005.                                  glioblastoma  multiforme  prognosis  prediction  by
                  365: p. 671–679.                                                                   using feature selection and multiple kernel learning.
       16.        X. Xu, Y.Z., L. Zou, M. Wang, A. Li, A gene signature                              IEEE/ACM  transactions  on  computational  biology
                  for  breast  cancer  prognosis  using  support  vector                             and bioinformatics, 2016. 13(5): p. 825-835.
                  machine, in Biomedical Engineering and Informatics                      29.        K. Tomczak, P.C., and M. Wiznerowicz, , The Cancer
                  (BMEI), 2012 5th International Conference on2012. p.                               Genome Atlas (TCGA): an immeasurable source of
                  928–931.                                                                           knowledge,. Contemp Oncol (Pozn),, 2015. vol. 19,
       17.        Chen, W., et al.,       i6mA-Pred: Identifying DNA N6-                             (no. 1A, ): p. pp. A68-A77,.
                  methyladenine     sites     in     the     rice     genome.             30.        J. Gao, B.A.A., U. Dogrusoz, G. Dresdner, B. Gross,
                  Bioinformatics, 2019.                                                              S. O. Sumer, Y. Sun, A. Jacobsen, R. Sinha, and E.
       18.        Chen,  W.,  et  al.,    iDNA4mC:  identifying  DNA  N4-                            Larsson, ,     “Integrative analysis of complex cancer
                  methylcytosine      sites  based  on  nucleotide  chemical                         genomics     and     clinical     profiles     using     the
                  properties.    Bioinformatics,  2017.  33(22):  p.  3518-                          cBioPortal,Science  signaling,  .          vol.  6,,  2013(  no.
                  3523.                                                                              269, ): p. pp. pl1, .
       19.        O. Gevaert, F.D.S., D. Timmerman, Y. Moreau, B. De                      31.        Yu, L., J. Zhao, and L. Gao, Drug repositioning based
                  Moor,    Predicting the prognosis of breast cancer by                              on triangularly balanced structure for tissue-specific
                  integrating   clinical   and   microarray   data   with                            diseases    in    incomplete    interactome.            Artificial
                  Bayesian  networks.         Bioinformatics,  2006.  22:  p.                        Intelligence in Medicine, 2017. 77: p. 53-63.
                  e184–e190.                                                              32.        Bo  Wang,  A.M.M.,  Feyyaz  Demir,  Zhuowen  Tu,
       20.        C. Nguyen, Y.W., and H.N. Nguyen, Random forest                                    Michael  Brudno,  Benjamin  Haibe-Kains,    Anna
                  classifier combined with feature selection for breast                              Goldenberg:  ,       Similarity    network    fusion    for
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/TNB.2019.2936398, IEEE
                                                                           Transactions on NanoBioscience
       > REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <                                                                         11
                  aggregating  data types on a genomic scale. . Nature                     47.        Jiang, L., et al., MDA-SKF: Similarity Kernel Fusion
                  Methods, , 2014. 11, : p. 333-337.                                                  for      Accurately      Discovering      miRNA-Disease
       33.        H. Peng, F.L., and C. Ding,, “Feature selection based                               Association. Frontiers in Genetics, 2018. 9(618): p. 1-
                  on mutual information criteria of max-dependency,                                   13.
                  max-relevance,     and     min-redundancy,”                   IEEE       48.        Zhang, X., et al., Meta-path methods for prioritizing
                  Transactions   on   pattern   analysis   and   machine                              candidate disease miRNAs. IEEE/ACM Transactions
                  intelligence, 2005., vol. 27, no. 8: p. 1226-1238.                                  on Computational Biology and Bioinformatics, 2018.
       34.        C.  Ding,  a.H.P.,        “Minimum   redundancy  feature                            Doi: 10.1109/tcbb.2017.2776280.
                  selection from microarray gene expression data,.                   ”     49.        Zeng,  X.,  et  al.,       Probability-based        collaborative
                  Journal of bioinformatics and computational biology,                                filtering     model     for     predicting     gene–disease
                  2005. 3(02): p. 185-205, .                                                          associations. BMC Medical Genomics, 2017. 10(5): p.
       35.        Dao,  F.Y.,  et  al.,     Identify  origin  of  replication  in                     76.
                  Saccharomyces  cerevisiae  using  two-step  feature                      50.        Liu,   Y.,   et   al.,   Inferring   MicroRNA-Disease
                  selection technique. Bioinformatics, 2018.                                          Associations by Random Walk on a Heterogeneous
       36.        Yang,  H.,  et  al.,      iRNA-2OM:  A  Sequence-Based                              Network with Multiple Data Sources. IEEE/ACM
                  Predictor  for  Identifying  2'-O-Methylation  Sites  in                            Transactions   on   Computational   Biology   and
                  Homo sapiens. J Comput Biol, 2018. 25(11): p. 1266-                                 Bioinformatics, 2017. 14(4): p. 905-915.
                  1277.
       37.        Thomas      N.      Kipf,      M.W.,           Semi-Supervised
                  Classification with Graph Convolutional Networks in
                  5th     International     Conference     on     Learning
                  Representations      (ICLR   2017)2017:   Palais   des
                  Congrès Neptune, Toulon, France.
       38.        Feng, P.M., H. Lin, and W. Chen,                Identification of
                  antioxidants from sequence information using naive
                  Bayes. Comput Math Methods Med, 2013. 2013: p.
                  567529.
       39.        Feng, P.M., et al., Naive Bayes classifier with feature
                  selection to identify phage virion proteins.              Comput
                  Math Methods Med, 2013. 2013: p. 530696.
       40.        Feng, C.Q., et al., iTerm-PseKNC: a sequence-based
                  tool     for     predicting     bacterial     transcriptional
                  terminators. Bioinformatics, 2018.
       41.        Z. Liu, X.-S.Z., S. Zhang, , Breast tumor subgroups
                  reveal diverse clinical prognostic power,.              Sci. Rep.,
                  2014. 4: p. 4002.
       42.        C.  Desmedt,  B.H.-K.,  P.          Wirapati,  M.  Buyse,  D.
                  Larsimont, G. Bontempi,et al., , Biological processes
                  associated with breast cancer clinical outcome depend
                  on the molecular subtypes,. Clin. Cancer Res, 2008. 14
                  p. 5158–5165.
       43.        Limin Jiang, Y.X., Yijie Ding, Jijun Tang, Fei Guo,
                  Discovering Cancer Subtypes via an Accurate Fusion
                  Strategy  on  Multiple  Profile  Data.              Frontiers  in
                  Genetics, 2019.
       44.        Tang, W., et al., Tumor origin detection with tissue-
                  specific   miRNA   and   DNA   methylation   markers.
                  Bioinformatics, 2018. 34(3): p. 398-406.
       45.        Wang, Q., et al., Briefing in family characteristics of
                  microRNAs and their applications in cancer research.
                  Biochimica    Et    Biophysica    Acta-Proteins    And
                  Proteomics, 2014. 1844(1): p. 191-197.
       46.        Limin Jiang, Y.X., Yijie Ding, Jijun Tang, Fei Guo,
                  FKL-Spa-LapRLS: an accurate method for identifying
                  human     microRNA-disease     association.                  BMC
                  Genomics, 2019. 19(911): p. 11-25.
1536-1241 (c) 2019 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

