                                                                                                                            TYPE       Original Research
                                                                                                                            PUBLISHED    20 May 2024
                                                                                                                            DOI 10.3389/fpubh.2024.1397260
                                                      Integrating gated recurrent unit in
OPEN ACCESS                                           graph neural network to improve
EDITED BY
Dmytro Chumachenko,                                   infectious disease prediction: an
National Aerospace University – Kharkiv
Aviation Institute, Ukraine
                                                      attempt
REVIEWED BY
Luigi Di Biasi,
University of Salerno, Italy                          Xu-dong Liu 1,2, Bo-han Hou 1,2, Zhong-jun Xie 1, Ning Feng 3* and
Xinqiang Chen,                                                                  4,5
Shanghai Maritime University, China                   Xiao-ping Dong               *
Ricardo Valentim,
Federal University of Rio Grande do Norte,            1Faculty of Information Technology, Beijing University of Technology, Chaoyang District, Beijing,
Brazil                                                China, 2Key Laboratory of Computational Intelligence and Intelligent Systems, Beijing University of
                                                      Technology, Chaoyang District, Beijing, China, 3Office of International Cooperation, Chinese Center
*CORRESPONDENCE                                       for Disease Control and Prevention, Chaoyang District, Beijing, China, 4National Institute for Viral
Ning Feng                                             Disease Control and Prevention, Chinese Center for Disease Control and Prevention, Chaoyang
  fengning@chinacdc.cn                                District, Beijing, China, 5National Key-Laboratory of Intelligent Tracking and Forecasting for Infectious
Xiao-ping Dong                                        Disease, National Institute for Viral Disease Control and Prevention, Chinese Center for Disease
  Dongxp@chinacdc.cn                                  Control and Prevention, Chang-Bai, Beijing, China
RECEIVED 07 March 2024
ACCEPTED 04 April 2024
PUBLISHED 20 May 2024
CITATION                                              Objective: This study focuses on enhancing the precision of epidemic time series
Liu X-d, Hou B-h, Xie Z-j, Feng N and Dong            data prediction by integrating Gated Recurrent Unit (GRU) into a Graph Neural
X-p (2024) Integrating gated recurrent unit in
graph neural network to improve infectious            Network (GNN), forming the GRGNN. The accuracy of the GNN (Graph Neural
disease prediction: an attempt.                       Network) network with introduced GRU (Gated Recurrent Units) is validated by
Front. Public Health 12:1397260.                      comparing it with seven commonly used prediction methods.
doi: 10.3389/fpubh.2024.1397260
COPYRIGHT                                             Method: The GRGNN methodology involves multivariate time series prediction
© 2024 Liu, Hou, Xie, Feng and Dong. This is          using a GNN (Graph Neural Network) network improved by the integration of
an open-access article distributed under the          GRU (Gated Recurrent Units). Additionally, Graphical Fourier Transform (GFT)
terms of the Creative Commons Attribution             and Discrete Fourier Transform (DFT) are introduced. GFT captures inter-
License (CC BY). The use, distribution or
reproduction in other forums is permitted,            sequence correlations in the spectral domain, while DFT transforms data
provided the original author(s) and the               from the time domain to the frequency domain, revealing temporal node
copyright owner(s) are credited and that the          correlations. Following GFT and DFT, outbreak data are predicted through one-
original publication in this journal is cited, in
accordance with accepted academic                     dimensional convolution and gated linear regression in the frequency domain,
practice. No use, distribution or reproduction        graph convolution in the spectral domain, and GRU (Gated Recurrent Units)
is permitted which does not comply with               in the time domain. The inverse transformation of GFT and DFT is employed,
these terms.
                                                      and final predictions are obtained after passing through a fully connected layer.
                                                      Evaluation is conducted on three datasets: the COVID-19 datasets of 38 African
                                                      countries and 42 European countries from worldometers, and the chickenpox
                                                      dataset of 20 Hungarian regions from Kaggle. Metrics include Average Root
                                                      Mean Square Error (ARMSE) and Average Mean Absolute Error (AMAE).
                                                      Result: For African COVID-19 dataset and Hungarian Chickenpox dataset,
                                                      GRGNN consistently outperforms other methods in ARMSE and AMAE across
                                                      various prediction step lengths. Optimal results are achieved even at extended
                                                      prediction steps, highlighting the model’s robustness.
                                                      Conclusion: GRGNN proves effective in predicting epidemic time series data
                                                      with high accuracy, demonstrating its potential in epidemic surveillance and early
                                                      warning applications. However, further discussions and studies are warranted to
                                                      refine its application and judgment methods, emphasizing the ongoing need for
                                                      exploration and research in this domain.
                                                      KEYWORDS
                                                      artificial intelligence technology, graph neural network, gated recurrent unit,
                                                      infectious disease, time series prediction
Frontiers in Public Health                                                   01                                                                frontiersin.org

Liu et al.                                                                                                                               10.3389/fpubh.2024.1397260
1 Introduction                                                                        diseases has obvious seasonality, and by referring to the changes in
                                                                                      active cases in previous years, one can roughly predict the current
    Multivariate time series forecasting plays a crucial role in various              trend of active case changes. The data from a certain point or period
real-world scenarios such as transportation forecasting (1, 2), supply                in the time series is related to the data from the current or future time
chain management (3), energy allocation (4,                 5) and financial          points, which reflects the correlation between the time nodes within
investment (6). The time series prediction is involves forecasting                    each single time series. On the other hand, the number of active cases
future values based on historical data points in a sequential order. This             in a certain area may be related to the case numbers in neighboring
makes the statical method and supervised learning method,                             areas or areas with frequent personnel movement. These time series
comparing with reinforcement learning (7,                8) or unsupervised           may exhibit leading, lagging, or even synchronous trends, which
learning methods, are more suitable for this task. In the field of public             demonstrates the correlation between different points within the time
health, the problem of acute epidemic forecasting is of great relevance               series. Deep learning models provide new ideas for this problem: on
as a typical multivariate time series forecasting: if the future evolution            the one hand, Temporal Convolutional Network (TCN) (18) has
of acute epidemic data can be  estimated quickly and accurately for                   excellent results in single time series prediction. Recurrent Neural
each geographic region, the forecasting results can be     used as a                  Network (RNN) based methods (19–             21) such as LSTM (Long Short-
reference to help governmental agencies make decisions on policy                      Term Memor y) (22), Gated Recurrent Unit (23), Gated Linear Unit
formulation and material deployment, and thus prevent the                             (GLU) (24) have good results in single time series prediction. GLU can
development and spread of epidemics.                                                  effectively capture and learn the correlation and nonlinear features
    The field of epidemiolog y and public health research has witnessed               among time nodes within a time series (24). Han et al. (          25) compared
a large number of studies on time series prediction of infectious                     the prediction effects of ARIMA, deep neural network (DNN), and
diseases which revealed the requirement of prediction method in the                   LSTM (Long Short-Term Memory) network for occupational
field of epidemiology and public health research. A selection of                      pneumoconiosis data in Tianjin, China, and proved that LSTM (Long
notable works has contributed to this progress, showcasing innovative                 Short-Term Memory) can effectively predict occupational
approaches and methodologies for forecasting and managing disease                     pneumoconiosis data, and at the same time has an advantage in
outbreaks. For instance, Pinto et al. (     9) applied a regressive model to          prediction accuracy comparing to DNN and ARIMA. There is an
estimate intervention effects over time by comparing rates of                         advantage in prediction accuracy. However, most of these models
congenital syphilis. Cori et al. (10) presents a novel tool for tracking              ignore the dependencies between multiple variables and can only
the spread of diseases by estimating time-varying reproduction                        capture and learn the features within a single time series in isolation,
numbers. Du et   al. (11) focus on the research of serial interval of                 which makes them perform poorly in practical multivariate time
COVID-19 which contribute to the foundation of transmission                           series prediction problems.
dynamics of COVID-19 and is essential for effective prediction and                         Meanwhile, in the problem of mining relationships between
control measures. However, when facing the outbreak of acute                          sequences, Yu et      al. used matrix decomposition to model the
epidemic, the traditional transmission dynamics may be uncapable to                   relationship between multiple time series (26). Discrete Fourier
prediction task. For example, in 2020, Ioannidis et al. (12) found that               Transform (DFT) is also useful in time series analysis by introducing
traditional transmission models failed in forecasting of COVID-19.                    it. For example, State Frequency Memor y Network (27) combines the
And many research attempt to apply machine learning method to                         advantages of DFT and LSTM (Long Short-Term Memor y) for stock
handle the problem. Dairi et    al. (13) compared 7 kinds of neural                   price prediction; Spectral Residual model (28) utilized DFT to achieve
network in the prediction of the number of COVID-19 cases. In fact,                   desirable results in time series anomaly detection. Another important
the neural networks were also applied to the prediction problem of                    aspect of multivariate time series forecasting is modeling the
other epidemics. Sanchez-Gendriz et al. (14) applied Long Short-Term                  correlation between multiple time series. For example, in traffic
Memory (LSTM) network in the prediction of dengue outbreak in                         prediction tasks, neighboring roads naturally interact with each other.
Natal, demonstrates the potential of neural network in disease                        The state-of-the-art models rely heavily on graph convolution
surveillance at a local scale. And It is worthwhile to research the                   networks (GCNs) derived from graph Fourier transform (GFT)
potential of neural network in epidemic time series data prediction.                  theor y (29). These models (1, 2) directly stack GCNs and temporal
    Early time series forecasting research mainly relied on traditional               modules (e.g., LSTM (Long Short-Term Memory), GRU (Gated
statistical models, including historical average (HA), autoregressive                 Recurrent Unit)), which require predefined graph-structured
(AR), autoregressive integrated moving average (ARIMA) (15), VAR                      relationships between sequences. By simultaneously capturing the
(16), and fuzzy methods (17). All of these statistical models rely on                 dependencies between time nodes within each single sequence and
inherent a priori assumptions and require an artificial analysis of the               between different time series to improve the learning of features of the
characteristics of the study population to determine the applicability                time series and thus improve the prediction accuracy. Convolutional
of the forecasting method.                                                            Neural Network (CNN) has a good performance in learning local
    Accurate prediction of multivariate time series data is a                         features (30). There have been several methods to model spatial
challenging type of time series forecasting problem, because both the                 features using CNNs (31–       35). Ma et al. ( 34) used deep CNN for traffic
correlation between the time nodes within each single time series and                 speed prediction. Huang et al. (36) tried to use transformer to predict
the correlation between the time series need to be     considered                     multiple time series variables and obtained good prediction results.
comprehensively. During the outbreak of an infectious disease in a                         The introduction of GRU (Gated Recurrent Unit) units provides
certain area, the changes in the number of active cases, on one hand,                 better learning and fitting capabilities in the time domain compared
is related to the number of existing active cases in the locality or                  to the linear units used in general GNN (Graph Neural Network)
previous epidemic data. For instance, the outbreak of some infectious                 research methods. In addition, the above processes are modularized
Frontiers in Public Health                                                         02                                                                    frontiersin.org

Liu et al.                                                                                                                                                                      10.3389/fpubh.2024.1397260
when implemented. Individual modules can be connected in series                                                preprocessing layer, the GRGNN module layer, and the output layer,
by shortcut connection to further improve the prediction accuracy of                                           and the overall structure is shown in Figure 1                       .
the neural network by constructing a deep network. Due to the                                                        The input is a multivariate time series data X                                                              x={}itÎNT´
advantages of RNN methods, such as LSTM (Long Short-Term                                                       containing T         time nodes in          N columns, and before being processed
Memor y) and GRU (Gated Recurrent Unit), comparing with normal                                                 layer by layer by the deep neural network, a graph structure
feed-for ward neural networks, exist clear advantages in time series                                           GX          W=     {}, describing the relationship between the input data is  rst
prediction, there have been a large number of attempts to use RNNs                                             obtained through the smoothing module and the graph building
combined with GNNs (Graph Neural Networks), CNNs, or other                                                     module, where           X   is the data of each node in the input, and                      WNN´       is
neural network architectures to predict multivariate time series: Lv                                           the connection weight matrix between each node.                               GX          W=     {}, is fed
et al. ( 33) combined RNN with CNN, where the RNN are responsible                                              into the GRGNN module layer and the output layer a er several
for mining and learning intra-sequence time series within single                                               rounds of training and learning to obtain the  nal prediction result
                                                                                                                       ì                                       ü
sequence features, and CNN captures the relationships between                                                   XX                                                                                                               =¼ïTT                                                                                                        TH++                                                                                                                      +12,,                                           ,XXï
sequences. Luo et al. (           37) introduced GRU (Gated Recurrent Unit)                                            í                                       ý. Where T           is the number of time
                                                                                                                       ï                                       ï
                                                                                                                       î                                       þ
into GCN to predict the change of gas composition in transformer oil                                           nodes of the input time series data and                      H is the prediction step size.
during transformer operation. Zhang et al. (                       38) proposed ST-ResNet                      A mathematical description of the above process can be expressed in
based on residual convolutional network for crowd flow prediction.                                             Equations 1, 2        :
Shi et   al. (  20) combines convolutional network with LSTM (Long                                                                               GgraphstructX=                                                                                                                                                                                                                                  () (1)
Short-Term Memory) network to extract spatio-temporal
information separately.
      Graph neural networks have also yielded many results in                                                                     ì                                                                                       ü
                                                                                                                                  ïXX,,¼¼,,XFï                                          XG
capturing dependencies among unstructured data (1, 2, 7, 29, 39–                                   43).                           í    TT                                                                                                                               TH++                                                                                                                                                               +12ý()(2)
                                                                                                                                  ï                                          ï       =
DCRNN (1) and STGCN (2) are two of the first studies to introduce                                                                 î                                          þ
graph convolutional networks into spatio-temporal data prediction for
better modeling of spatial dependencies. ASTGCN (40) adds an
additional attention layer to capture the dynamic change of spatio-
temporal dependencies. Adaptive learning of adjacency matrices can                                             2.1 Preprocessing layer
also be introduced to solve problems that require predefined graphs
for adjacency matrices (35, 39, 41, 42).                                                                       2.1.1 Smoothing processing module
      However, the previous studies have never processed the time                                                    The input data received by the smoothing module are multivariate
series data from three domains and they have hardly ever been applied                                          time series data X                                                              xi                                                                                                            t={}                         ÎÎit,,NT. Due to the different
in dealing with epidemic time series data predicting problems. But                                             statistical rules of the health statistics departments in each countr y,
they provide the fundamental framework of the GNN (Graph Neural                                                some countries will postpone the epidemic data from the weekend to
Network) and GRU (Gated Recurrent Unit) methods and prove the                                                  Monday of the following week, which is reflected in the data as a line
effectiveness of the methods so that we  can reform the methods to                                             graph with a weekly cycle showing an obvious “sawtooth waveform.”
cater the requirement that introducing GRU (Gated Recurrent Unit)                                              In order to eliminate the negative impact of this problem on the neural
units into GNN (Graph Neural Network) to achieve better results in                                             network prediction, but also to a certain extent to eliminate some of
time series data prediction problems.                                                                          the noise of the input data, the neural network will be used after the
      The goal of this study is to try to introduce a GRU (Gated                                               input of a moving window average smoothing processing for a
Recurrent Unit) layer in the graph neural network to enable the                                                data preprocessing.
network to better capture and learn the relationship of each single                                                  The principle of sliding window average smoothing processing is
time node within a sequence and the correlation between                                                        shown in Equation 3, Finally, we will get the smoothed data                                   X   a er
individual time series. Specifically, after this change, the neural                                            processing the data on day                 t of the time series will be   equal to the
network is able to learn features and make predictions from                                                    average of its data on that day and the data on the                         n days before it and
multivariate time series data in the frequency, spectral, and time                                             the   n days a er it, and         (                                                                                                           )21n         + is called the window size. Considering
domains: after GFT and DFT, it is easier to perform convolution                                                the characteristics of the data in this experiment,                         n is set to 3, that is,
and graphical convolution operations on the time series in the                                                 the window size is 7.
frequency and spectral domains respectively, which in turn allows                                                                                           tn+
for more effective predictions. The introduction of GRU (Gated                                                                           x   =                             +1=-[]                              (3)
Recurrent Unit) units provides better learning in the time domain                                                                          t    21                                                  ,,n                                                                                                                                            xt                                                          nN                                           nåi
compared to linear units used in the general GNN (Graph Neural                                                                                            it                       n=-
Network) research methods.
                                                                                                               2.1.2 Graph building blocks
2 Methods                                                                                                            GNN (Graph Neural Network)-based methods need to construct
                                                                                                               a graph structure GN         E=     {},        before forecasting multivariate time
      The overall structure of the improved GNN (Graph Neural                                                  series. In this study, the number of active cases in a certain
Network) network (later referred to as GRGNN) with the introduction                                            geographical area is taken as the object of the study, and the data of
of GRU (Gated Recurrent Unit) consists of three parts: the                                                     each subregion in the geographical area is taken as the node                                 N of the
Frontiers in Public Health                                                                                03                                                                                        frontiersin.org

Liu et al.                                                                                                                                                                                     10.3389/fpubh.2024.1397260
      FIGURE 1
      The overall structure of the improved GRGNN network.
                                                                                                                                                          WxaviernormalHK=                   ()                                  (5)
graph, and the edges E of the graph denote the correlation and the
magnitude of the in uence of each node on each other. In this study,
E is represented by the weight matrix                                 W                                              NNÎ                                            ´.  e elem敮t ìQ
wi                                                                             Nj                                                                              Nij     ,,Î-[]                                      Î--[ÎÎ[]-[ in W represents the magnitude of the ïQRW=
in uence weight of the                ith node on the            jth node.  e graph structure                                                               ï
                                                                                                                                                            ïKRW=            H                                                   (6)
in this study is denoted by                 GX          W=     {},.                                                                                         í                                 T
      Part of the graph structure can be     constructed by humans for                                                                                      ï
                                                                                                                                                            ïWsoft                                                                                   QK=                                                                                                                                                       æç                                                                                              ö÷
                                                                                                                                                            ï                                                                                                                maxèdø
obser vation or through experience or knowledge (e.g., road networks in                                                                                     î
tra c forecasting, grid systems in electrical energ y forecasting). However,
in general, there is usually no worthwhile su cient                             a priori experience to
accomplish graph construction artificially. For example, in this study,                                                         where Q and              K denote the query and key hiding matrices,
when dealing with data related to epidemics, there may be  a situation                                                   respectively, and the magnitude of their values are computed by two
where the transmission pathways and characteristics of the epidemics                                                     learnable parameter matrices                     W   Q and W    K, respectively, whose initial
under study have yet to be     studied, and the existing research and                                                    values are obtained by xavier initialization of the input H (44                                    ); d is the
knowledge about them cannot support the construction of the graph. In                                                    size of the dimensions of the two matrices                             Q and      K.  e  nal output
order to cope with this situation, the correlation between multiple time                                                 adjacency weight square matrix                     W                                              NNÎ                                            ´⁷ill b攝used with th攠楮put
series is captured in the preprocessing stage through the self-attention                                                 mult楤業敮sional t業e s敲楥猠                           X                                                         NTÎ                                            ´, which forms th攠Ὦal g牡ph
mechanism with the GRU (Gated Recurrent Unit) layer before the data                                                      str畣t畲攠         GX          W=     {},.
is input into the neural network, and the correlation of each time series
is determined in a data-driven manner, which then completes the
construction of the required graph structure for the neural network (42).                                                2.2 GRGNN layer
      A specific description of the self-attention mechanism approach
for the composition layer is given below :                                                                                      The GRGNN layer consists of multiple GRGNN modules
      First of all, the multivariate time series X                                                      NNÎ                                            ´⁷ill b攝fe搠楮to stacked in a shortcut connection manner, and the data will
th攠䝒U (Gated Rec畲r敮t U湩琩⁬ay敲, which ca汣ulat敳⁴h攠桩摤敮                                                                       be captured and extracted features in the GRGNN modules from the
stat攠捯rr敳ponding to e慣h t業e no摥⁳equentially.  攠桩摤敮⁳tat敳                                                                  three dimensions of the spectral domain, the frequency domain, and
捯rr敳pondin朠t漠e慣栠t業攠no摥猠ar攠捯mpute搠sequentially⸠ en,                                                                       the time domain, respectively. The specific structure of the GRGNN
w攝use the las琠桩摤敮⁳tate to ca汣ulate the weight matr楸⁴桲o畧h the block module, as shown in Figure        2                                                                                      . The features in data will
se汦ⵡtt敮t楯n mecha湩sm.  e math敭at楣al des捲ipt楯n is as be captured and extracted in three domains of the spectral domain,
Equat楯n 4�6:                                                                                                             the frequency domain, and the time domain respectively, in the
                                     Q                                                                                   GRGNN modules. The following is a description of each part of
                                 Wx=          aviernormalH()                                            (4)              GRGNN block and its functions:
Frontiers in Public Health                                                                                         04                                                                                                frontiersin.org

Liu et al.                                                                                                                                                                                                 10.3389/fpubh.2024.1397260
       Spectral domain graph convolution is a method that has been
widely used in time series forecasting problems. The method has been
widely used in time series forecasting problems due to its excellent
results in learning potential representations of multiple time series in
the spectral domain. The key to the method is the application of the
Graph Fourier Transform (GFT) to capture the relationships between
time series in the spectral domain. Its output is also a multivariate time
series, and the GFT does not explicitly learn the relationship between
the data at each time node within a given time series. Therefore, it is
necessar y to introduce the Discrete Fourier Transform (DFT) to learn
the characterization of the input time series in the frequency domain,
for example, to capture repetitive features in periodic data.
2.2.1 Frequency domain convolution part
       The function of the frequency domain convolution part aims to
transfer each individual time series into the frequency domain
representation after processing it by DFT, and to learn its features by
1DConv layer in the frequency domain. It consists of four sub-parts
in order: discrete Fourier transform (DFT), one-dimensional
convolution (1DConv), gated linear unit (GLU), and inverse discrete
Fourier transform (IDFT), where DFT and IDFT are used to
transform the time series data between time and frequency domains,
and 1DConv and GLU are used to learn the features of the time series
in the frequency domain. The DFT processing of time sequence
usually results in a complex sequence, and the frequency domain                  r
convolution is performed on the real part (i                                 X      u) and imaginar y part
(X      u) respectively, and the processing can be       expressed by
Equation 7 as:
                     *                                                       *                                                                                                                                                                                                  *                                                           *                                                              *                                                           *æ/circumflexnosp/circumflexnospöææöæ/circumflexnospöö
                MGLUçX,uu÷                      çqqt                                                                  t                                                           tçXX÷ç÷÷
                       ç        ÷         =     ç       ç        ÷       ç        ÷÷
                       è        ø               è       è        ø       è        øø
                                   =qqs                                       qtt**                                                                                         **                                                       *()                                                                                                                                                        ()XXuu⊙(                                                                                                                                   )              *Î,,{}ri(7)
       Where qt* denotes the size of the convolution kernel for 1D
convolution,            ⁤敮ot敳⁴he H慤amard prod畣t operat楯測⁡n搠                                                        s    *
denotes the                sigmoid          activation function. ie snal result
     r                                              uæ                                                                               r                                                                                                          i                                             uöæiö
 Mx                                                                        iM                                          xç÷ç÷ is converted back to the time domain after FIGURE 2
        ç      ÷        +    ç      ÷
        è      ø             è      ø                                                                                                  The overall structure of GRGNN module.
IDFT processing to participate in the subsequent part of
the processing.
                                                                                                                                                     --                                                                                                                                           ´11
2.2.2 Spectral domain graph convolution part                                                                                     L                               ID                            WD                                           I=-                                                                                                                                                                                               ÎNN                                                        NN22      , where I   N                                                                      NNÎ                                            ´⁩s the unit
       Graph C onvolution (29) consists of three parts.                                                                          matr楸⁡n搠            D is the degree matrix with diagonal element                                     D                                                                                   Wii=      åij.
       First, Transformation of multivariate time series inputs to the                                                           Then, the eigenvalue decomposition of the Laplace matrix is performed j
spectral domain via GFT. Second, performing a graph                                                                              to obtain L                           UU   T=L, where U                                                 NNÎ                                            ´⁩s the matr楸⁯f eig敮vectors
convolution operation on the spectral domain graph structure                                                                     慮d L is the diagonal matrix of eigenvalues. Aser, the GFT, time series
using a graph convolution operator with a convolution kernel to                                                                  will beQtransformed into complex numbers, for example, three datasets
learn. Third, performing the inverse graph Fourier transform                                                                     aser DFT are shown in                      FigureQ 3       . For a detailed introduction to the
(IGFT) on the spectral domain convolution result to generate the                                                                 dataset, see section 2.4.1. Given a multivariate time series                                          X                                                         NTÎ                                            ´Ⱐ
final output.                                                                                                                    the GFT and IGFT operators and spe捩ὣ⁯perat楯ns are, r敳pectively,
       The graph Fourier transform (GFT) (22) is the basic operator for                                                                                                                                    -1æ     ö
the convolution of spectral domain graphs. It projects the input graph                                                           denote搠a猠GF                ()XU==T            XX            and GF             ç  XU÷          = X. The graph
                                                                                                                                                                                                                è      ø
into a standard orthogonal space where the basis is constructed from                                                                                                                                            g     ()L
the eigenvectors of the normalized graph Laplacian. The normalized                                                               convolution operator is realized as a function                                   Q           of the eigenvalue
graph Laplacian matrix (15) can be       computed as follows:                                                                    matrix L, where              Q is the convolution kernel parameter.
Frontiers in Public Health                                                                                                 05                                                                                                      frontiersin.org

Liu et al.                                                                                                                                                         10.3389/fpubh.2024.1397260
     FIGURE 3
     The overview plot of time series after discrete Fourier transform. (A1) The overview plot of real parts in time series for African dataset after discrete
     Fourier transform. (A2) The overview plot of image parts in time series for African dataset after discrete Fourier transform. (B1) The overview plot of real
     parts in time series for European dataset after discrete Fourier transform. (B2) The overview plot of image parts in time series for European dataset after
     discrete Fourier transform. (C1) The overview plot of real parts in time series for Hungarian dataset after discrete Fourier transform. (C2) The overview
     plot of image parts in time series for Hungarian dataset after discrete Fourier transform.
Frontiers in Public Health                                                                        06                                                                                  frontiersin.org

Liu et al.                                                                                                                                           10.3389/fpubh.2024.1397260
2.2.3 Time domain GRU (gated recurrent units)                                                 epoch to 150 and the number of layers to 7. Additionally, the ADAM
layer                                                                                         optimizer was used in the training process.
     Recurrent Neural Networks (RNN) are a type of neural networks
with an inner recurrent loop structure (23). The reformed GRGNN
with its introduction and GRGNN’s application on the epidemic field                           2.4 Dataset, baseline methods and
is an important innovation in this study. GRU (Gated Recurrent Unit)                          evaluation indicators
processes sequences by traversing the sequence elements and
generating a hidden state that contains pattern information related to                        2.4.1 Datasets
the historical data, which contains the before-and-after relationships                              In this study, the prediction effect of GRGNN was tested using
of the sequences. GRUs (Gated Recurrent Units) (23) are a type of                             the 42 European countries’ COVID-19 dataset, the 38 African
recurrent neural networks in which each loop unit adaptively captures                         countries’ COVID-19 dataset and the 20 Hungarian regions’
dependencies at different time scales. Similar to LSTM (Long Short-                           chickenpox dataset, the over view plots of the datasets are shown in
Term Memor y) units, GRUs (Gated Recurrent Units) have a gating                               Figure 4    both COVID-19 datasets in this study were collected from
unit that regulates the information within the unit, but do not have a                        publicly available data provided by the Worldometers website (45).
separate storage unit like LSTM (Long Short-Term Memor y).                                    Worldometer is run by an international team of developers,
                                                                                              researchers, and volunteers with the goal of making world statistics
                             zWtz=s()·[]hxtt-1,                                  (8)available in a thought-provoking and time relevant format to a wide
                                                                                              audience around the world Government’s communication channels
                                                                                              which makes the data from it more reliable and realistic. The 42
                             rW=  s()·[]hx,                                                   European countries’ COVID-19 dataset contains 42 time series, and
                             tr               tt-1                               (9)the length of each time series in the dataset is 776. The 38 African
                                                                                              countries’ COVID-19 dataset contains 38 time series, and the length
                                                                                              of each time series in the dataset is 776. The 20 Hungarian regions’
                         hW                     rh                                               xtt                                              tt=tanh                                                                                                                       ·                                                          1,()[]-(10)chickenpox dataset contains 20 time series, and the length of each
                                                                                              time series in the dataset is 523. Two COVID-19 datasets analyzed
                                                                                              during the current study are available in the [Worldometers]
                                                                                              repositor y.1 The daily active case count data of each country were
                          hztt=-()1                                                                                                                                                                      1·· hztt-+ht (11)collected for a total of 776                                                                  days from Februar y 15, 2020 to April 1,
                                                                                              2022, and the data were cleaned to exclude from the data that existed
     The specific mathematical description of GRU (Gated Recurrent                            for more than 20                                                            days without updating the data, and the data that
Unit) is shown in Equation 8–11             , there are only two gate units in                had a negative number of active cases or other statistical errors,
GRU (Gated Recurrent Unit), one is reset gate and the other is                                finally we classify the data that met the above requirements to obtain
update gate, and the role of reset gate is similar to that of input gate                      the continental active case dataset. The 20 Hungarian regions’
and forgetting gate in LSTM (Long Short-Term Memor y), (                                                                                  )1    -z is chickenpox dataset was chosen to collect weekly chickenpox
equivalent to the input gate, and         z is equivalent to the forgetting gate.             diagnosis data from 20 regions in Hungary for 523                                           weeks from
 e GRU (Gated Recurrent Unit) method uses fewer threshold units                               Januar y 3, 2005 to December 29, 2014. The 20 Hungarian regions’
to accomplish a similar task as the LSTM (Long Short-Term                                     chickenpox dataset are available,2 the dataset was downloaded from
Memory) method, so the GRU (Gated Recurrent Unit) method is                                   Kaggle (46), a website that focuses on providing developers and data
usually considered when there is a lack of computational power or a                           scientists with a platform to hold machine learning competitions,
desire to improve the training speed and e ciency of neural network                           host databases, and write and share code. The Hungarian chickenpox
learning.  e GRU (Gated Recurrent Unit) method uses fewer gate                                dataset, as a typical multivariate time series prediction problem
units than the LSTM (Long Short-Term Memory) method and                                       dataset was consisted by the time series collected from the Hungarian
accomplishes a similar task.                                                                  Epidemiological Info, a weekly bulletin of morbidity and mortality of
                                                                                              infectious disease in Hungar y. This dataset was tested on the Kaggle
                                                                                              platform with many time series prediction methods and data
2.3 Implementation and parameter design                                                       visualization methods.
     The GRGNN method was developed using the Python language                                 2.4.2 Baseline methods
based on Pytorch and MATLAB language, the experiments of                                            Three widely used neural network architectures; LSTM (Long
GRGNN were performed on a deep-learning server with NVIDIA                                    Short-Term Memor y), GRU (Gated Recurrent Unit), CNN-LSTM and
Quadro GV100L GPU *1, Intel Xeon Gold 6,138 CPU *1 and DDR4                                   a statistical method, were chosen as the control group in this study, the
32G RAM *8, the operation system of Ubuntu 18.04.6 LTS. The                                   statistical methods include, weighted moving average method(WMA)
baseline methods were all implemented using MATLAB language. on
clearance version.
     Hyperparameters such as input length, learning rate, batch size,
training time and number of hidden units needed to be   set in the                            1   https://www.worldometers.info/coronavirus
GRGNN. Empirically, normalization method was set to z-score, input                            2   https://www.kaggle.com/datasets/die9origephit/
length to 15, learning rate to 4.7e-4, batch size to 15 and training                          chickenpox-cases-hungary
Frontiers in Public Health                                                                07                                                                           frontiersin.org

Liu et al.                                                                                                                                                  10.3389/fpubh.2024.1397260
    FIGURE 4
    The overview plot of the datasets. (A) The overview plot of ARMSE of the 38 African countries’ COVID-19 dataset. (B) The overview plot of the 42
    European countries’ COVID-19 dataset. (C) The overview plot of 20 Hungarian regions’ chickenpox dataset.
Frontiers in Public Health                                                                    08                                                                              frontiersin.org

Liu et al.                                                                                                                                                                                                                                                               10.3389/fpubh.2024.1397260
(47), Gaussian function method (48) and polynomial functions                                                                                                            Gaussian function                         Gx(), which has been widely applied in prediction.
method (48):                                                                                                                                                            )e speci(c de(nition of Gaussian function (tting method is given in
         The following 7 baseline methods were used to compare the                                                                                                      Equation 14. In this research we applied 3-order Gaussian function to
performance with the GRGNN:                                                                                                                                             (tting each time series.
         ARIMA (15): ARIMA (Autoregressive Integrated Moving Average
                                                                                                                                                                                                                                     2                                  2                                  2
Model) is a widely applied time series forecasting method, extensively                                                                                                                                              -                                        -æxb1æxb2                      æ  xb3
                                                                                                                                                                                                                      ç                                                                             öc÷                                                                                                                                -                                        -ç                                                                                öc÷                                                                                                                              -                                        -ç                                                                               öc÷
used across various fields. This paper adopts it as a classical statistical                                                                                                             Gx()         =+                                                                                                                                                                                                                                       +ae                                                                                                                                                       ae                                                                                                                                                          ae12                                                                                                                                                                                                                         3··                                                                                                                                                                                                                           ·                                                                                                                                øè1øè2øè3ø (14)
prediction method to compare with machine learning approaches for
forecasting COVID-19 data in Africa. Its specific definition is given in
Equation 12.                                                                                                                                                                     Polynomial function fitting method (48): one of the most popular
                                                                                                                                                                        curve fitting algorithms for fitting the time series with a n-order
                           æp                         öæ                                             q             ö                                                    polynomial function, which has been widely applied in prediction.
                           ç11                                                                                                                                                                                                                                                  1-ååjqi                     i                                                                                                                                   d                                         iLL÷()                                                                                =+-XLçi                     i                                             t÷e(12)
                           ç÷ç                                                                                     ÷                                                    The specific definition of polynomial function fitting method is given
                           èi==11øè                                                                 i              ø                                                    in     Equation 15. in this research we     applied 5-order polynomial
                                                                                                                                                                        function Gx() to (tting each time series.
         Herein, L represents the lag operator, with                                                      dd                                     Z>Î0,. The main
steps of this method are as follows:                                                                                                                                                    Gx()          =+                                                                                                               ++px                                                   px                                                    px                                                  px                                                    px                                   p1                           5                                                              2                              4                                                              3                            3                                                              4                              2····                                                                                                                                                                                                                          ·++56 (15)
         The prediction will finish in 4 steps: step      1, Time series
preprocessing. The primar y purpose here is to make the input to the
ARIMA model a stationary time series. If the data series is                                                                                                                      LSTM (Long Short-Term Memor y): Long Short-Term Memor y
non-stationary and exhibits certain growth or decline trends, it is                                                                                                     networks were first introduced by Hochreiter in 1997 (22). They are a
necessary to differentiate the data. Step    2, Establishing the model                                                                                                  specific form of RNN (Recurrent Neural Network), which is a general
based on identification rules for time series models. If the partial                                                                                                    term for a series of neural networks that can process sequential data.
autocorrelation function of the stationar y series is truncated while the                                                                                                        Generally, RNNs possess three characteristics: first, they can generate
autocorrelation function is tailed, the series is suitable for an AR                                                                                                    an output at each time step, with connections between hidden units being
model; if the partial autocorrelation function is tailed while the                                                                                                      cyclic; second, they produce an output at each time step, where the output
autocorrelation function is truncated, the series is suitable for an MA                                                                                                 at a given time step is only cyclically connected to the hidden unit of the
model; if both the partial autocorrelation and autocorrelation                                                                                                          next time step; third, RNNs contain hidden units with cyclic connections
functions are tailed, the series fits an ARIMA model. Step      3,                                                                                                      and can process sequential data to produce a single prediction.
Determining the order of AR and MA. Utilize the Akaike Information                                                                                                               LSTM (Long Short-Term Memory) is such a gated RNN. The
Criterion (AIC) and Bayesian Information Criterion (BIC) to                                                                                                             ingenuity of LSTM (Long Short-Term Memor y) lies in the addition of
determine the orders p and                                          q of AR and MA, respectively. Step  4,                                                              input, forget, and output gates, allowing the self-recurrent weights to
ARIMA  tting and forecasting. Fit the ARIMA model, then use the                                                                                                         var y. Thus, the integration scale at different moments can dynamically
 tted results to forecast the test set. It’s worth mentioning that these                                                                                                change even when the model parameters are fixed, thereby avoiding
results are a er one di erentiation, and the forecasted values need to                                                                                                  problems of gradient vanishing or exploding.
be restored through inverse di erentiation.                                                                                                                                      Each LSTM (Long Short-Term Memor y) unit is composed of a
         weighted moving average method (WMA) (                                                                     47    ): the weighted                               memor y cell and three gating units: the input gate, the output gate,
moving average (WMA) method is a time series analysis technique                                                                                                         and the forget gate. Within this architecture, LSTM (Long Short-Term
that assigns di erent weights to historical obser vations based on their                                                                                                Memory) attempts to create a controlled flow of information by
relative importance. Unlike the simple moving average (SMA)                                                                                                             deciding what information to “forget” and what to “remember,”
method, which assigns equal weight to all observations, the WMA                                                                                                         thereby learning long-term dependencies.
method seeks to accentuate the impact of more recent data and reduce
the impact of older data points.  e WMA method calculates the                                                                                                                                                               zW                                  hxtz                                      tt=                                                                                                                []s()·-1, (16)
weighted average of a sequence of observations, with the most recent
values carrying the highest weightings.  e weightings assigned to                                                                                                                                                  f=+                                                                                                                      +s                                                                                                                                                                                                                                                                                                1Ux                                          Wh                                                                      b
each observation are typically determined by a prede ned set of                                                                                                                                                      tg                   tg                  tg()-                 (17)
coe cients or by subjective judgment based on the characteristics of
the data being analyzed.  e WMA method is frequently used in                                                                                                                                                    c                                                                                                                       Ux                                        Wh                                                                   b=+                                                                                                                       +tanh                                                                                                                                                                                                                                    1()
 nancial market analysis to identify trends and forecast future prices.                                                                                                                                           tc                 tc                           tc-                 (18)
 e speci c de nition of WMA is given in                                                            Equation 13                 .
                             /circumflexnosp                                                                                                                                                      /midhorizellipsis      cg   c                    i ct  tt             tt=+                                    −1(19)
                            XXt                                                                                                                                  tt                                                                                                                                                      Nt                    N+1                                                            01=+                                                                                                   ++ww                                                                                                                                                      wXX--11+ (13)
         Where               X      t+1  denotes the prediction for the time point t       +     1,                                                                                                               oto                to               to=+                                                                                                                  +s                                                                                                                                                                                                                                                                                      1()Ux                                       Wh                                                                  b- (20)
X  * stands for the observation value, and                                                w* stands for the weight of                                  X  *.
         Gaussian function ´tting method (                                                 48    ): one of the most popular                                                                                                   ho                                                                                                                                                           c=                                                                                                                                                                                    ()tanh(21)
curve ´tting algorithms for ´tting the time series with a n-order                                                                                                                                                                tt                                                                                                                                                                     t
Frontiers in Public Health                                                                                                                                      09                                                                                                                                      frontiersin.org

Liu et al.                                                                                                                                                                                                                                                                                     10.3389/fpubh.2024.1397260
          More specifically, the input gate it alongside the second gate                                                                                               ct            datasets are then organized into matrices, with each column
control the new information stored in the memor y state                                                                                    ct at a certain                            representing a di erent lagged version of the data, making it suitable
time         t.  e forget gate                             f t controls the disappearance or retention of                                                                             for sequential processing by the model.
information from time                                     t       -    1 in the storage unit, while the output gate                                                                             For the LSTM (Long Short-Term Memor y) component, it is the
ot controls which information can be outputted by the storage unit.                                                                                                                   same like the LSTM (Long Short-Term Memory) methods
Equations 16–21                            succinctly describe the operations performed by an                                                                                         we     introduced above. And for the CNN component, the data is
LSTM (Long Short-Term Memor y) unit.                                                                                                                                                  initially processed through a sequence folding layer, transforming the
          Herein,              xt represents the input at a certain moment,                                                                      W* and             U *               sequential input into a format amenable to convolutional operations.
represent weight matrices,                                        b* denotes the bias vector,                                      s is the sigmoid                                   tis step is pivotal for extracting spatial features from the lagged
function, and the operator                                         ⁲数r敳敮瑳⁥lem敮t-wise multiplicat楯渮                                                                                   inputs, which are then unfolded and  attened to preser ve the temporal
Finally, th攠桩摤敮⁳tate uni琠ht, which forms part of the memor y cell’s                                                                                                                   sequence structure, allowing the subsequent LSTM (Long Short-Term
output, is calculated as shown in                                                 Equation 21                   .                                                                     Memor y) layers to learn long-term dependencies from these extracted
          It is noteworthy that if multiple LSTM (Long Short-Term                                                                                                                     features e ectively. By meticulously mapping our datasets through
Memor y) layers are stacked together, the memor y state                                                                                  ct and hidden                                these preparator y stages, we ensure that the CNN-LSTM architecture
st ate      ht of each LSTM (Long Short-Term Memor y) layer will ser ve as                                                                                                            leverages both spatial and temporal dimensions of the data, thereby
inputs to the next LSTM (Long Short-Term Memor y) layer.                                                                                                                              enhancing the model’s forecasting accuracy.
          In this paper, the main hyperparameters for the LSTM (Long                                                                                                                            In this paper, the hyper parameters for the CNN-LSTM method
Short-Term Memory) method are set as follows: the number of                                                                                                                           are set as follows: the number of maximum training epoch is 150, the
iterations is 150, the number of hidden units is 400, the initial learning                                                                                                            batch size is 12, the lag is 8, the number of hidden units [LSTM (Long
rate is 0.001, and the optimizer used is ADAM.                                                                                                                                        Short-Term Memor y) component] is 150, the initial learning rate is
          GRU (Gated Recurrent Unit): te GRU (Gated Recurrent Unit)                                                                                                                   0.001, and the optimizer used is ADAM.
is also a type of recurrent neural network. Like LSTM (Long Short-
Term Memor y), it was developed to address issues related to long-                                                                                                                    2.4.3 Evaluation indicators
term memory and gradients in backpropagation. Compared to                                                                                                                                       Average RMSE and average MAE are used as evaluation metrics
LSTM (Long Short-Term Memor y), using GRU (Gated Recurrent                                                                                                                            to measure the magnitude of error in the prediction results:
Unit) can achieve comparable results and is easier to train,                                                                                                                                    The average RMSE is calculated by sequentially calculating the
signihcantly enhancing training e ciency. terefore, GRU (Gated                                                                                                                        RMSE for each of the N countries in the prediction result of the
Recurrent Unit) is o en preferred, especially in scenarios with                                                                                                                       sequence prediction step                                       H     .  e speci c mathematical description is
limited computational power or when there is a need to conser ve                                                                                                                      as following                   Equation 22                    :
computational resources.
          GRU (Gated Recurrent Unit) has only two gating units: a reset                                                                                                                                                      ì                                åH                                  pred          i                                                      obs        iyy,,
gate and an update gate, as shown in                                                                   Equations 8–11                           , where                xt                                                    ïRMSEi                                                                                                        i=                                                                                                                                                                                                       -=1
represents the input at a given time,                                                     W* represents a weight matrix,                                               s                                                     ï                                                       H                                                         (22)
                                                                                                                                                                                                                             í                                         N                                                                                                     i
denotes the tanh function,                                           zt is the state of the update gate, and                                                      rt is                                                      ï                                 å        =1
the reset gate. te function of the reset gate is similar to the input and                                                                                                                                                    ïRMSE                                                                                                                                                        RMSEave                                                                                i=
                                                                                                                                                                                                                             î                                               N
forget gates in LSTM (Long Short-Term Memor y), where                                                                                             1   -             zt acts
like the input gate, and                                 zt functions as the forget gate. Given that GRU
(Gated Recurrent Unit) uses fewer gating units to accomplish tasks                                                                                                                              The average MAE is calculated by sequentially calculating the
similar to those of LSTM (Long Short-Term Memor y), GRU (Gated                                                                                                                        MAE for each of the N countries in the prediction result of the
Recurrent Unit) is typically considered in situations where                                                                                                                           sequence prediction step H                                            , and then calculating the average value,
computational capacity is limited.                                                                                                                                                    which is mathematically described as following                                                                         Equation 23                   :
          In this paper, the hyper parameters for the GRU (Gated Recurrent
Unit) method are set as follows: the number of maximum training                                                                                                                                                                ì                         åH                                              pred          i                                                      obs        iyy,,
epoch is 150, the batch size is 12, the number of hidden units is 400,                                                                                                                                                         ïMAEi                                                                                i=                                                                                                                                                                                            -=1
                                                                                                                                                                                                                               ï                                                 H                                                             (23)
the initial learning rate is 0.001, and the optimizer used is ADAM.                                                                                                                                                            í                                      N                                                                                  i
          CNN-LSTM: CNN-LSTM is an advanced neural network                                                                                                                                                                     ï                              åi=1
                                                                                                                                                                                                                               ïMAE                                                                                                                                                       MAEave=
architecture that combines Convolutional Neural Networks (CNNs)                                                                                                                                                                î                                          N
and LSTMs (Long Short-Term Memor y networks) to harness the
strengths of both in processing sequential data. tis hybrid model is
particularly e ective for tasks where the input data involves both
spatial and temporal dimensions, making it popular in areas such as                                                                                                                   3 Results
video analysis, natural language processing, and time
series forecasting.                                                                                                                                                                             Predictions were made using GRGNN, LSTM (Long Short-Term
          Crucially, to adapt the time series data for the CNN-LSTM                                                                                                                   Memor y), GRU (Gated Recurrent Unit), CNN-LSTM, and ARIMA
architecture, we    employ lag features transformation. tis involves                                                                                                                  for 42 countries in Europe, 38 countries in Africa, two continents’
creating new datasets where each feature corresponds to the original                                                                                                                  COVID-19 active case datasets, and Hungar y’s 20 regions’ varicella
data shi ed by values within a specized lag range, e ectively capturing                                                                                                               datasets, respectively. The last 2                       weeks (14                                                                   days), 3                                                           weeks (21                                                                   days),
temporal dependencies across multiple time steps. tese transformed                                                                                                                    4             weeks (28                                           days), 5             weeks (35                                           days), and 6            weeks (42                                           days) data were
Frontiers in Public Health                                                                                                                                                    10                                                                                                                                                 frontiersin.org

Liu et al.                                                                                                                                     10.3389/fpubh.2024.1397260
taken as the test set in the prediction, and after dividing the test set, all                   As can be     seen from Table        2, the comparison of the overall
the data prior to the test set data were divided into the training set and                 prediction results when extending the prediction step to 3                       weeks
validation set in the ratio of 10:1.                                                       (21                                 days) is not much different from that of the prediction step of
     The prediction results of each method for each dataset at different                   2         weeks. The GRGNN method still achieves the best results in the
step sizes are shown in Tables 1–       5.                                                 prediction of both the African and Hungarian datasets, and is slightly
     As can be   seen from      Table   1, with a prediction step of 2                          weeks less accurate in the prediction of the European dataset than the
(14                                            days), GRGNN achieves optimal results for both the African and CNN-LSTM and the ARIMA methods. The prediction accuracy of the
Hungarian datasets, and slightly underperforms the CNN-LSTM                                LSTM (Long Short-Term Memory) method and the GRU (Gated
method and the ARIMA method for the European dataset. The LSTM                             Recurrent Unit) method is the worst two of the eight methods in the
(Long Short-Term Memor y) method and the GRU (Gated Recurrent                              African and European datasets. The prediction errors of LSTM (Long
Unit) method underperform in all datasets. The CNN-LSTM method                             Short-Term Memor y) and GRU (Gated Recurrent Unit) methods in
performs best in the prediction of the European dataset, and                               the African and European datasets are the worst two out of the eight
underperforms GRGNN and ARIMA in the African dataset, and                                  methods. The CNN-LSTM method still performs the best in the
performs worse in the Hungarian dataset. The ARIMA method has                              prediction of the European dataset. The ARIMA method does not
the best prediction accuracy of the eight methods. The CNN-LSTM                            achieve the optimal prediction accuracy but outperforms LSTM (Long
method performs best in the prediction of the European dataset, while                      Short-Term Memor y) and GRU (Gated Recurrent Unit) in the African
it does not perform as well as GRGNN and ARIMA on the African                              and European datasets, and outperforms CNN-LSTM in the
dataset, and performs even worse on the Hungarian dataset. The                             Hungarian dataset in terms of prediction error. The WMA method
prediction accuracy of the ARIMA method is in the middle of the                            still yields slightly inferior results compared to ARIMA and marginally
range of the eight methods. The WMA method can achieve predictions                         better outcomes than the LSTM (Long Short-Term Memor y) method.
with an accuracy approximately equal to that of ARIMA. Conversely,                         However, the Gaussian function method and the polynomial function
the Gaussian function method and the polynomial function method                            method continue to exhibit the poorest two results.
produce predictions significantly deviating from the real data,                                 As can be   seen from Table   3      , with a prediction step of 4                         weeks
obtaining the lowest accuracies among all eight methods across all                         (28                    days), GRGNN still maintains the optimal prediction in the
three datasets.                                                                            prediction of the African and Hungarian datasets, and the prediction
TABLE 1        Prediction results for each prediction method for each dataset for 2     weeks (14     days).
                                      African dataset                                  European dataset                                   Hungarian dataset
                                ARMSE                     AMAE                    ARMSE                      AMAE                     ARMSE                     AMAE
 GRGNN                            683.27                   621.38                  54568.57                  49345.78                    28.82                    23.64
 LSTM                            1288.20                  1071.58                  78093.59                  64940.05                    29.69                    24.57
 CNN-LSTM                         812.45                   790.14                  38421.52                  31634.68                    32.85                    26.29
 G RU                            1115.73                   907.52                  56406.04                  47197.14                    32.21                    27.66
 ARIMA                            783.04                   657.50                  40086.60                  42310.69                    29.61                    23.83
 Poly                            4620.15                  4480.89                 301141.17                 298245.73                    44.11                    36.52
 Gauss                           2289.51                  2214.87                 109168.62                 103422.19                    41.55                    34.48
 WMA                              820.68                   691.10                  70424.89                  62469.22                    35.13                    29.24
TABLE 2        Prediction results for each prediction method for each dataset for 3     weeks (21     days).
                                      African dataset                                  European dataset                                   Hungarian dataset
                                ARMSE                     AMAE                    ARMSE                      AMAE                     ARMSE                     AMAE
 GRGNN                            836.26                   770.61                  75623.18                  83044.94                    31.47                    28.47
 LSTM                            1375.33                  1116.70                 113619.62                 135365.11                    33.62                    28.75
 CNN-LSTM                         915.06                   892.35                  48978.62                  55363.46                    35.65                    31.46
 G RU                            1608.06                  1260.72                 115653.55                 144957.77                    34.41                    28.90
 ARIMA                            997.03                   848.51                  68989.77                  82938.58                    35.22                    29.77
 Poly                            5428.18                  5195.21                 409270.15                 401718.39                    29.61                    28.93
 Gauss                           2641.67                  2531.15                 188754.28                 181754.93                    36.31                    28.60
 WMA                             1007.55                   831.20                 119667.10                 104058.45                    29.70                    24.21
Frontiers in Public Health                                                             11                                                                       frontiersin.org

Liu et al.                                                                                                                                              10.3389/fpubh.2024.1397260
results in the European dataset are only slightly inferior to those of the                           As can be   seen from         Table     5, when the prediction step size is
CNN-LSTM method. The prediction errors of the LSTM (Long Short-                                 6              weeks (42                                                                                      days), the average of the prediction results of GRGNN in
Term Memor y) method and the GRU (Gated Recurrent Unit) method                                  the prediction of the European dataset exceeds that of the CNN-LSTM
are still poor in the African and European datasets. The CNN-LSTM                               (Long Short-Term Memor y) method to become the smallest among
method still performs optimally in the prediction of the European                               the results of each prediction method, and realizes the prediction
dataset, but poorly in the European dataset. The ARIMA method is                                accuracy of the prediction of each data to be  the highest among all
still in the mid-range of the eight prediction mid-range levels. Still                          eight prediction methods. The prediction error of WMA only slightly
performs the best in prediction, but has poor prediction in the                                 exceeds that of LSTM (Long Short-Term Memor y) and GRU (Gated
Hungarian dataset. The prediction accuracy of the ARIMA method is                               Recurrent Unit), placing its results ahead of both LSTM (Long Short-
still in the middle of the range of the 5 prediction mid-range. The                             Term Memor y) and GRU (Gated Recurrent Unit). However, it falls
performance of the WMA method is slightly inferior to the ARIMA                                 short compared to GRGNN, CNN-LSTM, and ARIMA methods. The
method but slightly superior to the GRU (Gated Recurrent Unit) and                              polynomial method and Gaussian function method persist as the least
LSTM (Long Short-Term Memor y) methods. However, the Gaussian                                   effective, exhibiting the highest ARMSE and AMAE values.
method and the polynomial method remain the least effective,                                         The average indictors of the prediction results of each method in
exhibiting significant errors in their prediction results.                                      each dataset are plotted at different step sizes, as shown in Figure 5.
     As can be seen from        Table 4  , when the prediction step size is set to                   To enhance the clarity and simplicity of conveying the prediction
5                  weeks (35                                                                                                      days), the ranking of the prediction results of each method results, we have selected 5 time series from each dataset, focusing on
is not much different from that of the case with a step size of 4                      weeks,   a prediction step set to 6                                               weeks (42                                                                                                     days) for visualization. Specifically,
and it is worth noting that: the main change occurs in the prediction                           we   depict the time series data of 5 countries from the 38 African
results for European data, and the average index of GRGNN exceeds                               countries’ COVID-19 dataset in Figure    6              , and the time series of 5
that of CNN-LSTM as the smallest among the prediction methods. The                              countries from the 42 European countries’ COVID-19 dataset in
performance of the WMA method deteriorates rapidly, reaching a                                  Figure        7, and illustrate the time series of 5 regions from the 20
point where it only outperforms two other methods. The Gaussian                                 Hungarian regions’ chickenpox dataset in Figure   8. Through these
function method and the polynomial function method still remain the                             figures, it becomes evident that GRGNN generally captures and
poorest performers, with their accuracy indices worsening even                                  mirrors the trends obser ved in the majority of the time series from the
further as the prediction steps increase.                                                       original real-world data.
TABLE 3        Prediction results for each prediction method for each dataset for 4     weeks (28     days).
                                        African dataset                                     European dataset                                      Hungarian dataset
                                  ARMSE                      AMAE                      ARMSE                       AMAE                       ARMSE                       AMAE
 GRGNN                              748.42                    858.05                   111743.17                   123580.61                     27.48                     21.75
 LSTM                              2296.05                    2775.97                  125159.58                   151888.46                     28.02                     22.36
 CNN-LSTM                           882.52                    921.98                    88773.80                   97859.95                      29.10                     22.21
 G RU                              1718.08                    2188.22                  188863.87                   161955.56                     28.88                     23.04
 ARIMA                              921.32                    1082.60                  136034.82                   112387.49                     27.55                     20.98
 Poly                              6628.94                    6351.37                  534460.56                   520915.42                     35.01                     26.59
 Gauss                             3254.86                    3078.51                  214257.05                   194298.74                     33.07                     25.51
 WMA                               1437.04                    1228.77                  152546.44                   132098.87                     28.13                     22.02
TABLE 4        Prediction results for each prediction method for each dataset for 5     weeks (35     days).
                                        African dataset                                     European dataset                                      Hungarian dataset
                                  ARMSE                      AMAE                      ARMSE                       AMAE                       ARMSE                       AMAE
 GRGNN                              820.70                    1004.61                  120230.14                   127749.64                     27.27                     21.45
 LSTM                              2507.72                    3072.21                  255698.55                   219314.79                     27.40                     21.55
 CNN-LSTM                          1536.70                    1593.65                  128824.08                   111020.42                     28.14                     21.91
 G RU                              2234.65                    2667.76                  250900.80                   213550.93                     28.26                     22.00
 ARIMA                             1537.67                    1731.40                  150250.91                   125333.12                     29.96                     22.37
 Poly                              8436.73                    8093.56                  652543.68                   625536.10                     34.23                     26.48
 Gauss                             4526.36                    4227.41                  212263.48                   193738.61                     32.04                     25.00
 WMA                               2525.29                    2301.08                  238699.26                   209711.46                     30.51                     22.38
Frontiers in Public Health                                                                  12                                                                            frontiersin.org

Liu et al.                                                                                                                                      10.3389/fpubh.2024.1397260
TABLE 5        Prediction results for each prediction method for each dataset for 6     weeks (42     days).
                                      African dataset                                   European dataset                                  Hungarian dataset
                                ARMSE                     AMAE                     ARMSE                      AMAE                     ARMSE                     AMAE
 GRGNN                           1545.62                   1763.28                 124665.83                 133453.75                   25.51                    19.17
 LSTM                            3418.20                   4090.81                 308407.33                 367230.08                   27.58                    22.07
 CNN-LSTM                        1657.79                   1810.84                 124829.94                 153435.48                   26.18                    20.08
 G RU                            4648.19                   5709.85                 232157.67                 269820.41                   25.72                    20.48
 ARIMA                           2673.66                   3035.86                 188922.10                 229932.70                   27.45                    20.27
 Poly                           10843.29                  10305.87                 739501.24                 697691.22                   33.02                    24.93
 Gauss                           4735.57                   4382.75                 251950.31                 218247.07                   30.33                    23.09
 WMA                             3435.84                   3093.05                 426603.38                 363924.52                   27.98                    20.14
                                                                                           collected, the actual predictions obtained at the same prediction step
4 Discussion                                                                               size are less than other two dataset. Therefore, as shown in
                                                                                           Figures      5E,F, when the prediction step length is extended from
     Obser ving Tables 1–       5, it can be    found that for the prediction              2           weeks to 3                                    weeks, each prediction method shows an increase in
results of the data of the 38 African countries’ COVID-19 dataset and                      prediction error, whereas the error of each prediction method except
the 20 Hungarian regions’ chickenpox dataset, GRGNN is able to                             ARIMA method shows a decreasing trend when the step length is
achieve better prediction results compared with other prediction                           extended from 4                                                          weeks to 6                                              weeks. Meanwhile, GRGNN was able to
methods at different prediction steps, and the average RMSE and                            achieve better results than the other seven comparison methods in
average MAE of its prediction results are the smallest among the                           both average RMSE and average MAE. This indicates that GRGNN
prediction methods at different steps, which indicates that GRGNN is                       and the neural network prediction methods in the baseline methods
able to capture and learn the features in the data better than the three                   can realize the capture of the overall trend characteristics of the data,
neural network methods and statistical methods in the baseline                             which in turn shows that the prediction accuracy will be improved
methods, and make accurate predictions.                                                    when the data prediction step length is extended to a certain length,
     Obser ving Figures      6  , 8, it becomes apparent that for African                  and compared with the seven comparative methods, GRGNN
dataset and Hungarian dataset, the prediction results of GRGNN                             achieves more accurate prediction results, which indicates that
consistently align with the developmental trend of the original time                       GRGNN is more adequate than the other seven methods for the
series, albeit with var ying degrees of error. This obser vation suggests                  capture and learning of the overall trend characteristics of the data.
that GRGNN, to a certain extent, can predict the developmental                             This indicates that GRGNN is more adequate than the other seven
trends within the datasets.                                                                methods for capturing and learning the general trend features of
     The prediction errors at different step lengths are compared with                     the data.
the step lengths on each dataset, as shown in Figure     5              and it can              Finally, the GRGNN do not always make the most accurate
be   found that the prediction errors for the African data generally                       prediction, as can be    seen from Figures      5C        ,D, for the prediction
increase with the extension of the prediction step lengths, and the                        experiments of 42 European countries, the errors of each prediction
errors of the GRGNN method increase relatively less with the                               method are much larger than the errors of the prediction results for
extension of the prediction step lengths compared with the others,                         the African data, and the indicators of each prediction result under
which indicates that the GRGNN compared with the three neural                              the same hyper-parameters mostly reaches 10,000 counts or even
network in the baseline methods and statistical methods to capture                         100,000 counts, in which case the CNN-LSTM method has the best
and learn more adequately the relationships and features among the                         prediction results in the experiments with the prediction step lengths
temporal nodes of the time series. This also indicates that GRGNN                          of 14, 21, and 28                                                                     days, and its indicators are the smallest values among
learns the data in three dimensions: time domain, frequency domain                         the eight prediction methods, but these two metrics of CNN-LSTM
and spectral domain, compared to the seven comparative forecasting                         become larger with the increase of the prediction step. When the
methods that only learn and capture the data in the time domain,                           prediction step is extended to 35                            days, the average of CNN-LSTM is
which proves that GRGNN can capture more features in the data,                             still the smallest among the eight methods, but the mean becomes
better grasp the overall trend of the data, and realize more accurate                      sub-optimal, and the optimal value is obtained from the prediction
medium- and long-term forecasting results for the two datasets,                            results of GRGNN. When the prediction step size is increased to
namely, the data of the 38 countries in Africa and the data of the 20                      42                           days, the prediction result of GRGNN becomes optimal in both
regions in Hungar y. The results demonstrate that this allows GRGNN                        indicators. The prediction results of each prediction method in the
to explore more features in the data, better grasp the general trend of                    experiment are not satisfactor y in the European dataset, which may
the data, and thus achieve more accurate medium-term and long-term                         be   caused by the inadequacy of the type of data collected and the
predictions for the 38 African countries’ COVID-19 dataset and the                         insufficient amount of data collected for this phenomenon. Data
20 Hungarian regions’ chickenpox dataset.                                                  inapplicability is an insurmountable problem for data-driven methods,
     For the 20 Hungarian regions’ chickenpox dataset, it should                           and if the applicability of the prediction methods to the data cannot
be   separately stated that since the data in this dataset are weekly                      be   assessed, this will greatly limit the application prospects of the
Frontiers in Public Health                                                             13                                                                        frontiersin.org

Liu et al.                                                                                                                                                                             10.3389/fpubh.2024.1397260
     FIGURE 5
     The overview plot of evaluation indicator of datasets (A) the overview plot of ARMSE of the 38 African countries’ COVID-19 dataset. (B) The overview
     plot of AMAE of the 38 African countries’ COVID-19 dataset. (C) the overview plot of ARMSE of the 42 European countries’ COVID-19 dataset. (D) The
     overview plot of AMAE of the 42 European countries’ COVID-19 dataset. (E) the overview plot of ARMSE of the20 Hungarian regions’ Chickenpox
     dataset. (F) The overview plot of AMAE of the 20 Hungarian regions’ Chickenpox dataset.
prediction methods. Therefore, there is a need to discuss the                                                      between the time series marked by the x-axis and y-axis the lighter the
applicability of GRGNN to different data:                                                                          color of the block is the related closer the time series are. it can
      Plotting the heatmap of the weight matrix (W) for each dataset in                                            be observed that the accuracy of GRGNN is linked to the correlation
Figure       9, where the blocks in the plot represent the correlation                                             among time series in the datasets. In cases such as the African and
Frontiers in Public Health                                                                                    14                                                                                            frontiersin.org

Liu et al.                                                                                                                                                                    10.3389/fpubh.2024.1397260
     FIGURE 6
     The plots of original data and prediction result for countries from the 38 African countries’ COVID-19 dataset of GRGNN. (A) The plot of original data
     and prediction result for the total cases of the 38 African countries’ COVID-19 dataset of GRGNN. (B) The plot of original data and prediction result for
     Country1 from the 38 African countries’ COVID-19 dataset of GRGNN. (C) The plot of original data and prediction result for County2 from the 38
     African countries’ COVID-19 dataset of GRGNN. (D) The plot of original data and prediction result for Country3 from the 38 African countries’
     COVID-19 dataset of GRGNN. (E) The plot of original data and prediction result for Country4 from the 38 African countries’ COVID-19 dataset of
     GRGNN. (F) The plot of original data and prediction result for Country5 from the 38 African countries’ COVID-19 dataset of GRGNN.
Hungarian datasets in this research, where the correlation between                                                  We find that for the weight matrix W                    obtained a er preprocessing
time series is relatively close, GRGNN exhibits accurate predictions                                          of the dataset, the average of the sum of the weights of each node over
and the ability to forecast the developmental trend of the time series.                                       the other nodes is calculated, as shown in                     Table 6   , and it can be found
However, when facing datasets like the European dataset in this                                               that when the average value tends to 1 then the dataset yields better
research, where the correlation among time series is less pronounced,                                         prediction results by GRGNN.
GRGNN struggles to achieve a more accurate prediction compared to                                                    erefore, we hypothesize that if the average value of the sum of the
other neural network methods.                                                                                 weights of each node in the weight matrix over the other nodes converges
Frontiers in Public Health                                                                               15                                                                                       frontiersin.org

Liu et al.                                                                                                                                        10.3389/fpubh.2024.1397260
    FIGURE 7
    The plots of original data and prediction result for countries from the 42 European countries’ COVID-19 dataset of GRGNN. (A) The plot of original data
    and prediction result for the total cases of the 42 European countries’ COVID-19 dataset of GRGNN. (B) The plot of original data and prediction result
    for Country1 from the 42 European countries’ COVID-19 dataset of GRGNN. (C) The plot of original data and prediction result for County2 from the 42
    European countries’ COVID-19 dataset of GRGNN. (D) The plot of original data and prediction result for Country3 from the 42 European countries’
    COVID-19 dataset of GRGNN. (E) The plot of original data and prediction result for Country4 from the 42 European countries’ COVID-19 dataset of
    GRGNN. (F) The plot of original data and prediction result for Country5 from the 42 European countries’ COVID-19 dataset of GRGNN.
to 1, then the dataset will yield better prediction results by GRGNN. As                    dimensions in the time, spectral, and frequency domains by
a matter of fact, there are some researches to construct the graph by                       introducing a GRU (Gated Recurrent Unit) layer in the GNN (Graph
SoftMax and other methods to make the average value of the sum of the                       Neural Network) network. This gives the following advantages to the
weights of each node in the weight matrix of each node to other nodes                       neural network used in this study : Firstly, the multiple-input multiple-
converge to 1 (40), but this hypothesis is only based on the obser vation                   output temporal prediction of multiple time series variables is more
of the phenomenon shown in the experimental results, and the                                efficient compared to the single-input single-output prediction
mathematical proofs and the verification of the actual additional                           method of a single time series variable; Secondly, due to the
experiments are still need to be further supplemented.                                      introduction of the GRU (Gated Recurrent Unit) layer, it yields a more
     This paper is significantly innovative: the main focus of this study                   accurate prediction in terms of prediction accuracy ; and Thirdly, as a
is to realize the ability of the network to analyze datasets in multiple                    data-driven method, it does not require human a priori knowledge as
Frontiers in Public Health                                                              16                                                                         frontiersin.org

Liu et al.                                                                                                                                     10.3389/fpubh.2024.1397260
    FIGURE 8
    The plots of original data and prediction result for regions from the 20 Hungarian regions’ Chickenpox dataset of GRGNN. (A) The plot of original data
    and prediction result for the total cases of the 20 Hungarian regions’ Chickenpox dataset of GRGNN. (B) The plot of original data and prediction result
    for Region1 from the 20 Hungarian regions’ Chickenpox dataset of GRGNN. (C) The plot of original data and prediction result for Region2 from the 20
    Hungarian regions’ Chickenpox dataset of GRGNN. (D) The plot of original data and prediction result for Region3 from the 20 Hungarian regions’
    Chickenpox dataset of GRGNN. (E) The plot of original data and prediction result for Region4 from the 20 Hungarian regions’ Chickenpox dataset of
    GRGNN. (F) The plot of original data and prediction result for Region5 from the 20 Hungarian regions’ Chickenpox dataset of GRGNN.
a basis, which makes it easy to migrate the application to the other                      problem. Compared with classical prediction methods, graph
data processing.                                                                          neural networks, as an multiple-input-multiple-output method, can
                                                                                          quickly and easily construct graphs for multiple time series and
                                                                                          realize effective prediction in a data-driven manner. In terms of
5 Conclusion                                                                              prediction accuracy, when the predicted multivariate correlation
                                                                                          reaches a certain level (specifically, the phenomenon obser ved in
     In this paper, gated recurrent units are attempted to                                this study is that the closer the average of the sum of the connection
be  introduced into graph neural network, enabling graph neural                           weights of each node to the other nodes tends to be 1, the better the
networks to capture and learn features from data in three                                 prediction results obtained from the GRGNN for the dataset), the
dimensions, namely, null, frequency, and time domains, which is                           graph neural network with the introduction of gated recurrent units
utilized to produce notable results in the epidemic data prediction                       can achieve more accurate predictions in medium-term or long-
problem, which is a typical multivariate time series prediction                           term forecasting.
Frontiers in Public Health                                                             17                                                                       frontiersin.org

Liu et al.                                                                                                                                 10.3389/fpubh.2024.1397260
    FIGURE 9
    The heat maps of the weight matrices of datasets (A) The heat map of the weight matric of the 38 African countries’ COVID-19 dataset. (B) The heat
    map of the weight matric of the 42 European countries’ COVID-19 dataset. (C) The heat map of the weight matric of the 20 Hungarian regions’
    Chickenpox dataset.
TABLE 6        The average node sum weights of each dataset.                            – original draft. NF: Writing – original draft, Writing – review &
                     African            European               Hungarian                editing. X-pD: Writing – original draft, Writing – review & editing.
                     dataset              dataset                dataset
 average node           1.10                 0.78                    0.96
 sum weights                                                                            Funding
                                                                                             The author(s) declare that financial support was received for the
                                                                                        research, authorship, and/or publication of this article. This work was
Data availability statement                                                             supported by National Natural Science Foundation of China
                                                                                        (82341035), the Grant (2019SKLID603) from the State Key Laborator y
    The original contributions presented in the study are included in                   for Infectious Disease Prevention and Control (Current name is
the article/supplementar y material, further inquiries can be directed                  National Key-Laborator y of Intelligent Tracking and Forecasting for
to the corresponding authors.                                                           Infectious Disease), China CDC.
Author contributions                                                                    Acknowledgments
    X-dL: Writing – original draft, Writing – review & editing. B-hH:                        The authors gratefully acknowledge Tao Hong, Mingkuan Feng, Yi
Writing – original draft, Writing – review & editing. Z-jX: Writing                     Yang, for their assistance with data collection and inspiration of the idea
Frontiers in Public Health                                                          18                                                                     frontiersin.org

Liu et al.                                                                                                                                                                   10.3389/fpubh.2024.1397260
in this study. And sincerely acknowledge Olasehinde Toba Stephen, for                                        Publisher’s note
helping improve the manuscript from language perspective.
                                                                                                                   All claims expressed in this article are solely those of the
Conflict of interest                                                                                         authors and do not necessarily represent those of their affiliated
                                                                                                             organizations, or those of the publisher, the editors and the
      The authors declare that the research was conducted in the                                             reviewers. Any product that may be evaluated in this article, or
absence of any commercial or financial relationships that could                                              claim that may be made by its manufacturer, is not guaranteed or
be construed as a potential conflict of interest.                                                            endorsed by the publisher.
References
  1.  Li Y, Yu R, Shahabi C, Liu Y. Diffusion convolutional recurrent neural network:                        Proces Syst. Berlin, Heidelberg: Springer. (2015):28:802–810. doi:
data-driven traffic forecasting. arXiv preprint arXiv. (2017) 1707:01926. doi: 10.48550/                     10.5555/2969239.2969329
arXiv.1707.01926                                                                                                21. Qin Y, Song D, Chen H, Cheng W, Jiang G, Cottrell G. A dual-stage attention-
  2. Yu B, Yin H, Zhu Z. Spatio-temporal graph convolutional networks: A deep                                based recurrent neural network for time series prediction. arXiv preprint arXiv. (2017)
learning framework for traffic forecasting. arXiv preprint arXiv. (2017) 1709:04875. doi:                    1704:02971. doi: 10.48550/arXiv.1704.02971
10.48550/arXiv.1709.04875                                                                                       22.                                Graves A. Long short-term memory. Supervised sequence labelling with recurrent
  3.                            Yang S, Zhang Z, Zhou J, Wang Y, Sun W, Zhong X, et al. Financial risk analysis for neural networks. Berlin, Heidelberg: Springer (2012).
SMEs with graph-based supply chain mining. In Proceedings of the Twenty-Ninth                                   23. Cho K, Van Merriënboer B, Gulcehre C, Bahdanau D, Bougares F, Schwenk H,
International Conference on International Joint Conferences on Artificial Intelligence                       et al. Learning phrase representations using RNN encoder-decoder for statistical
(2021) (pp. 4661–4667).                                                                                      machine translation. arXiv preprint arXiv. (2014) 1406:1078. doi: 10.48550/
  4.        Khodayar M, Wang J. Spatio-temporal graph deep neural network for short-term                     arXiv.1406.1078
wind speed forecasting. IEEE Transactions on Sustain Energy. (2018) 10:670–81. doi:                             24. Dauphin YN, Fan A, Auli M, Grangier D. Language modeling with gated
10.1109/TSTE.2018.2844102                                                                                    convolutional networks. In International conference on machine learning. (2017)
  5.    Wu Q, Zheng H, Guo X, Liu G. Promoting wind energy for sustainable development                       (pp. 933–941). PMLR.
by precise wind speed prediction based on graph neural networks. Renew Energy. (2022)                           25. Lou HR, Wang X, Gao Y, Zeng Q. Comparison of ARIMA model, DNN model
199:977–92. doi:    10.1016/j.renene.2022.09.036                                                             and LSTM model in predicting disease burden of occupational pneumoconiosis in
  6.  Wang J, Zhang S, Xiao Y, Song R. A review on graph neural network methods in                           Tianjin, China. BMC Public Health. (2022) 22:2167. doi:                                      10.1186/
financial applications. arXiv preprint arXiv. (2021) 2111:15367. doi: 10.48550/                              s12889-022-14642-3
arXiv.2111.15367                                                                                                26. Yu HF, Rao N, Dhillon IS. Temporal regularized matrix factorization for high-
  7.                            Chen W, Chen L, Xie Y, Cao W, Gao Y, Feng X. Multi-range attentive bicomponent dimensional time series prediction. Adv Neural Inf Proces Syst. (2016) 29:847–855. doi:
graph convolutional network for traffic forecasting. In Proceedings of the AAAI                              10.5555/3157096.3157191
conference on artificial intelligence (2020) (Vol. 34, pp. 3529–3536).                                          27. Zhang L, Aggarwal C, Qi GJ. Stock price prediction via discovering multi-
  8.                         Chen X, Liu S, Zhao J, Wu H, Xian J, Montewka J. Autonomous port management     frequency trading patterns. In Proceedings of the 23rd ACM SIGKDD international
based AGV path planning and optimization via an ensemble reinforcement learning                              conference on knowledge discovery and data mining (2017) (pp. 2141–2149).
framework.       Ocean Coast Manag. (2024) 251:107087. doi: 10.1016/j.                                          28. Ren H, Xu B, Wang Y, Yi C, Huang C, Kou X, et al. Time-series anomaly detection
ocecoaman.2024.107087                                                                                        service at microsoft. In Proceedings of the 25th ACM SIGKDD international conference
  9.  Pinto R, Valentim R, da Silva LF, de Souza GF, de Moura Santos TG, de Oliveira                         on knowledge discovery & data mining (2019) (pp. 3009–3017).
CA, et al. Use of interrupted time series analysis in understanding the course of the                           29. Kipf TN, Welling M. Semi-supervised classification with graph convolutional
congenital syphilis epidemic in Brazil. Lancet Regional Health–Americas                      . (2022)        networks. arXiv preprint arXiv. (2016) 1609:02907. doi: 10.1145/3292500.3330680
7:100163. doi: 10.1016/j.lana.2021.100163                                                                       30. Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep
  10.                                        Cori A, Ferguson NM, Fraser C, Cauchemez S. A new framework and software to convolutional neural networks. Communications of the ACM. (2017) 60:84–90. doi:
estimate time-varying reproduction numbers during epidemics. Am J Epidemiol. (2013)                          10.1145/3065386
178:1505–12. doi:    10.1093/aje/kwt133                                                                         31. Yang D, Li S, Peng Z, Wang P, Wang J, Yang H. MF-CNN: traffic flow prediction
  11.                                      Du Z, Xu X, Wu Y, Wang L, Cowling BJ, Meyers LA. Serial interval of COVID-19 using convolutional neural network and multi-features fusion. IEICE Trans Inf Syst.
among publicly reported confirmed cases. Emerg Infect Dis. (2020) 26:1341–3. doi:                            (2019) E102.D:1526–36. doi:        10.1587/transinf.2018EDP7330
10.3201/eid2606.200357                                                                                          32. Yu H, Wu Z, Wang S, Wang Y, Ma X. Spatiotemporal recurrent convolutional
  12. Ioannidis JP, Cripps S, Tanner MA. Forecasting for COVID-19 has failed. Int J                          networks for traffic prediction in transportation networks. Sensors. (2017) 17:1501. doi:
Forecast. (2022) 38:423–38. doi:      10.1016/j.ijforecast.2020.08.004                                       10.3390/s17071501
  13. Dairi A, Harrou F, Zeroual A, Hittawe MM, Sun Y. Comparative study of machine                             33. Lv Z, Xu J, Zheng K, Yin H, Zhao P, Zhou X. Lc-rnn: a deep learning model for
learning methods for COVID-19 transmission forecasting. J Biomed Inform. (2021)                              traffic speed prediction. IJCAI. (2018) 2018:27. doi: 10.24963/ijcai.2018/482
118:103791. doi: 10.1016/j.jbi.2021.103791                                                                      34. Ma X, Dai Z, He Z, Ma J, Wang Y, Wang Y. Learning traffic as images: a deep
  14. Sanchez-Gendriz I, de Souza GF, de Andrade IG, Neto AD, de Medeiros TA,                                convolutional neural network for large-scale transportation network speed prediction.
Barros DM, et al. Data-driven computational intelligence applied to dengue outbreak                          Sensors. (2017) 17:818. doi: 10.3390/s17040818
forecasting: a case study at the scale of the city of Natal, RN-Brazil. Sci Rep. (2022)                         35.   Wu Z, Pan S, Long G, Jiang J, Zhang C. Graph wavenet for deep spatial-temporal graph
12:6550. doi: 10.1038/s41598-022-10512-5                                                                     modeling. arXiv preprint arXiv. (2019) 1906:00121. doi: 10.48550/arXiv.1906.00121
  15. Zhang GP. Time series forecasting using a hybrid ARIMA and neural network                                 36. Huang L, Mao F, Zhang K, Li Z. Spatial-temporal convolutional transformer
model. Neurocomputing. (2003) 50:159–75. doi:            10.1016/S0925-2312(01)00702-0                       network for multivariate time series forecasting. Sensors. (2022) 22:841. doi: 10.3390/
  16. Zivot E, Wang J. Vector autoregressive models for multivariate time series. Modeling                   s22030841
financial time series with S-PLUS®. New York, NY: Springer (2006). doi:                                         37. Luo D, Chen W, Fang J, Liu J, Yang J, Zhang K. GRU-AGCN model for the content
10.1007/978-0-387-21763-5_11                                                                                 prediction of gases in power transformer oil. Front Energy Res. (2023) 11:1135330. doi:
  17. Yang H, Jiang Z, Lu H. A hybrid wind speed forecasting system based on a                               10.3389/fenrg.2023.1135330
‘decomposition and ensemble’ strategy and fuzzy time series. Energies. (2017) 10:1422.                          38. Zhang J, Zheng Y, Qi D, Li R, Yi X, Li T. Predicting citywide crowd flows using
doi: 10.3390/en10091422                                                                                      deep spatio-temporal residual networks. Ar tif  Intell. (2018) 259:147–66. doi:             10.1016/j.
  18. Bai S, Kolter JZ, Koltun V. An empirical evaluation of generic convolutional and                       artint.2018.03.002
recurrent networks for sequence modeling. arXiv preprint arXiv. (2018) 1803:01271. doi:                         39.                                        Song C, Lin Y, Guo S, Wan H. Spatial-temporal synchronous graph convolutional
10.48550/arXiv.1803.01271                                                                                    networks: a new framework for spatial-temporal network data forecasting. In Proceedings
  19.                                   Yu R, Zheng S, Anandkumar A, Yue Y. Long-term forecasting using higher order of the AAAI conference on artificial intelligence (2020) (Vol. 34, pp. 914–921).
tensor RNNs. arXiv preprint arXiv. (2017) 1711:00073. doi: 10.48550/arXiv.1711.00073                            40. Guo S, Lin Y, Feng N, Song C, Wan H. Attention based spatial-temporal graph
  20.                                   Shi X, Chen Z, Wang H, Yeung DY, Wong WK, Woo WC. Convolutional LSTM convolutional networks for traffic flow forecasting. In Proceedings of the AAAI conference
network: a machine learning approach for precipitation nowcasting. Adv Neural Inf                            on artificial intelligence (2019) (Vol. 33, pp. 922–929).
Frontiers in Public Health                                                                              19                                                                                       frontiersin.org

Liu et al.                                                                                                                                                                           10.3389/fpubh.2024.1397260
  41. Wu Z, Pan S, Long G, Jiang J, Chang X, Zhang C. Connecting the dots: multivariate                              45. COVID  - Coronavirus Statistics. Worldometer. (2023). Available from:                       https://
time series forecasting with graph neural networks. In Proceedings of the 26th ACM                                www.worldometers.info/coronavirus/
SIGKDD international conference on knowledge discovery & data mining (2020)                                          46.                                Kaggle: Your Machine Learning and Data Science Community. (2023). Available
(pp. 753–763).
  42.                                       Cao D, Wang Y, Duan J, Zhang C, Zhu X, Huang C, et al. Spectral temporal graph from: https://www.kaggle.com/
neural network for multivariate time-series forecasting. Adv Neural Inf Proces Syst.                                 47. Anggraini P, Amin M, Marpaung N. Comparison of weighted moving average
(2020) 33:17766–78. doi:       10.48550/arXiv.2103.07719                                                          method with double exponential smoothing in estimating production of oil palm fruit.
  43.                                    Zheng C, Fan X, Wang C, Qi J. Gman: a graph multi-attention network for traffic Building of Informatics, Technology and Science (BITS). (2022) 4:705–22. doi:            10.47065/
prediction. In Proceedings of the AAAI conference on artificial intelligence (2020) (Vol.                         bits.v4i2.2066
34, pp. 1234–1241).
  44. Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward                                  48. Liu XD, Wang W, Yang Y, Hou BH, Olasehinde TS, Feng N, et al. Nesting the SIRV
neural networks. In Proceedings of the thirteenth international conference on artificial                          model with NAR, LSTM and statistical methods to fit and predict COVID-19 epidemic
intelligence and statistics (2010) (pp.      249–256).                      JMLR Workshop and                     trend in Africa. BMC Public Health. (2023) 23:138. doi: 10.1186/s12889-023-
Conference Proceedings.                                                                                           14992-6
Frontiers in Public Health                                                                                   20                                                                                           frontiersin.org

Liu et al.                                                                                                                                                10.3389/fpubh.2024.1397260
Glossary
 AMAE                                                                                   Average Mean Absolute Error
 ARIMA                                                                                  Autoregressive Integrated Moving Average
 ARMSE                                                                                  Average Root Mean Square Error
 AR                                                                                     Autoregressive
 CNN                                                                                    Convolutional Neural Network
 DFT                                                                                    Discrete Fourier Transform
 DNN                                                                                    Deep Neural Network
 GFT                                                                                    Graph Fourier Transform
 G LU                                                                                   Gated Linear Unit
 GNN                                                                                    Graph Neural Network
 G RU                                                                                   Gated Recurrent Unit
 HA                                                                                     Historical Average
 IDFT                                                                                   Inverse Discrete Fourier Transform
 IGFT                                                                                   Inverse Discrete Fourier Transform
 LSTM                                                                                   Long Short-Term Memory
 MAE                                                                                    Mean Absolute Error
 RMSE                                                                                   Root Mean Square Error
 RNN                                                                                    Recurrent Neural Network
 SMA                                                                                    Simple Moving Average
 VA   R                                                                                 Vector Autoregressive Model
 WMA                                                                                    Weighted Moving Average
Frontiers in Public Health                                                                   21                                                                             frontiersin.org

