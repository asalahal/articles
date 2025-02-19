                                                                        Contents lists available at ScienceDirect
                                                           Artificial Intelligence In Medicine
                                                             journal           homep               age:      www.el               sevier.com/l                    ocate/artmed
Graph  neural  network  modelling  as  a  potentially  effective  method  for
predicting  and  analyzing  procedures  based  on  patients'  diagnoses
Juan  G.  Diaz  Ochoa               a,*,  Faizan  E  Mustafab
aPerMediQ GmbH, Pelargusstr. 2, 70180 Stuttgart, Germany
bQUIBIQ GmbH, Heßbrühlstr. 11, D-70565 Stuttgart, Germany
ARTICLE                          INFO                         ABSTRACT
Keywords:                                                     Background: Currently, the healthcare sector strives to improve the quality of patient care management and to
Graph neural networks                                         enhance/increase its economic performance/efficiency (e.g., cost-effectiveness) by healthcare providers. The
Recommender systems                                           data stored in electronic health records (EHRs) offer the potential to uncover relevant patterns relating to dis-
Diagnoses                                                     eases and therapies, which in turn could help identify empirical medical guidelines to reflect best practices in a
Medical procedures                                            healthcare system. Based on this pattern of identification model, it is thus possible to implement recommender
                                                              systems with the notion that a higher volume of procedures is often associated with better high-quality models.
                                                              Methods: Although there are several different applications that uses machine learning methods to identify such
                                                              patterns, such identification is still a challenge, due in part because these methods often ignore the basic
                                                              structure of the population, or even considering the similarity of diagnoses and patient typology. To this end, we
                                                              have developed a method based on graph-data representation aimed to cluster ‘similar’ patients. Using such a
                                                              model, patients will be linked when there is a same and/or similar patterns are being observed amongst them, a
                                                              concept that will enable the construction of a network-like structure which is called a patient graph.1 This
                                                              structure can be then analyzed by Graph Neural Networks (GNN) to identify relevant labels, and in this case the
                                                              appropriate medical procedures that will be recommended.
                                                              Results: We were able to construct a patient graph structure based on the patient's basic information like age and
                                                              gender as well as the diagnosis and the trained GNNs models to identify the corresponding patient's therapies
                                                              using a synthetic patient database. We have even compared our GNN models against different baseline models
                                                              (using the SCIKIT-learn library of python) and also against the performance of these different model-methods.
                                                              We have found that the GNNs models are superior, with an average improvement of the f1 score of 6.48 % in
                                                              respect to the baseline models. In addition, the GNNs models are useful in performing additional clustering
                                                              analysis which allow a distinctive identification of specific therapeutic/treatment clusters relating to a particular
                                                              combination of diagnoses.
                                                              Conclusions: We found that the GNNs models offer a promising lead to model the distribution of diagnoses in
                                                              patient population, and is thus a better model in identifying patients with similar phenotype based on the
                                                              combination of morbidities and/or comorbidities. Nevertheless, network/graph building is still challenging and
                                                              prone to biases as it is highly dependent on how the ICD distribution affects the patient network embedding
                                                              space. This graph setup not only requires a high quality of the underlying diagnostic ecosystem, but it also re-
                                                              quires a good understanding on how patients at hand are identified by disease respectively. For this reason,
                                                              additional work is still needed to better improve patient embedding in graph structures for future investigations
                                                              and the applications of this service-based technology. Therefore, there has not been any interventional study yet.
    Abbreviations: GNN, Graph Neural Networks; RF, Random Forest Classifier; LR, Logistic Regression; ET, Extra Trees Classifier; KNN, K Neighbors Classifier; DT,
Decision Tree Classifier; EHR, Electronic Health Record; PHI, Patient Health Information; IQI, Inpatient Quality Indicators; PQI, Prevention Quality Indicators; PSI,
Patient Safety Indicators; TK, Therapy Keys (equivalent to medical procedures); ICD, International Classification of Diseases; ID, Internal Patient identification; GIE,
Graph Information Encoding.
 *Corresponding author.
    E-mail addresses: juan.diaz@permediq.de (J.G. Diaz Ochoa), faizan.e.mustafa@quibiq.de (F.E. Mustafa).
 1 In several fields in science the concept of network is preferred over the concept of graph, which is commonly employed in informatics. In this article we will
continue using the concept of graph due to the fact that the implemented methodology relies on methods coming from informatics but we will point out to the
concept of network when we explicitly require a more visual definition that considers nodes interlinked with edges.
https://doi.org/10.1016/j.artmed.2022.102359
Received 14 December 2021; Received in revised form 29 June 2022; Accepted 9 July 2022

J.G. Diaz Ochoa and F.E. Mustafa
1.                 Introduction                                                                                          The goal is to find a good method to effectively and efficiently extract
                                                                                                                    relevant    features    from    electronic    health    records    (EHR)    with    little    in-
     The analysis of electronic patient records (EHRs) is not only a source                                         formation loss to solve these problems. For example, the codification and
of        information       on       the       patient's       condition       to       obtain       useful        data       for representation         of         knowledge         from         structured         and         unstructured         re-
diagnosis   and   patient   management,2 but   also   to   find   empirical   guide-                                positories    was    the    prerequisite     to    design    such    expert    systems    based    on
lines      for      the      best      possible      medical      procedures      for      the     patient      [1,2]. graph representations [5].
Therefore,    the    integration    of    the    information    contained    in    EHRs    and                           There       have       already       been       ongoing       projects       leading       to      the       Graph
their modelling is essential to improve patient management, for example                                             representation      of      ICDs      as      well      as      GNNs      to      generalize      the      ability      of
for the design of individual and adaptive therapies [3], or the extraction                                          learning  implicit  medical  concept  structures  into  a  wide  range  of  data
of     quality     indicators.     Up     to     the     present     such     analysis     are     performed        sources.4 These applications range from the evaluation of health-records
based  on  well-established  parameters  like  Inpatient  Quality  Indicators                                       for  its  prediction  of  the  stay  time  of  patients  in  intensive  care  stations
(IQI),   Prevention   Quality   Indicators   (PQI)   or   Patient   Safety   Indicators                             [7], the analysis of the ontology structure of ICD-9 codes [8,9], and the
(PSI)      [4].      However,      these      parameters      are      measured      as      consolidated           application    of    Graph-Concepts    in    predicting    adverse    pharmacological
metrics, ignoring the inherent patterns and dynamics in a way therapies                                             events by modelling of the interactome and polypharmacy [10]. Based
and          medical          procedures         are         being          recommended          or         prescribed          to on    this    initial    research,    it    is    expected    that    in    the    years    to    come,    the
patients.                                                                                                           application of GNNs to the analysis of EHRs  will become  more ubiqui-
     In this research, we want to evaluate these IQI indicators (simply as a                                        tous, particularly in the empirical identification of medical guidelines.
proof    of    concept),    i.e.    the    evaluation    of    these    procedures    where    it    is                  Of  course,  medical  guidelines  are  available  from  the  literature  and
evident    that    a    higher    volume    of    processes    is    associated    with    its    high              are currently applied in healthcare services. The problem is the lack of
quality  [4].  To  this  end,  we  do  not  evaluate  consolidated  volume,  but                                    information    on    the    use    of    these    guidelines    in    actuality.    For    this    very
rather  identify  the  patterns  of  diagnoses  and  medical  procedures  with                                      reason, these  methods  are  required in  empirical research  to  show how
each individual patient who has a chronic disease to find out how these                                             these      guidelines      had            “        emerged”                   from      health      records.      Therefore,      our
diagnoses (encoded by the International Classification of Diseases ICD-                                             main  goal  for  our  research  is  to  identify  the  correlations  between  TKs
10)   are   correlated   with   the   medical   procedures   recommended.   In   this                               and ICD-10 groups, such that TKs can be predicted by providing patients'
way,  we  have  developed  a  novel  method  based  on   Graph  Neural  Net-                                        ICDs as well patients' metadata. An additional feature is the identifica-
works (GNNs), which enables a detailed and coarse-grained IQI analysis.                                             tion   of   patient   clusters,   as   well   as   the   characterization   of   certain   ICD
Based on the evaluation of these IQI indicators, the goal is to provide a                                           combinations and structures, which in turn will allow the identification
simple   system   to   assist   physicians   in   recommending   the   best   possible                              of  patients'  group  with  specific  morbidity  combinations.  Based  on  this
medical   procedure.   As   healthcare   professionals   are   always   confronted                                  correlation   between   ICDs   and   TKs,   it   is   therefore   possible   to   assess   if
with the selection and simultaneously coding of medical procedures on a                                             such  patient  groups  are  getting  the  same  therapy  or  are  there  any  de-
daily   basis,   they   often   ignore   alternative   procedures   or   even   forget   to                         viations within such clusters.
code   a  selected  medical  procedure.  Thus,  when  using  this  framework,                                            While      the      goal      has      a      practical      character,      i.e.,      the      application      of
we have designed a system that, firstly works like a “        remembering system”                                   clustering methods to identify correlations and patient groups, it is also
to    assist    and   help   physicians   to    retain   items    and   to    better   navigate   a                 crucial to resolve the methodological problem, namely the identification
complex   ontology  system.3 Secondly,  it   could  also  assist  physicians  in                                    of       the       best       possible       method       to       reconstruct       the       correlation       amongst
their final decision on whether a procedure is appropriate and/or correct                                           similar patients.
or not for their patient at hand.                                                                                        In the first part of the research paper, we have provided an overview
     With         regard         to         this         problem,         Hema         et         al.         proposed         graph-like of the implemented methodology, including the description of the basic
structures   and   methods   that   capture   and   represent   these   codes   in   the                            input    structures,    the    description    of    the    base    line    models,    the    method
form       of      knowledge      from       semantic      structured      repository      with      better         applied to the ICD-Graph definition and the implementation of the GNN
materialization,   quality,   and   repository   utilization,    along   with   logical                             method to predict TKs. In the results section, we presented the output of
unity and logical entailments to form conceptual relationships between                                              the analysis and made a critical comparison between the accuracy of the
them [5]. This method aligns to linked health data, which is considered                                             GNN model in regard to the baseline models, as well as the application of
as    a    relevant    way    to    leverage    the    complex    information    stored    in    the                this       methodology       for      subsequent       clustering       analysis.      Thereafter,       we
health system [6].                                                                                                  have  explored  the  present  model  and  its  results,  considering  potential
     In      our      case,      we      have      analyzed      data      from      patients      with      multiple ethical concerns too. Finally, we presented the main conclusions found
morbidities,  which meant  that each patient can be assigned to specific                                            in this study.
combinations        of        ICD        groups.        Accordingly,        groups        of        therapies        and
accountable      medical     procedures     are      coded     by     the     so-called     Therapy                 2.                Methods
Keys (TKs). A good and sound disease management data system requires
a solid knowledge of how ICD and TK clusters arise and how it correlates                                            2.1.                 Data  extraction  and  synthetization
in patient populations. Of course, the detection of these clusters leads to
an  improved  clinical  and  economic  management of  the  disease.  But  in                                             For this study we have obtained a highly qualitatively fully synthetic
order  to have a  better knowledge of the system, we are faced with the                                             database        (with    <   1        %        mean        error)        computed        from        an        anonymized
following challenges:                                                                                               database.     The     use     of     synthetic     records     is     a     risk     avoidance     strategy     in
                                                                                                                    protecting patient information, eliminating any potential risk of data privacy
 •Unknown frequency of use of TKs regarding ICDs. Discover clusters                                                 breaches    encountered    by    de-identified    PHI5 [11]    [12].   Furthermore,    the
     of ICDs and TKs;                                                                                               analysis of real-life data for experimental purposes requires the explicit
 •Unknown correlations of TKs to ICDs in real databases;                                                            authorization of each patient, even when the data is anonymized, which
 •Unknown patient clustering after considering diseases and therapies.                                              is not the case in this study. Despite this, synthetic data could bias real
                                                                                                                    data contained in EHRs [13], we decided to adhere to very strict ethical
  2  For  example, of  COVID-19 patients
  3  Note   that   in   this   framework,   the   concept   of   the   recommendation   system,                       4  https://github.com/NYUMedML/GNN_for_EHR.
which is usually used in areas such as e-commerce, cannot be applied directly to                                      5  https://cmte.ieee.org/futuredirections/tech-policy-ethics/2018articles
healthcare.                                                                                                         /ethical-issues-in-secondary-use-of-personal-health-information/.

J.G. Diaz Ochoa and F.E. Mustafa
guidelines    in    this    study,    and    thus    use    this    synthetic    data    to    continue
exploring   the   performance   of   different   algorithms   without   risking   the
data  safety  of  real  patients  and  before  using  these  results  for  interven-
tional purposes, medical decisions or for any countability in medicine.
        In    addition,    we    had    cloned    the    synthetic    patient    information    and
thus  generated a completely new and larger database  (data augmenta-
tion)    for    machine    learning    research.    The    data    cloning    was    generated
using     the     synthpop     package     implemented     in     R     [14].     In     Table     1,     we
present the main parameters of this database.
2.2.                 Multilabel  matrices
        The task is to predict the Therapy Keys (TK) for a patient depending
on      the      diagnosis      (ICDs)      and      additional      metadata,      like      patient's      age,
gender,  and  pertinence  to a  specific  health  center (geographical  distri-
bution). For this, we represent the TKs and ICDs as a multi-label matrix
̂                  I×M
M     ICDϵℝ                ,   where   M   is   the   number   of   unique   ICDs   and   I   is   the   total
number of patients, such that each element in the matrix can be defined
as M        im     .
            ICD
        The          ICD-10          catalog         contains     >   140,000          labels,          which         is          rather
intractable in a single model. Since we are dealing with chronic ill pa-
tients, the specific combination of morbidities and co-morbidities entails
a smaller ICD space, consisting of 252 items.
        This  definition  implies  that  each  row  in  ̂M                                                           ICD      is  the  corresponding
ICD vector of patient i, which has a length m, and where each element is
the number of times an ICD is assigned to each patient, considering that
we evaluated the contained data in a specific time period (for instance 6
months). An example of the ̂M                                               ICD is shown in Table 2: For instance, the
patient identified  as 22,045  (third row), has been controlled  and  diag-
nosed several times, in this case with the same diagnosis identified with
the   ICD   codes  N18.2  (chronic   kidney  disease,  82  times),  D64.8  (other
specified anemias, 41 times), and N25.8 (other disorders resulting from
impaired renal tubular function, 41 times) respectively.
        In a similar way, a multi-label matrix of TKs,  ̂M                                                                       TKϵℝN×I              with N the
number    of   unique   TKs    (50   in   our   study),   can   also   be    assigned   to   the
patient's population, where each row is a vector of length I assigned to
each patient i of unique therapy keys TKs. However, each element in this
vector is binary, i.e., the patient gets a given therapy (M                                                                                   TK     =  1) or not
(M      TK =  0). Thus, this definition implies that we will essentially have a
multilabel problem.
        According   to   this   definition,  ̂M                                          ICD      and  ̂M           TK     are   high   dimensional
matrices that are difficult to reduce. The heterogeneity of the distribu-
tion of the used-frequency of ICDs and TKs poses a challenge: as we were
evaluating   patients   with   a   characteristic   chronic   disease   associated   to
specific   co-morbidities,  there  were  frequent  ICDs   and  TK  combination
associated to the more frequent morbidities/co-morbidities, while other
diseases     were     less     represented     in     the     total     ICD     distribution.     For     this
reason,   we   have   managed   to   obtain   a   heterogeneous   ICD   distribution
Table 1
Principal parameters in the synthetic database.
    Variable                                                                                    Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Kind
    GI                                                                                                                                              Geographical Identification                                                                                                                                                                                                                                                                                    Character
    ID                                                                                                                                              Internal Patient Identification                                                                                                                                                                                                                                                             Numerical
    Gender                                                                                                                                                                                                                              –                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Binary
    Age                                                                                                                                                                                                                                                                                                           –                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Numerical
    TK                                                                                                                                          Therapy Key, equivalent to medical procedures                                                                                    Character
    ICD                                                                                                                                 International Classification of Diseases                                                                                                                                                                        Character
(see Fig. 1).
        If       we       are       dealing       with       larger       ICD/TC       spaces,       it       is       necessary       to
perform a dimension reduction to cluster the ICDs and TKs. From these
clusters   we   receive   information   about   the   structure   of   the   population

J.G. Diaz Ochoa and F.E. Mustafa
depending on the current epidemiological situation (ICD clustering) as                                             try to deduce a graph reconstruction method following statistical rules
well       as       the       medical       management       of       the       population       (ICD       clustering like the degree of distribution of edges, graph topology, etc. [15]. If the
limited to TKs). However, any clustering methodology implies an ad hoc                                             derived    graph   complies    with    empirical    observations,    then    it   makes    it
way to compress these matrices, implying a loss of information. Since we                                           possible to validate the identified graph.
are dealing with 252 different ICDs, it is sufficient for this investigation                                            We assumed that the available data from the health records stored as
to count their frequency to perform the subsequent matrix construction.                                            a structured database can also be stored in a graph structure [16], with
     Additional     meta     parameters     include     the     patient's     age     and     gender               nodes   representing   individual   patients   and   edges   representing   similar
were  also  used  as  input,  both  in  the  baseline  models  and  in  the  graph                                 inherent    causal    relationships    between    patients     like    similar    diagnosis,
NNs.     These     meta     parameters     are     additionally     normalized     for     further                 age  and  gender. However,  there  is  no  a  priori  information  in  the  data-
estimation of the score for patient embedding in the matrix: the age of                                            base about the graph structure of the health data. Thus, a methodology
each patient is defined as αi              =      agei   , and the binary parameter μi is a                        for identifying the graph structure must be identified.
                                               max(age)
binary parameter representing the patient's gender (0 for women, 1 for                                                  Several   methods   are   available   from   the   literature   that   are   particu-
men).                                                                                                              larly        well       suited        to       derive       these       graph        structures       from       EHR.       [17].
                                                                                                                   However,  identifying  the  graph  structure  is  challenging  because  gener-
2.3.                 Baseline  models                                                                              alizing to invisible nodes requires “aligning” newly observed subgraphs to the
                                                                                                                   node embeddings that the algorithm has already optimized [18]. In addition,
     The    ICD    frequency    from    the   ̂M            matrix,    and    the    patient's    meta             we work with a  graph generation  without reference to other similar graphs.
                                                      ICD                                                          Therefore,  we  need  an  unsupervised  learning  method  to  identify  this  graph
parameters, are used as an input feature to train different models aiming                                          structure to integrate different kinds of data.
to predict the TK labels, i.e., the ̂M               TK matrix.                                                         Some methods assume an a priori existence of the edges and the use
     The following methods were implemented as baseline models in the                                              Embedding   propagation  (Ep)  combined   with  a  score  threshold  for  the
current study:                                                                                                     edges     to     define     the     graph     structure     from     patient     data     [7].     We     have
                                                                                                                   chosen      a      method,      based      on      the      definition       of      its      pairwise      similarity
 •Random Forest classification (RF)                                                                                instead, which has been reported as an effective method over embedding
 •Logistic Regression (LR)                                                                                         propagation [7]. Furthermore, with our implemented method, we do not
 •Extra Trees Classifier (ET)                                                                                      require ad-hoc definitions or parameters, like the number of clusters by k
 •K-Neighbors Classifier (KNN)                                                                                     or c-means clustering methods, as the graph-based clustering establishes
 •Decision Tree classifier (DT)                                                                                    a direct correlation between ICDs and TKs.
                                                                                                                        Furthermore,      though      embedding      methods      have      been      successfully
     While these methods can be applied in a straightforward way, they                                             proven     in     numerous     applications,     they     are     subjected     to     fundamental
strongly  compress  information  about  the  structure  and  the  relatedness                                      limitation: their ability to model complex patterns is inherently bounded
between patients, which is important for further clustering analysis. In                                           by  the  dimensionality  of  the  embedding  space  [19].  We  have  resolved
the  next section  we  demonstrate  how  Graph-NNs  can retain  this  infor-                                       this problem by performing patient embedding in the graph object G, as
mation,  thus  enabling  us  to  perform  clustering  analysis  on  the  patient                                   shown in Fig. 2, by encoding the corresponding ICD distributions to an
population. All these models are trained with the default parameters as                                            object representing the interlinking between similar patients, an object
defined in the SCIKIT-learn library.                                                                               that can be defined as a similarity score ̂G                    ij, which is essentially for the
                                                                                                                   adjacency matrix in linking patients i and j.
2.4.                 Graph  clustering   –                and generation  of  graph  data                               Since  we  have  in  our  example  256  different  ICDs,  we  were  able  to
                                                                                                                   generate  patient  embedding  by  directly  counting  the  ICD  frequency  in
     The aim of this method is to perform patient clustering assuming that                                         the system. This procedure is however limited when more ICDs are used
each patient k, owning the same or similar ICD pattern, is linked to other                                         for the codification (considering that there are about 9000 different ICD
patients with similar ICDs, such that patients with a similar combination                                          codes),       as        the       length       of       the       vectors       become        intractable,       making       it
of morbidities and comorbidities, should get similar TKs. The advantage                                            necessary  for  application  of  methods  such  as  autoencoders  (which  are
of this method over traditional clustering is its flexibility, such that new                                       switched   off    in   the   current   implementation).   After    generating   patient
graphs with different connectivity can easily be evaluated.                                                        embeddings     graphs,     they     are    then     used     as    vertex     embedding     for    the
     Graph       generation       still      remains      as       one      of      the       most       complex      and computation of the graph structure.
challenging      task      in      graph      science.      The      only      way      to      implement      this     Since a fix clustering method aggregates information in the defined
generation methodology is to assess the intrinsic graph statistics and to                                          cluster,  with  these  graph  structures  we  had  been  aggregating  informa-
                                                                                                                   tion   on   each   node   encoding   the   similarity   between    patients   [7].   This
                                                                                                                   similarity  is  defined  by  means  of  a  score ̂G                ij,  which  is  a  function  that
                                                                                                                   depends on the ICD distribution as well as the gender and age related-
                                                                                                                   ness between patient i and patient j
                                                                                                                            M                              )                  ⃒
                                                                                                                   dij=∑        gm  smsm   +c[δ μ,μ           1     ⃒⃒αi αj⃒)],                                                                                                                                                                                                                                                                                                 (1)
                                                                                                                                  ij  i  j            i   j
                                                                                                                           m=1
                                                                                                                   where gm is essentially a weight for the ICD distribution for patients i and
                                                                                                                              ij
                                                                                                                   j,      m      is      the      number      of      elements      of      each      of      the              s→i vector,      i.e.              s→i=
                                                                                                                   {s1,s2,…          ,sm,…        },   and   M   is   the   total   number   of   columns   of   the  ̂M  ′
                                                                                                                      i    i        i                                                                                     ICD
                                                                                                                   (matrix   with   reduced   dimension   from  ̂M                ICD  matrix).   Additionally,   we
                                                                                                                   introduce      a      scoring      that      depends      on      the      patient's      meta      parameters,
Fig. 1.            Typical distribution of ICDs, where the y axis is the ICD frequency (as a                       where δ(μi,μj) is the Kronecker's delta (1 if μi =μj, 0 otherwise), and c is a
logarithmic   scale).   Observed  is  that  the  distribution   is  almost  scale  free,   with                    parameter (we call this parameter “        importance hyper parameter”        ) such
certain  ICDs  owning  a  larger  frequency  (frequent  morbidities)  joined  by  ICDs                             that    patients    with    a    similar    gender    and    similar    age    get    a    high    score,
with  less frequency (co-morbidities).                                                                             biasing the initial ICD scoring. The goal of this definition/formula is to

J.G. Diaz Ochoa and F.E. Mustafa
                                                                                                                                                                                  Fig.  2.            Graph  with  patient  embeddings
                                                                                                                                                                                  si,        represented       by        a       list        (vector)        that
                                                                                                                                                                                  integrates    the    patient    diagnoses    (enco-
                                                                                                                                                                                  ded by ICDs) and basic patient data like
                                                                                                                                                                                  patient's   sex   and   age,   for   each   individ-
                                                                                                                                                                                  ual patient pi. The central patient p1 has
                                                                                                                                                                                  three           different           ICDs            (ICD_1,           ICD_14,
                                                                                                                                                                                  ICD_20) and is a 27 old woman. Despite
                                                                                                                                                                                  the neighboring nodes (similar patients,
                                                                                                                                                                                  p2,      p3,      and      p4)      are      all      male      and      with
                                                                                                                                                                                  different ages, they share common ICDs
                                                                                                                                                                                  with p1; this similitude is represented by
                                                                                                                                                                                  an             edge             between             the             patients.             The
                                                                                                                                                                                  thickness  of  the  link  (edge) codifies  the
                                                                                                                                                                                  degree        of        similitude        between        the        pa-
                                                                                                                                                                                  tients. On the other hand, patient p5 has
                                                                                                                                                                                  completely    different    ICDs,    and    for    this
                                                                                                                                                                                  reason    there    is    no    connection    between
                                                                                                                                                                                  both  patients  (despite  both  patients  are
                                                                                                                                                                                  women).     Observe     that     the     links     repre-
                                                                                                                                                                                  sents         an         inherent         causal         relation         be-
                                                                                                                                                                                  tween           patients           because           they           express
                                                                                                                                                                                  similar        symptoms        and        therefore        have
                                                                                                                                                                                  diseases with a common causal relation.
have a graph construction of similar patients, based not only on similar                                                 definition    of    the    distance    in    the    nearest-neighbors    algorithm,    we    can
diagnoses, but also with those with similar age and gender.                                                              therefore  consider  gm as  a  definition  of  the  background  metric  used  in
                                                                                                                                                          ij
     This  score  will  then  be  used  to  define  the  strength  linking  the  two                                     the construction of the network. With this modelling strategy, we aim to
nodes. The corresponding mathematical structure of the similarity score                                                  define    networks    with    uniform    topologies    and    avoid    the    formation    of
̂                                                                                                                        hubs.
G  ij is a function that depends on this relatedness dij and the clustering of
similar elements in the graph, and which essentially works as the adja-                                                        Considering these aspects, it is necessary to examine which form of
                                                                                                                         gm   is   more   appropriate   to   perform   network   encoding.   We   have   tested
cency matrix linking the nodes pi and pj.                                                                                  ij
     For    the    definition    of    this    matrix,    neighboring    elements    should    be                        different  definitions  following  the  results  reported by  Nickel  and  Kiela
recognized        depending        on        the        distance        between        pi and        pj.        For        this [19].
computation we have estimated the nearest neighbors using a k-Nearest                                                          To     validate     the     discovered     graph     structure,     we     have     employed     a
neighbors' method by finding a predefined number of training samples                                                     second similarity score to measure the ICD similarities between patients
closest  in  distance  to  the  new  point  and  predict  the  label  from  these67                                      using   an   adequate   similarity   measure   [22].   This   similarity   measure   is
[20], such that                                                                                                          based    on    the    overlap    measure8            [22],    and    is    defined    as    the    similarity
                                                                                                                                                                                                  ∑  I   ∑   J  Sim
                                                               ⃒                                                         between one node and its neighbors as Sim   =                               i=1     j=1    ij , i.e. the sum
̂                                                 ⃒            ⃒                                                                                                                                          J•I
G  ij=̂G         dij(a),k,r)=dij⇔            r>    ⃒dijk  dij→0                  forj∈{j1,j2,…          ,jk},                                            (2)  over all the nodes i of the similarity scores computed on each node and
                                                                                                                         its corresponding interlinked nodes j, where Sim                                =  |s    ⋂         s|.
where   k   is   the   maximal   number   of   nearest   neighbors   explored   by   the                                                                                                               ij       i      j
clustering algorithm,a is an internal calibration parameter of the metric,                                                     The reason for using an overlap value over a Jaccard measure is that
and r is the radius where the clustering is performed.                                                                   the node embeddings are defined as vectors that have a different number
     For this evaluation, different plausible definitions of the metric can                                              of elements between different patients.9 Since there is no high similarity
be found. This selection  depends on the topology where  the points are                                                  value   expected   from   the   derived   graph   and   the   individual   nodes   (pa-
embedded:   since   we   are   dealing   with   high   dimensional   spaces   with   a                                   tients)  are  still  very  heterogeneous,  so  even  if  the  graph  structure  was
complex  distribution, it is  expected that the best suited metric be non-                                               derived   with   a   similarity   value   based   on   a   distance   concept   (in   some
Euclidean      [21].      In      our      model      we      define      gm in      relation      to      the      total cases    with    larger    local   similarity    values),    we   do    not    expect   to    get   an
                                                                       ij
weight   of   ICDs   measured   as   Tm              = ∑     I   M   im  ,   such   that   one   plausible               overlap value of 1 (full overlap) then, but rather at least 0.5. This means
                                                             i=1     ICD                                                 that at least 50 % of the ICDs of the neighboring agents overlap with the
definition of gm is gm            = a •1      , where a is a constant. Therefore, we give
                      ij      ij           Tm                                                                            ICDs   of   the   reference   node.   The   calculated   values   will   be  >   0.5   if   the
more        weight       to        the       ICDs        with        low       frequency        which        can       be        easily number of neighbors per node is <   4, suggesting that the structure is not
neglected        during        training.       Since        this        definition        influences       the        final simply  random.  For  the  final  computation,  we  have  defined  the  nodes
                                                                                                                         with   4   neighbors,   which   delivered   a   similarity   score   of   0.53   for   both
  6  https://scikit-learn.org/stable/modules/neighbors.html.
  7  NearestNeighbors   implements   unsupervised   nearest   neighbors   learning.   It                                    8  https://towardsdatascience.com/calculate-similarity-the-most-relevant-me
acts     as      a     uniform     interface     to      three     different     nearest     neighbors      algorithms:  trics-in-a-nutshell-9a43564f533e
BallTree,     KDTree,    and    a    brute-force    algorithm    based    on    routines    in    sklearn.                  9  https://developer.nvidia.com/blog/similarity-in-graphs-jaccard-versus-the
metrics.pairwise.                                                                                                        -overlap-coefficient-2/

J.G. Diaz Ochoa and F.E. Mustafa
metrics.     This     similarity     score     will     tend     to     decrease     by     an     increasing                                  where   TPλ is   the   total   number   of   observed   true   positives,   and   FPλ the
number of neighbors: for J >           4 the similarity score is about 0.4.                                                                    total    number    of    observed    false    positives    of    the    label  λ.    Similarly,    the
                                                                                                                                               micro-recall is defined as
2.5.                 Graph  neural network  (GNN)  model                                                                                                       ∑    TP
                                                                                                                                                                 λ       λ
      The  graph  encoding,  as  shown in  Fig. 3,  is  then  used as  input  for a                                                            R  =∑       TPλ+∑           FNλ,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             (5)
convolutional neural graph in order to identify corresponding labels for                                                                                λ               λ
each patient, located in each node of the network10 respectively. In our                                                                       where  FN            the  total  number  of  observed  false  negatives  of  the  label λ.
                                                                                                                                                                 λ
case,         the        labels        are        the        corresponding        TKs        assigned        to        each        node        According to these both definitions, the f1 score was defined as
(patient).                                                                                                                                              2    •P •R
      Since    we    are    considering    cumulated    events    after    several    different                                                f1     =
periods    of    time,    we    essentially    encode    this    information    as    static    em-                                                        P +R       .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (6)
beddings.11 Additionally, the corresponding model meta parameters are                                                                                This       methodology       seems       to      be       plausible      considering       that      in       the
tuned during the model training in short epoch intervals to minimize the                                                                       database we have labels in more instances, and as we want to deviate the
training   loss   and   simultaneously   get   the   optimal   parameter.   The   final                                                        validation          metric          towards          the          most          populated          ones,          we          therefore
model structure is provided in the table below (Table 3).                                                                                      implemented   this   kind   of   micro   averaging12 in   order   to   bias   the   final
      The labels are the vectors         s→                  TK    ={ TK1,TK2,…          ,TKN}, in this case                                   result by class frequency,
                                                            i                   i        i              i
the corresponding TKs at the patient node i, with N the maximal number
of labels, i.e., of TKs. Thus, the trained model МGNN takes as input the      s→′
                                                                                                                                    i          3.2.                 Selection of best GNN model: analysis of  ̂ℳ                       ICD    matrix on the graph
vector   containing   the   corresponding   ICDs   and  meta-parameters   of   pa-                                                             construction
tient i and delivers as output         s→                   TK, i.e.
                                                           i
           →′          →    TK                                                                                                                       For the first analysis of the graph construction, we focus on the role
МGNN     (si)=    si            .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 (3) of the ̂Mon the graph definition ̂G, i.e., we assume that c =0. Here,
                                                                                                                                                               ICD                                                   ij
      The embedding at the output of the Graph model can be used to make                                                                       we have performed different validations using models with different gm
                                                                                                                                                                                                                                                                                  ij
new  edges  between  different  patients.  This  new  graph  may  give  us  an                                                                 parameters, according to the score definition in Eq. (1) and Fig. 3. The
improved    edge    connections    as    compared    to    the    graph    we    built    using                                                evaluation of the k-Neighbors was made by setting k =3, a =1, and r =
relatedness score. In order to make a possible edge between node i with                                                                        1.
node,  we  can  use  dot  product  between  the  node  embedding  vectors  of                                                                        We selected these parameters because the results are degraded when
                                                                    TK          TK ij→    TK      →    TK                                      the value of k other than 3 is used. In particular, larger k values imply an
node i and j. The dot product score ̂G                                   = g          s  i     •s     j    is used for the                     increase of the entropy of the system, which leads to more noise in the
                                                                                                                                 TK
definition of the final adjacency matrix restricted to the TKs, where ̂G                                                                       final        result.        Furthermore,        we        have        observed        that        the        radius        r        has
is the similarity score for the TKs. In this case, we simply considered that                                                                   negligible effect on the metric score.
gTK ij  = 1 (see Fig. 4 for the representation of the reconstruction of the TK                                                                       After           performing           different           experiments,           we           have           obtained           the
adjacency matrix).                                                                                                                             following validation results for g1m and g2m, which corresponded to the
                                                                                                                                                                                                              ij              ij
      The  complete  workflow,  from  data  collection  and  low-dimensional                                                                   qualitative results deployed in Fig. 3 (see Table 4). Thus, there is in fact a
data     representation,     using     methods     for     dimension     reduction,     passing                                                dependence             on            the             background            geometry            where             the             network            is
over    graph   encoding    and   graph   neural    networks,    to    ICD   clustering   is                                                   embedded, with a slight influence on the network's hub distribution.
presented in Fig. 5. This model has been implemented in Python using                                                                                 Furthermore, these parameters do have an influence on the rank of
Pytorch,     with     the     corresponding     libraries     for     graph     neural     networks                                            the  scoring  parameter ̂G                       :  while  g1m delivers  a  scaling  with  values  as
                                                                                                                                                                                              ij                   ij
(GNNs).                                                                                                                                        [0,300], g2m delivers a scaling with values as [0, 8.75] respectively, with
                                                                                                                                                                   ij
      In the next section we will present the main results of this study, in                                                                   most      values      ranging      between      0      and      1,      which      then      allows      a      simple
particular      focusing      on      the      comparison      between      classical      and      graph-                                     coupling to the patient meta parameters. Thus, the following results and
based clustering methods.                                                                                                                      final validation were computed using g2m.
                                                                                                                                                                                                                          ij
3.                 Results                                                                                                                     3.3.                 Final validation  and comparison between  baseline  models  and GNN
3.1.                 Validation  metrics                                                                                                             In     the     final    validation     we     have     considered     the    patient's     meta    pa-
      The first main goal is to perform a comparative study of the different                                                                   rameters, i.e., c >           0. As expected, the coupling to the meta parameters
modelling methodologies. To this end, each model is trained using TKs                                                                          influences the final validation: for c =0.2 we obtained f1 =79.7 (in %),
as labels, i.e., the  M               TK matrix, implying that the model is essentially a                                                      while for c =1.0 we obtained f1 = 73.1. We were tempted to select c in
multi-label  classification problem for different λ labels. The micro pre-                                                                     order to get the best possible validation value. However, this could bias
cision was defined as a micro-average                                                                                                          the model too much to the ICD selection and ignore the fact that the TK
                ∑                                                                                                                              selection    depends    on    the    patient's    age    and    gender    as    well.    Since    the
                    TPλ                                                                                                                        scaling of the selected metric has the most values between 0 and 1, we
P  =∑            λ                 ,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (4)  fixed c =1.0, which delivers a rank scoring ̂Gof [0,2] in order to give
            TPλ+∑           FPλ                                                                                                                                                                                                      ij
         λ               λ                                                                                                                     importance to the patient's meta parameters.
                                                                                                                                                     We have thus selected the best performing GNN model and compared
                                                                                                                                               it against the baseline models to assess the quality of the GNN models.
                                                                                                                                               The results of this comparison are shown in the Table 5 (Notice that the
 10   See Table 4 for the final validation results and final assessment of the best                                                            scores  for  the   GNN  models  are   smaller  when   considering  the   patient's
possible  weight  gm.
 11                          ij
      In future applications, we must consider the evolution of graphs according
to  changes  in  the  graph-encoding  at  different  time  period,  which  might/could                                                          12   https://stats.stackexchange.com/questions/156923/should-i-make-decis
lead  to a dynamic  network  structure.                                                                                                        ions-based-on-micro-averaged-or-macro-averaged-evaluation-mea

J.G. Diaz Ochoa and F.E. Mustafa
Fig.  3.            Qualitative  comparison  of  the  network  encoding  linking  patients  owning  similar  ICDs,  using  the  same  rule  presented  in  Fig.  2,  considering  two  different
metric  definitions  for  the  computation  of  patient  neighbors.  The  different  colors  represent  different  health  centers  (5  different  colors  depicting  5  different  health
centers  with a  different  geographical distribution). Observe  the  sensitivity in  the  hub distribution depending  on the  metric  definition.
Table 3                                                                                                                                           further relevant information in the population.
Parameters for each layer defined in our GNN model.
   Parameter                                                                                                  Meaning                                                                                                                                                      Value                                                                                                                               Activation function 3.4.                 Model  output  and  clustering  analysis  of  TKs
   Layer 1                                                                                                                             Convolutional                                                                                                  (252,32)                                                                                                  Relu, dropout 0.5 After   training   the   GNN   model,   we   were   able   to   use   it   for   the   pre-
   Layer 2                                                                                                                             Convolutional                                                                                                  (32,32)                                                                                                              Relu, dropout 0.5 diction of the best TKs. For instance, assuming a male patient, 43 years
   Layer 3                                                                                                                             Convolutional                                                                                                  (32, 50)                                                                                                        Sigmoid
                                                                                                                                                  old,     has     the     following     ICDs     (input     ICDs):     Z94.0     (Condition     after     an
                                                                                                                                                  organ   or   tissue   transplantation  organ),   M32.1   (Systemic   lupus   erythe-
                                                                                                                                                  matosus  with  involvement  of  organs  or  organ  systems),  D90  (Immune
                                                                                                                                                  compromise  after  radiation,  chemotherapy  and  other  immunosuppres-
                                                                                                                                                  sive measures). The system  predicts  and match its prediction with  TKs
                                                                                                                                                  that are already assigned to the patient, for instance the measurement of
                                                                                                                                                  glucose,       chloride,       or       sodium       (output       procedures).       The       predictions
                                                                                                                                                  outside     form     this     matching     group     are     TKs     that     a     health     professional
                                                                                                                                                  might      probably      forget      to     consider,      for      instance      to      perform      a      blood-
                                                                                                                                                  infusion   or   to   measure   the   pH   value   in   patient's   urine   (remembering
                                                                                                                                                  system).
                                                                                                                                                         For this output, some other additional analyses are required:
                                                                                                                                                     •Analysis     of     the     relation     between     geographical     distribution     of     the
                                                                                                                                                         patients and predicted TKs;
                                                                                                                                                     •Analysis of the ICD prominence in the prediction of TKs.
Fig. 4.            By using the TKs as labels we aim to use an ICD-graph (in blue left) to
train   a  model   and  predict  TKs  order   as  a  graph  as  well  (in  black,  right).  (For                                                         Based  on  ̂G           TK we   can  reconstruct  a  new  graph  where   patients  are
interpretation    of    the    references    to    colour    in    this    figure    legend,    the    reader    is                               interlinked  if  they  have  similar  TKs.  This  new  network  has  a  structure
referred  to the  web version  of this  article.)                                                                                                 with     clustered     patients,     such     that     each     cluster     reveals     the     similitude
meta parameters here).                                                                                                                            amongst patients depending on their therapies.
      From these results we have concluded that the Graph-NN surpasses                                                                                   This   implies,   from   the   initial   ICD-patient   network  ̂G                             we   have   ob-
the        metric        scores        obtained        by        baseline        methods        by        a        considerable                   tained a structure that defines how similar the TKs assigned to individual
margin. After testing the GNN model over test data, we obtained an f1                                                                             patients are.
score   of   73.1   %,   which   is   satisfactory  when   dealing   with   the   baseline                                                               As we have different health centers, we have hypothesized that this
models;   only   RF   (Random   Forest   Classifier)   was   superior   in   precision                                                            clustering           is            directly            correlated           to            the            health            center            distribution
than the GNN. These results suggest that GNNs seems to outperform the                                                                             (geographical      clustering      represented      by      different      colors,      see      Fig.      6).
other methods, with an average difference (i.e. the averaged difference                                                                           However, with the final output we have identified five different clusters
between   the  validation  metrics  of  GNN   and  the  baseline   methods,  i.e.                                                                 that          originate          from          similar          patient          therapies          (and          not          from          their
Δ        =   mean(Val               Val                 ),  where   Val   is  the   corresponding   valida-                                       geographic distribution). Thus, with this modelling, we can observe how
                           GNN               Baseline                                                                                             patients are usually treated, what is the way to identify specific medical
tion       metric)      of      4.26      for      recall,      5.94      for      precision,       and      6.48      for      f1.               guidelines commonly used to treat patient populations.
Furthermore,  this methodology allows  a detailed clustering analysis  of                                                                                Since   these   guidelines   are   correlated   to   the   ICDs,   we   analyzed   the
the patient population, helping us to better interpret results and obtain                                                                         correlation between the ICDs (left graph structure in Fig. 6) and the TKs

J.G. Diaz Ochoa and F.E. Mustafa
Fig. 5.            Algorithm's architecture. The main steps in this algorithm are the encoding of the input structured health data (stored in an EHR) into graph data, and the
subsequent  modelling of  the graph  data  using convolutional GNNs  to predict OPS.
Table 4                                                                                                                                                     observed)   respect   to   the   input   (different   single   combinations   of   ICDs)  [∑        М)/
                                                                                                                                                                   n                                 →′                     →′
Parameters employed in the K-Nearest Neighbor algorithm.                                                                                                    wTK        =∂                   GNN(s         i)]TKn          ∂s     i(from  Eq.  (3)),  which  depend  on  the
                                                                                                                                                            vector              s→i,      i.e.,      for      TKn then      wTKn       =       {wTKn(ICD1),   wTKn(ICD2),       …                       }      (an
   Weighting                                                                                                 Precision (P in %)                                                                                                 Recall (R in %)                                                                                                 f1 (in %) extended and more intuitive interpretation of this score is provided by
   g1m                               86.1                                                                                                                                                                                                                               69.8                                                                                                                                                                                                    77.1
       ij                                                                                                                                                   [25]). This score is computed either for the whole label-network wall as
   g2m                               87.8                                                                                                                                                                                                                               66.5                                                                                                                                                                                                    76.0
       ij                                                                                                                                                   well as for each one of the sub-networks w                                               , in this case 5 subnetworks
                                                                                                                                                                                                                                                    c
                                                                                                                                                            according to Fig. 6B.
                                                                                                                                                                   Thus, the saliency score basically measures the   “        prominence”               of an
Table 5                                                                                                                                                     ICD       for       the       definition       of       a       TK       based       on       the       analysis       of       the       whole
Comparative   validation   results  between   baseline  models  (Random   Forest  Clas-                                                                     population.
sifier (RF), Logistic Regression (LR), Extra Trees Classifier (ET), K Neighbors Clas-                                                                              In     Fig.     7,     we     use     the     Saliency     Score     (y-axis),     where     we     basically
sifier (KNN), Decision Tree Classifier (DT)) and GNNs: Recall (R), precision (P), and                                                                       discovered 5 different ICD groups for each TK cluster (clusters as shown
f1.  The  final  column  of  the  table, Δ,  is  the  averaged  difference  of  the  validation                                                             in Fig. 6B). Using this method, we can thus identify a pyramid structure
metric computed for the GNN model and the other baseline. Models.                                                                                           in   which   specific   patient   clusters   are   related   to   specific   ICDs   and   TKs
                           RF                                                              LR                                                              ET                                                              KNN                                             DT                                                            GNN Model Δ leading to the discovery of specific guidelines for specific comorbidities
   R (in %)                                             54.0                                        62.9                                       58.5                                        49.2                                          64.1                                        62.0                                                                                                              4.26 distributed around a major morbidity (ICD with the highest distribution
   P (in %)                                               88.7                                        76.5                                       91.8                                        87.1                                          61.2                                        87.0                                                                                                              5.94 in   Fig.   7).   The   x-axis   of   the   plots   are   the   ICDs   that   are   important   for
   f1 (in %)                                        67.2                                        69.0                                       71.4                                        62.9                                          62.6                                        73.1                                                                                                              6.48  predicting TKs for each subgraph. The y-axis shows how the meaning of
                                                                                                                                                            these  ICDs  is  based  on  the  salience  value.  The  last  diagram  shows  the
(right   graph   structure   in   Fig.   6)   by   measuring   the   difference   between                                                                   importance of different ICDs for the full graph.
both structures. To this end we have employed a saliency method, often                                                                                             While     there     are     ICDs     that     are     important     for     the     full     or     complete
used in natural language processing (NLP) and image recognition to find                                                                                     graph,   each   subgraph   has   other   ICDs   that   are   important   for   that   sub-
out  the  significance/relevance  of  input  features  (in  this  case  ICDs)  for                                                                          graph. For example, for the subgraph 3, H26.8 (other specified cataract)
the computation of the output (in this case TKs). In language recognition                                                                                   and    I15.0    (renovascular    hypertension)    are    the    most    relevant    diagnoses
this implies for instance the recognition of relevant words used for the                                                                                    leading to the TKs in this group. However, the H26.8 ontology does not
composition of sentences [23], or in image recognition the pixel ranking                                                                                    seem   to   be   important   in   the   full   graph,   i.e.,   for   the   entire   population
leading to an specific class classification for an image [24], in order to                                                                                  (where Z92.4 was the most relevant diagnose, i.e., medical treatment in
understand how the model is looking into the image.13                                                                                                       the         own         anamnesis).         Simultaneously         the         renovascular         hypertension,
      In this investigation we define the saliency score in the same way as                                                                                 I15.0,     was     common     for     all     the     subgroups     (except     subgraph     0).     This
Simonyan et al. [24], where the score of a label ScTK is in principle a non-                                                                                shows that clusters are identified based on different TKs, and predicting
linear function that can be approximated by a Taylor approximation to a                                                                                     these TKs therefore requires different types of ICDs.
linear function to the model input, in this case the ICD matrix, such that
̅→Sc         =w •s→, where w is the saliency (or importance) score. The score                                                                               4.                Discussion
       TK                     i
w can in this case be computed by back propagation, i.e., the gradient of                                                                                          In    this    research,    we    have    conducted    a    thorough    study    comparing
the        model        output        (difference        of        number        of        times        a        single        label        is              different      AI      methods      and      GNNs      to      extract      patterns      from      EHRs      that
                                                                                                                                                            empirically derive guidelines currently used in a healthcare system. To
                                                                                                                                                            this end, we have also implemented conventional modelling techniques
 13   And for this reason, this method is used for AI explainability. See e.g. https                                                                        in     AI/ML     to     discover     ICD-TK     correlations,     which     are     the     product     of
://medium.datadriveninvestor.com/visualizing-neural-networks-using-salien                                                                                   implicit guidelines currently used by the medical community.
cy-maps-in-pytorch-289d8e244ab4

J.G. Diaz Ochoa and F.E. Mustafa
Fig.  6.            Graph  data  of  patients  correlating  by  similar  TKs.  Left: input  ICD  patient  graph  (with  different  colors  representing  5  different  geographical  regions  where
patients are located). Right: output TK graph, with edges representing patients with similar therapy keys; the different colors represent the geographic distribution as
well.  By  observing,  in fig.  B, we have identified  5  different clusters,  representing patients  with similar TK  groups.
     The derivation of graph-data structures from EHR has the advantage                                             representations. Furthermore, the assigned ICDs for each patient are not
as different kinds of information can be integrated. However, since the                                             free of error: according to some studies, only 63 % of the encoded ICDs
original  data   does  not   have  this  graph  structure,   information   loss  and                                in EHRs are accurate,14 which implies that the inputs used in the model
uncertainties    can    be    cumulated    in    the    transformation    process.    In    our                     may not be accurately enough.
model we were able to generate graph-structures delivering an accept-                                                    An  additional  observation  concerning  the  graph  construction  focus
able  overlapping  score.  Furthermore,  the   general  quality  of  its  valida-                                   on validation score of the model: a model with a small coupling of the
tion,   and   the   model   sensitivity   on   the   metric   definition,   demonstrates                            patient's  meta parameters  with  ICDs  leads  to  a  better  validation of  the
that the GNN is not only learning from the node's features but also from                                            model,   with   a  score   f1   of   77   %,   which   is   much  better   than  the   initial
its graph structure.                                                                                                reported score of 73 %. However, this gives less relevance to factors like
     In addition to improving the GNN results as compared to classical AI/                                          gender and age to the modelling. However, both parameters are actually
ML   approaches,   graph-structure   facilitates   better   cluster   identification                                very        relevant        when        it        comes        to        avoiding        model        bias,        for        instance
which    gives    us    an    intuitive    explanation    of    different    types    of    patients                considering        that        the        ICD        combination        for        a        woman        can        lead        to        a
based      on      their      morbidities/comorbidities      and      TKs.      Therefore,      GNNs                different  therapy  recommendation  as  for  a  man.  Therefore,  this  is  one
allows      us      to     make     additional     cluster      analysis     that      would     have     been      crucial example on how it is better to accept lower accuracy in order to
difficult    to    perform    with    other    methods   due    to    their    inherent    lack   of                preserve and represent basic characteristics of a population.
internal structure. This confirmed other findings from other groups that                                                 Since       we       have       few       ICDs       encoding       the       patient       morbidities       /       co-
have also highlighted the advantage of graph methods for the analysis of                                            morbidities,    we   have   directly   computed    the   vertex   embeddings   using
health data [6]. Finally, graph structures allow an easy model general-                                             the ICD frequency. However, for other patient populations, it is normal
ization since the identification of a structure allows the fast translation                                         to have larger ICD vectors per patient. Considering that currently there
of  the  model  to  other  scenarios  without  requiring  additional  data  and                                     are         about          11,000          different         ICD-10          codes,         the         use          of          compression
training runs.                                                                                                      methods,       like       autoencoders,       will       tend       to       be       necessary       to       make       the
     Of course, this methodology is not applicable to all 140,000 ICD-10                                            implemented workflow functional.
labels,  however  it  is  a  tool  that  is  better  suited  when  dealing  with  pa-                                    Also,   databases  are   evolving,   implying  that  derived   network  struc-
tients having specific morbidities. This implies that before applying this                                          tures   are   not   valid   for   different   times.   To   this   end,   and   following   the
methodology, a dimension reduction of the ICD landscape is required.                                                work of Rocheteau et al. [7] we plan to implement methods to consider
     Despite  the  advantage  of  using  graphs  both  to  predict  and  provide                                    time  series  and  couple  this  information  into  the  node  characterization
structures leading to cluster analysis, we are well aware of the problem                                            according to Fig. 4.15 Since the derivation of the graph-data structure is
in  using networks paradigm as they can introduce undesired biases:  in                                             still   challenging,   a   bi-level   program    can   also   be   implemented   in   next
several     fields     we     observe     how     network-methods     sometimes     establish                       work    in    order    to    describe    the    discrete    probability    distribution    of    the
incorrect interlinking between elements that are not truly correlated, for                                          edges   of   the   graph   [27].   In   order   to   better   account   for   the   persistent
instance the familiar linking of an individual to a criminal family does                                            uncertainties in the graph generation (epistemic uncertainty) we aim to
not imply that the individual is a criminal. Network paradigms reinforce                                            implement    methods    that    could    handle    and    perform    an    improved    sta-
biases and can be even dangerous, particularly when indicators are used                                             tistical prediction in the GNN model for such uncertainty.
to     encode     features     in     graph     structures     [26],     or     when     these     network
representation   is   viewed   as   a   fundamental   natural      “        principle”                of   both     5.                Conclusion
social and biological systems. In this study we have observed the same
problem,   since   Graph   information   encoding   (GIE)   based   only   on   ICDs                                     We have reported the use of GNNs for the identification of correla-
might    ignore    relevant    medical    information    encoded    in    other    sources,                         tion   patterns   between   ICDs   and   TKs.   We   have   demonstrated   that   this
like physician notes or sentiment analysis.                                                                         modelling method has a high accuracy in respect to base-line models, as
     One     option     to     resolve     such     problems     is     to     continue     making     data
enrichment to the present structures including additional data sources.
However,     these    methods     cannot    avoid    the     risk    to    bias     the    model     by
establishing               misleading              links/edges.               Therefore,              more              research               is 14 https://healthinfoservice.com/most-common-icd-10-error-codes/.
required                to                avoid                eventual                misleading                conclusions                from                graph 15 For instance, using LSTM methods to consider events from different times as
                                                                                                                    an additional  node feature.

J.G. Diaz Ochoa and F.E. Mustafa
Fig. 7.            Saliency score computed for the 5 predicted TK clusters (subgraph 0 to subgraph 4), as is shown in Fig. 6B. This saliency score provides information about
which  ICD  (numbered, on  the  x axis)  is  more relevant for  the  prediciton of TK  in each subcluster (y  axis).
it allows a more accurate identification of patient clusters based on their                                         network     models     is     still     required     to     further     improve     this     technology.
ICDs  as  well  as  other  similar  methods.  Furthermore,  this  methodology                                       There are also alternative ways to interpret and implement this problem;
also     allows    a    better    identification     of    patient    groups    with    specific    TK              for instance, a bipartite graph between TKs and patients can be defined.
groups correlated to specific ICD combinations. This result is therefore                                            With the GNN model, the ‘similar patient’         will have similar embeddings,
helpful to identify pyramid-like structures in the ICD distribution, which                                          as  they  may  share similar  features and  similar  structure features (with
is  an important information not only to  leverage the quality of the  pa-                                          the     similar     neighbor     TK).16      Thus,     the     problem     will     in     this     be     a     link
tient management (depending on specific ICD combinations) but also to                                               prediction        between        the        patients        and        the        TKs.        All        these        alternative
better  identify  the  performance  of  health  centers  by  region.  Naturally,                                    implementations,   as   well   as   the   use   of   other   alternative   methods   like
the         computed         accuracy         also        demonstrates         that         also         other        baseline Graph Attention Networks are for a future research work.
models are performing so good as our GNN model. But, in a nutshell, the                                                  Furthermore, tests should be performed directly on EHRs to support
main  advantage  of  this  methodology lies  in  the  identification  of  graph                                     advantage of this methodology using authentic data before using it as an
structures,  which  improves  the  intuitive  interpretation  of  the  relation-                                    interventional  method.  Finally,   once  this  system  is  deployed  it  should
ships in the patient data; this single aspect goes beyond the mere focus                                            offer          an          option          to          the          practitioner          to          decide          if          the          prediction          is
on accuracy and is a relevant criterion for improving transparency in the                                           misleading or not, with a score quantifying the quality of the prediction.
application of artificial intelligence in medicine.
     Despite   these   positive   results,   an   improvement   in   the   ICD   identifi-
cation as well as better methods to reduce potential bias induced by the                                             16
                                                                                                                         Suggestion provided by  an anonymous referee.

J.G. Diaz Ochoa and F.E. Mustafa
This  score  will  be  pivotal  to  gradually  improve  the  quality  of  the  final                                        [7]             Rocheteau E, Tong C, Veliˇckovi´c P, Lane N, Li`o P. Predicting patient outcomes with
deployed “        remembering system”        .                                                                                    graph representation learning. ArXiv210103940 Cs; 2021.
                                                                                                                            [8]             Rasmy L, Xiang Y, Xie Z, Tao C, Zhi D. Med-BERT: pretrained contextualized
                                                                                                                                  embeddings on large-scale structured electronic health records for  disease
Data  availability  and  ethical  issues                                                                                          prediction. NPJ Digit Med 2021;4:86. https://doi.org/10.1038/s41746-021-
                                                                                                                                  00455-y.
     The data used in this study is fully synthetic. All data produced in the                                               [9]             Shang J, Ma T,  Xiao C, Sun J. Pre-training of  graph augmented transformers for
                                                                                                                                  medication recommendation. ArXiv190600346 Cs; 2019.
present study are available upon reasonable request.                                                                       [10]             Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy side effects with graph
                                                                                                                                  convolutional networks. 2018.
CRediT  authorship  contribution  statement                                                                                [11]             Gallagher T, Dube K, Mclachlan S. Ethical Issues in Secondary Use of  Personal
                                                                                                                                  Health Information. URL IEEE Future Dir Technol Policy Ethics 2018. https://cmte.
                                                                                                                                  ieee.org/futuredirections/tech-policy-ethics/2018articles/ethical-issues-in-second
     JGDO       and       FM       had       conceived       the       idea;       JGDO       implemented       and               ary-use-of-personal-health-information/; 2018; 2018.
performed  the  data  synthetization;  FM  implemented  the  GNNs  routine                                                 [12]             Phillips M, Dove ES, Knoppers BM. Criminal  prohibition of wrongful re-
                                                                                                                                  identification: legal solution or minefield for big data? J Bioethical Inq 2017;14:
and wrote the python notebook; JGDO performed further modifications                                                               527–        39. https://doi.org/10.1007/s11673-017-9806-9.
on   the   notebook,   created   the   artwork   and   wrote   the   first   draft   of   the                              [13]             Chen RJ, Lu MY, Chen TY, Williamson DFK, Mahmood F. Synthetic data in machine
article. FM and JGDO wrote the final version of the article together.                                                             learning for medicine and healthcare. Nat Biomed Eng 2021;5:493–        7. https://doi.
                                                                                                                                  org/10.1038/s41551-021-00751-8.
                                                                                                                           [14]             Nowok B, Raab GM, Dibben C. Synthpop: bespoke creation of synthetic data in R.
Declaration  of competing  interest                                                                                               J Stat Softw 2016;74:1–        26. https://doi.org/10.18637/jss.v074.i11.
                                                                                                                           [15]             Zhu Y, Du Y, Wang  Y, Xu Y, Zhang J, Liu Q, Wu  S. A survey on deep graph
     We declare that there is no competing interest.                                                                              generation: methods and applications. ArXiv220306714 Cs Q-Bio; 2022.
                                                                                                                           [16]             Tong C, Rocheteau E, Veliˇckovi´c P, Lane N, Li`o P. Predicting patient outcomes with
                                                                                                                                  graph representation learning. URL. In: Shaban-Nejad A, Michalowski M, Bianco S,
Acknowledgments                                                                                                                   editors. AI for  disease surveillance and pandemic intelligence: intelligent disease
                                                                                                                                  detection in  action. Cham: Springer International Publishing; 2022. p. 281–        93.
                                                                                                                           [17]             Schrodt J, Dudchenko A, Knaup-Gregori P, Ganzinger M. Graph-representation of
     We are very grateful to Martin Grundman and Thomas Shimper for                                                               patient data: a systematic literature review. J Med Syst 2020;44:86. https://doi.
their        relevant        inputs        and        ideas        in        formulating        the        motivation        and  org/10.1007/s10916-020-1538-4.
applicability     of    this    methodology     in    the    analysis    of    health    records.    I                     [18]             Hamilton WL, Ying R, Leskovec J. Inductive representation learning on large
                                                                                                                                  graphs. ArXiv170602216 Cs Stat; 2017.
would      also     like     to     thank     Felix     Weil     for     his     constant     support     in     this      [19]             Nickel M, Kiela D.  Poincar\’    e embeddings  for learning hierarchical
research,  and  Elena  Ramírez  for  her  critique  and  proof  reading  of  this                                                 representations. ArXiv170508039 Cs Stat; 2017.
manuscript.  We  are  also  deeply  grateful  for  two  anonymous  reviewers                                               [20]             Altman NS. An introduction to kernel and nearest-neighbor nonparametric
                                                                                                                                  regression. Am Stat 1992;46:175–        85. https://doi.org/10.1080/
who have significantly contributed improve this research paper.                                                                   00031305.1992.10475879.
                                                                                                                           [21]             Bronstein MM, Bruna J, LeCun Y, Szlam  A, Vandergheynst P. Geometric  deep
References                                                                                                                        learning:  going beyond euclidean data. IEEE Signal Process Mag 2017;34:18–        42.
                                                                                                                                  https://doi.org/10.1109/MSP.2017.2693418.
                                                                                                                           [22]             Rawashdeh A, Ralescu A. Similarity  measure for social networks   –             a brief survey.
 [1]             Malik H, Fatema N, Alzubi JA.  AI and machine learning paradigms for health                                      2015.
       monitoring system: intelligent data analytics. Springer; 2021.                                                      [23]             Samardzhiev K, Gargett A, Bollegala D.  Learning neural word salience scores.
 [2]             Reddy CK, Aggarwal  CC. Healthcare data analytics. Boca Raton: Apple Academic                                    ArXiv170901186 Cs; 2017.
       Press Inc.; 2015.                                                                                                   [24]             Simonyan K, Vedaldi A, Zisserman A. Deep inside convolutional networks:
 [3]             Diaz Ochoa JG, Weil F. From  personalization to patient centered systems                                         visualising image classification models and saliency maps. ArXiv13126034 Cs;
       toxicology and pharmacology. Comput Toxicol 2019;11:14–        22. https://doi.org/                                        2014.
       10.1016/j.comtox.2019.02.002.                                                                                       [25]             Ochoa JGD, Mustafa FE. A method to improve the reliability of saliency scores
 [4]             Farquhar M. AHRQ quality indicators. In: Hughes RG, editor. Patient Safety and                                   applied to graph neural network models in patient populations. 2022. https://doi.
       Quality: An Evidence-Based Handbook for Nurses. Rockville (MD): Agency for                                                 org/10.1101/2022.04.06.22273515.
       Healthcare  Research and Quality (US); 2008. p.                                                                     [26]             Sugimoto CR. Scientific success by numbers. Nature 2021;593:30–        1. https://doi.
 [5]             Hema N, Justus S. Conceptual  graph representation framework for ICD-10.                                         org/10.1038/d41586-021-01169-7.
       Procedia Comput Sci 2015;50:635–        42. https://doi.org/10.1016/j.                                              [27]             Franceschi L, Niepert M, Pontil M, He X. Learning discrete structures for graph
       procs.2015.04.097.                                                                                                         neural networks. ArXiv190311960 Cs Stat; 2020.
 [6]             Kotwal  S, Webster AC, Cass A, Gallagher M. A review of  linked health data in
       australian nephrology. Nephrology (Carlton) 2016;21:457–        66. https://doi.org/
       10.1111/nep.12721.

