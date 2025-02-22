Received April 20, 2020, accepted May 6, 2020, date of publication May 11, 2020, date of current version June 3, 2020.
Digital Object Identifier 10.1109/ACCESS.2020.2993876
Multiple-Aspect Attentional Graph Neural
Networks for Online Social Network
User Localization
TING ZHONG                     , TIANLIANG WANG                             , JIAHAO WANG, JIN WU, AND FAN ZHOU
SchoolofInformationandSoftwareEngineering,UniversityofElectronicScienceandTechnologyofChina,Chengdu610054,China
Correspondingauthor:JiahaoWang(wangjh@uestc.edu.cn)
ThisworkwassupportedbytheNationalNaturalScienceFoundationofChinaunderGrant61602097andGrant 61472064.
   ABSTRACT            Identifying the geographical locations of online social media users, a.k.a. user geolocation
   (UG), is an essential task for many location-based applications such as advertising, social event detection,
   emergency localization, etc. Due to the unwillingness of revealing privacy information for most users,
   it is challenging to directly locate users with the ground-truth geotags. Recent efforts sidestep this lim-
   itation through retrieving users’ locations by alternatively unifying user generated contents (e.g., texts
   and public proﬁles) and online social relations. Though achieving some progress, previous methods rely
   on the similarity of texts and/or neighboring nodes for user geolocation, which suffers the problems of:
   (1) location-agnostic problem of network representation learning, which largely impedes the performance
   of their prediction accuracy; and (2) lack of interpretability w.r.t. the predicted results that is crucial
   for understanding model behavior and further improving prediction performance. To cope with such
   issues, we proposed a Multiple-aspect Attentional Graph Neural Networks (MAGNN) – a novel GNN
   model unifying the textual contents and interaction network for user geolocation prediction. The attention
   mechanism of MAGNN has the ability to capture multi-aspect information from multiple sources of data,
   which makes MAGNN inductive and easily adapt to few label scenarios. In addition, our model is able
   to provide meaningful explanations on the UG results, which is crucial for practical applications and
   subsequentdecisionmakings.Weconductcomprehensiveevaluationsoverthreereal-worldTwitterdatasets.
   The experimental results verify the effectiveness of the proposed model compared to existing methods and
   shedlightsontheinterpretableusergeolocation.
   INDEXTERMS             Attentionmechanism,usergeolocation,onlinesocialnetwork,graphneuralnetworks.
I. INTRODUCTION                                                                                                          locating OSN users has become a key Internet service for
With  the  popularity  of  online  social  network  (OSN),                                                               manydownstreamapplications,includinglocation-basedtar-
e.g., Twitter, Facebook, Wikipedia and Instagram, unprece-                                                               getedadvertising,emergencylocationidentiﬁcation,ﬂutrend
dented volumes of heterogeneous data have been gener-                                                                    prediction,politicalelection,localevent/placerecommenda-
ated,e.g.,publishedmessagecontents,mentiontagsandfol-                                                                    tion, restricted content delivery following regional policies,
low/followeerelations,whichcouldbeleveragedtogeolocat-                                                                   naturaldisasterresponse,etc[8].
ingOSNusers.Forexample,peoplefromSanFranciscomay                                                                             Sincethesocialmediadataisunstructured,learninguseful
frequently mention ‘‘49ers’’ and ‘‘Warrios’’ and those from                                                              representations for both users and their generated contents
New York City have high probability of tweeting contents                                                                 becomes a key step for geolocation and downstream tasks.
referring to the words ‘‘Knicks’’ and ‘‘Yankees’’. As such,                                                              A plethora of works have been proposed towards structur-
the problem of         user geolocation          (UG) has received a lot of                                              ing heterogeneous data towards better OSN user geoloca-
research attention in the past decade [1]–[7]. Successfully                                                              tion. Early efforts [1], [3], [9]–[13] mainly focus on mining
                                                                                                                         indicative information from user posting contents, such as
    The associate editor coordinating the review of this manuscript and                                                  tweets and microblogs. These approaches rely on indica-
approvingitforpublicationwasTingWang                                .                                                    tive words that can link users to their home locations via
VOLUME 8, 2020                     This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see https://creativecommons.org/licenses/by/4.0/                                                                                                          95223

                                                                        T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
various natural language processing (NLP) techniques, e.g.,                                                           bytheextensiveexperimentalevaluationsinSec.V.Wecon-
topic models and statistic models. For example, TF-IDF                                                                cludethisworkinSec.VI.
(termfrequency–inversedocumentfrequency)[14]isacom-
monly used method to measure the distribution of location                                                              II. RELATEDWORK
words [1], [15]. Besides publishing text, users are usually                                                           In  previous  work  on  geolocating  online  social  networks
involved in OSNs to establish relationship and interact with                                                          (OSN), the models be broadly categorized as three groups
friendstosharework/lifeexperience.Therefore,thelocations                                                              accordingtothetypeofdatausedtomakeprediction.Wenow
of users can be inferred by the clues extracting from their                                                           reviewrelevantworksandpositionourpaperintheliterature.
social networks on the OSN, which has spurred a variety
of network-based approaches [16]–[19]. In general, these                                                               A. CONTENT-BASEDAPPROACHES
methods leverage the user interactions, including followee                                                            User generated contents (UGC) such as textual posts and
relationships and mutual/unidirectional mentions, to learn                                                            photos may be casually attached with real-time locations
users’onlineproximitywithvariousgraphlearningmethods.                                                                 facilitatedbyincreasingpopularityofGPS-equippeddevices.
    While achieving promising performance, previous works                                                             However,thesegeo-taggedtweetsareextremelysparse,e.g.,
fail to tackle two main issues in user geolocation. During                                                            no more than 1% of published tweets are tagged with geo-
usercontentlearning,thepostingcontentsareeithermanually                                                               graphicallocations[25].Aplethoraofworks[1],[3],[9]–[13]
associatedwithindicativelocationwords[1],[9]–[12],orare                                                               have studied the possibility of leveraging UGC for locating
simply embedded into low-dimensional vectors using NLP                                                                users. These methods address the geolocation problem by
techniques such as TF-IDF and doc2vec [5], [6], [20], both                                                            inferringlocationsfromthelocation-relevantwordswithvar-
of which fail to capture users’ writing style especially their                                                        iousclassiﬁcationmodels.Therefore,identifyingmeaningful
preference over meaningful location-related words. Further-                                                           indicative words is an important step towards accurate user
more, existing user geolocation methods, especially those                                                             geolocation, where TF-IDF (term frequency–inverse docu-
deep learning based ones, work as a ‘‘black-box’’ model,                                                              mentfrequency)[14]isawidelyadoptedtextualcontentrep-
fail to provide explanations regarding the model behavior                                                             resentation method in the literature [1], [6], [15], [26], [27].
and   prediction   results.   These   limitations   substantially                                                     For example, inverse location/city frequency has been used
preventpreviousmethodsfrommanysafety-criticalapplica-                                                                 to measure the location words in the content [1], [15] while
tions, e.g., identifying the epidemic propagation and accu-                                                           probabilisticmodelsareusedtocharacterizetheusers’loca-
rate/personalizedadvertising.                                                                                         tiondistributionsw.r.t.theirpublishedUGC,which,however,
    Inspired by the recent success of graph neural networks                                                           requiresextensivemanuallylabeledlocation-relatedwordsto
(GNN)[21],[22]andattentionmechanism[23],[24],wepro-                                                                   achievesatisfactoryresults.
pose a novel GNN-based user geolocation model, called                                                                      Inspired by recent advances in applying deep learning in
Multiple-aspect  Attentional  GNN  (MAGNN),  to  address                                                              natural language processing, a few studies turn to model
aforementioned limitations. It is a multi-view UG model                                                               users’ textual contents with various neural networks based
that captures both linguistic and interactive information for                                                         models in order to learn the tweet representation in an end-
interpretableusergeolocation.Themaincontributionsofthis                                                               to-end manner [4], [5], [20], [28]. Among these methods,
workcanbesummarizedasfollows     :                                                                                    doc2vec[29]andrecurrentneuralnetworks(RNNs)aresim-
    •A novel multi-aspect GNN model for efﬁcient user                                                                 ple yet effective choices for learning vector representation
        generated content and network information fusion. The                                                         of textural contents. For example, Do                     et al.   [4] combine
        proposed model exploits the interactions among tweets                                                         TF-IDF and doc2vec representations of textual information
        words and the relationship information of social net-                                                         to enhance the prediction performance. Miura                           et  al.   [5]
        worksinanend-to-endmanner.                                                                                    use  GRU  [30]  with  attention  mechanism  [31]  to  model
    •By stacking multi-head attention layers, our model is                                                            user tweet content and obtain a timeline representations.
        able to distinguish different aspects of user publishing                                                      Thoughdoc2vecandRNN-basedmethodscanlearnlanguage
        preferenceandthedifferentimportanceofon-lineinter-                                                            efﬁciently  without  manual  location  feature  engineering,
        actions in a dynamic learning way rather than a ﬁxed                                                          a recent study [32] ﬁnd that TF-IDF is consistently superior
        representationinpreviouswork.                                                                                 to doc2vec due to the location-indicative words captured
    •Weconductedextensiveexperimentstoevaluatethepro-                                                                 inTF-IDF.
        posed model on three large-scale real Twitter datasets.
        TheexperimentalresultsdemonstratethatourMAGNN                                                                  B. NETWORK-BASEDMETHODS
        model signiﬁcantly improves the user location predic-                                                         Online social relationships are also important indicators for
        tion accuracy compared with the state-of-the-art base-                                                        usergeolocationunderthehomophilyassumption[16]–[19],
        lineswithexplainableresults.                                                                                  i.e., people prefer to interact with others in nearby areas.
    The  remainder  of  this  paper  is  organized  as  follows.                                                      Backstrom        etal.  [16]examinetherelationshipbetweenusers’
We discuss the related work in Sec.II, and introduce the                                                              geographicalproximityandonlinefriendshipsonFacebook,
problem and provide the necessary background in Sec.III.                                                              and ﬁnd that the likelihood of friendship between any user
ThedetailsofMAGNNwillbeexplainedinSec.IV,followed                                                                     pair drops monotonically as a function of distance. Rather
95224                                                                                                                                                                                                                                                                                          VOLUME 8, 2020

T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
thansolelyrelyingonfriendships,moreandmoreworksuti-                                                                 TABLE1.        Notations.
lizevarioustypesofconnections,suchastheco-mentiontags
andmentionsbetweennon-friends,toconstructclosersocial
interactionsbeyondfriendships[8],[20].Inthisway,similar
interests among users can be retrieved from such implicit
networks to improve geolocation accuracy [28], [33], [34].
Moreover, researchers also identify some noisy interaction
factors that may degrade the prediction performance. For
example,socialinﬂuenceofcelebritiesisadistractingfactor
that may confuse the prediction and thus is removed from
thebuiltusernetwork[28],[35].Thoughexplicitlymodeling
location dependency between social connected users, some
challengesoftheseworkshavenotbeenproperlyaddressed,
e.g., sparsity of geo-tagged users and inaccurate label prop-
agation, and, most important, the locations of friends are
usually contradicting with each other, which hinder these
approachesfrompracticalapplications.
C. MULTI-VIEWMODELS
Recenteffortshaveleverageddeepgraphlearningmethodsto                                                                alreadyinTwitter,andwordsstartingwith‘‘#’’are                          hashtags
modeluserinteractionnetworkbyfusingusergeneratedcon-                                                                usedtomentionatopic.
tents and various meta-data, such as user proﬁles, tweeting                                                              Deﬁnition1(TweetContent):                   For each user, we collect
time and user timezone. For example, MENET [4] exploits                                                             his/her tweets as linguistic content, including both tweet
node2vec [36] to learn user representations, combined with                                                          messages by himself and retweets forwarding other users’
text representation learned by doc2vec, for predicting users’                                                       postings. Following previous works [4]–[6], [26], we ﬁlter
locations.Anotherwork[6]employsGCNs[21]forlearning                                                                  out the photos and symbols for each user. We denote the
network structures with the graph convolution and pooling                                                           learned content embedding vector of user                        v  as  x v, and the
operations, which has achieved the state-of-the-art geolo-                                                          tweetcontentforallusersas             X .
cation performance. A recent work [32] investigate several                                                               Inadditiontopostingtext,weconstructthementiongraph
graph embedding methods and found that NetMF [37] per-                                                              torepresentthesocialrelationshipsamongusersbyextracting
forms better than node2vec and GraphSAGE [38] on user                                                               mention(@-somebody)informationfromtweetmessages.
geolocation task, but does not show superior performance                                                                 Deﬁnition2(MentionNetwork):                     The mention network is
thanGCN-basedmodels[6],[32].                                                                                        deﬁned as     G=(V,E), where    V  is a set of all users (nodes)
    Itisworthwhiletonotethattherearesomeworksmaking                                                                 and  E  is a set of edges between nodes. Each node                        v∈V  is
use of various meta-data (e.g., self-declared location in pro-                                                      associatedwithatweetcontentvector                   x v asitsfeature.
ﬁle and timezone information) for improving the prediction                                                               We focus on predicting the ‘‘home’’ location of users [8],
performance. For example, user timezone, as well as UTC                                                             i.e., the location that a user most probably resides in. Since
offset and country noun, have been used for user geoloca-                                                           each user location is described by a pair of numbers (lon-
tion[4],[5],[20],[26],[39].Whilesuchauxiliaryinformation                                                            gitude and latitude), we convert this problem to the classiﬁ-
is a strong indicator for regularizing the locations the model                                                      cation problem by dividing the surface of earth into closed
predicted, a majority of users are not willing to open these                                                        and non overlapping clusters using                    k-d trees. Each user is
privacy information, which are sometimes camouﬂaged or                                                              thereforetaggedwithone(andonlyone)labelindicatingthe
posted casually. We further note that there is another line                                                         clusterhe/shebelongsto.Eachlabelisencodedasaone-hot
of efforts [7], [17], [40]–[42] studying the Twitter message                                                        vector,andwedenotealllabels(clusters)as                    Y∈R n×c,where
geolocationproblemwhichtrytoidentifythetweetingloca-                                                                n isthenumberofusers,and              c isthenumberofclusters.Now,
tions rather than the Twitter user location discussed in this                                                       weformallydeﬁnetheusergeolocationproblemas:
work.                                                                                                                    Deﬁnition3(UserGeolocationPrediction):                          Givenalluser
                                                                                                                    tweet content and mention graph               G , as well as partially
III. PRELIMINARIES                                                                                                  labeledusers,weareinterestedinidentifyingthegeographi-
Inthissection,weintroducetheproblemdeﬁnitionaswellas                                                                callocationsofunlabeledusers.
basicnotationsusedthroughoutthispaper,cf.Table1.
    Inthiswork,weconsidertheproblemoflocatingtheusers                                                                IV. METHODOLOGY:MAGNN
inTwitter,wherea          tweet   isashorttext(within140characters)                                                 The proposed MAGNN model is shown in Figure1. It con-
with some other content, e.g., photos and emojis. The extra                                                         sists of three main components, i.e., attention-based content
information are usually associated with a tweet describing                                                          learning,graphneuralnetworksbasedinteractrelationlearn-
speciﬁcmeanings,e.g.,‘‘@’’usedto                    mention      peoplewhoare                                       ingandgeolocationpredictor.First,multi-headself-attention
VOLUME 8, 2020                                                                                                                                                                                                                                                        95225

                                                                        T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
FIGURE1.        The framework of MAGNN fusing the content features and networks information with multi-head attention to predict the home location of
Twitter users.
is utilized to learn user posting content embedding from                                                                   Inspiredbyrecentadvancesinnaturallanguagerepresenta-
tweets content. Secondly, the user tweet content embedding                                                             tionlearning[23],[24],weproposetoutilizemulti-headself-
and the topological features of mention network are fused                                                              attentiontodealwithusertweetscontentwhichcouldcapture
with attention mechanism on graph. After that, the fully                                                               plentiful syntactic features w.r.t. user posting behaviors, and
connectedlayerwithsoftmaxisusedtopredictthelocations                                                                   moreimportantly,themoresigniﬁcantlocationrelatedinfor-
forusers.                                                                                                              mation. First, we use multi-head self-attention to learn user
                                                                                                                       tweetsentenceembeddingbypayingattentiontoinformative
                                                                                                                       words and building correlation with other relevant words.
A. LEARNINGUSERCONTENTWITHMULTI-HEAD                                                                                   Then,weapplyalearnablematrixtransformationtosentence
ATTENTION                                                                                                              embedding so as to form the user tweet content representa-
To   represent   user   generated   text,   TF-IDF   [14]   and                                                        tion which will be considered as user tweet content features
doc2vec  [29]  are  two  widely  used  techniques  in  previ-                                                          associatedwiththenodeinthementionnetwork.
ous works [4], [6], [26], [32]. TF-IDF is a relative fre-                                                                  Speciﬁcally,  we  ﬁrst  tokenize  the  tweet  sentence  and
quency  approach  that  captures  linguistic  information  at                                                          convert   the   sequence   of   words   into   a   sequence   of
the word-level, while doc2vec embeds user content into a                                                               low-dimensional embedding vectors, of which the                              i-th word
low-dimensional latent space. However, in some situation,                                                              is denoted as       ei (ei∈R 1×d e). Next, the relative importance
we need to capture the meaning behind the language to                                                                  scoreof     j-thwordto      i-thwordunderaspeciﬁcattentionhead
achieve good performance. For example, the tweet ‘‘Too                                                                 w  iscomputedwithsoftmaxfunctionoverthetweetsentence       :
unlucky,I’llnevercometoAlaskaagain’’inFigure1implies                                                                                                                                                    ),               (1)
that this user had negative impression on Alaska according                                                                                    αwij=softmax ((ei2  wQ )(ej2  wK )T√
to the emotion conveyed by her tweet. Meanwhile, it is very                                                                                                                              d k
likelythatsheisjustatouristoronabusinesstriptoAlaska,
i.e., the user probably does not reside in Alaska. However,                                                            where   2  wQ∈R d e×d k  and  2  wK∈R d e×d k  are  Query     and   Key
theseinformationcan’tbecapturedbythetraditionalmethods                                                                 parameter matrices [23], respectively, and                       d k  is its column
suchasTF-IDFanddoc2vec.                                                                                                number.Inaddition,thesoftmaxoperationanddivisionwith
    Inaddition,tweetssentbythesameuseralwayshavesome                                                                   thesquareroot        d k enablethescoretohavemorestablegradi-
irrelevantinformationthatwouldbeconfoundingfactorsfor                                                                  ents. Then, we update the               i-th word’s representation in head
geolocation prediction, e.g., users usually (re)tweet some                                                             w  throughcombiningfeaturesofallrelevantwordsguidedby
information having nothing to do with indicative location                                                              importancescore    αwij:
names.Thus,differenttweetssentbythesameusermayalso
have different importance in representing the user geloca-                                                                                                            m∑         (ej2  wV    ),                        (2)
tions.Forexample,theﬁrsttwotweetsaremoreinformative                                                                                                      h wi=             αwij
thanthelastoneinFigure1.                                                                                                                                             j=1
95226                                                                                                                                                                                                                                                                                          VOLUME 8, 2020

T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
where   2  wV∈R d e×d v is Value    parametermatrix[23]and               d v is                               reviews.Forexample,GAT[22]introducesattentionmecha-
its column number,            m  represents the length of the sentence,                                       nismintheprocessofGNNlearning,whereanodeinvolves
i.e., the number of word token in the sentence. Furthermore,                                                  mostrelevantinformationfromitsneighborhoodsandupdate
the different heads are supposed to focus on different words                                                  its own features with the learned attention weights, which
and learn different aspects of the sentence. And the new                                                      enables the model to focus on the most informative features
representation of         i-th word is calculated by collecting com-                                          while alleviating noise signals during message passing in
binatorialfeatureslearnedineachheadas    :                                                                    the network. Here we extend GAT with multi-head attention
            ˆei=(h 1i⊕h 2i⊕···⊕h wi⊕···⊕h Wi                                 )2  O,       (3)                 to learn the structural representation while propagating the
                                                                                                              contentfeaturesinnetwork.
where ⊕representstheconcatenationoperator,                    W   isthenum-                                       Speciﬁcally, we ﬁrst compute the relevant coefﬁcients
ber of total heads, and       2  O∈R Wd   v×ˆd e is the output weight                                         between a pair of nodes with multi-head attention. Namely,
matrix.  With  such  attentional  operation,  the  embedding                                                  the correlations between node                 u and node      v in the   r-th head
of i-thword     eiisupdatedinto ˆei,whichcapturesmulti-aspects                                                canbecalculatedas(          r> 0):(
meanings guided by multi-heads attention among all words.                                                                             ervu=        σ([x vW   r∥x uW   r]a T)),                 (7)
Andtheﬁnalembeddingofthissentenceisthesummationof
thecontextualwordrepresentations,formulatedas      :                                                          where     W   r∈R d×d′is a linear transformation matrix of the
                                                 m∑                                                           r-th head which maps input features into high-level repre-
                                        s=            ˆei,                                (4)                 sentations, and  ||denotes the concatenation operation. Here
                                                i=1                                                           we use a feedforward neural network with parameters                              a∈
where     s∈R 1×ˆd e. In order to select and learn more infor-                                                R 1×2d′as the attention layer and     σ(·) as the non-linear
mative information from multiple tweet sentences sent by                                                      activation function (LeakyReLU(    ·) in our implementation).
the same user automatically, we design an additive linear                                                     Inordertomakethecorrelationcomputationstable,softmax
transformation network to generate the tweet content repre-                                                   isappliedoverallnodesin          N  v:
sentationforusers.Theformulationisshownasfollows       :                                                                                   βrvu=      exp(  ervu )∑
               x=(s1⊕s2⊕···⊕st⊕···⊕sT)2  S,          (5)                                                                                                    q∈N  v exp(  ervq ),                      (8)
where      T   is the number of tweets of each user, which is                                                 where the coefﬁcient     βrv∗is expected to capture the most
ﬁxed in our datasets.       2  S∈R Tˆd e×d  is the learnable matrix                                           relevant features while dynamically ﬁltering out the useless
transformingthemultiplesentenceembeddingsintoasingle                                                          featuresfornode         v.Subsequently,linearcombinationisused
vector. The tweet content representations of all users are                                                    to fuse the neighboring features with the built coefﬁcient in
denoted by       X   (X∈R n×d ), which is the input features of                                               r-thheadas :
networkslearninginMAGNN.                                                                                                                          frv=∑            βrvu x u.                            (9)
                                                                                                                                                           u∈N  v
B. MULTI-ASPECTINFORMATIONFUSIONUSINGGNN                                                                          Next,wecalculatethenewrepresentationofeachnodeby
GNN  is  a  powerful  tool  for  graph  representation  learn-                                                averagingthefeaturesofallmulti-headsthroughanon-linear
ing, which has received increasing attention over the past                                                    transformation :
years [21], [22], [43], [44]. A GNN model consists of a                                                                                                           (1       R∑
stack of neural network layers, where each layer aggregates                                                                         x′v=LeakyReLU                              frvW   r),              (10)
neighborhoodinformationaroundeachnodeandthenpasses                                                                                                                   R   r=1
the aggregated message into the next layer. Given a network                                                   where     R  is the number of attention heads and                   W   r  is the
G=(V,E)andtheinitialfeatures          x v ofcorrespondingnode             v,                                  linear transformation weight matrix of                     r-th head. The new
ageneralGNNarchitectureupdatingthenoderepresentation                                                          representation of all users is represented as                    X′(X′∈R n×d′)
in k-th( k> 0)layercanbeimplementedas[38]:(          ({                          }))                          whichwillbefedintothe             geolocationpredictor            forpredicting
        x (k)v=fθ2merge         x (k−1)                  x (k−1)   ⏐⏐u∈N  v              ,    (6)             the ﬁnal results. Note that we mask the labels of validation
                                   v  ,fθ1aggr             u
where  θ1  and θ2  are  trainable  parameters  optimized  via                                                 andtestingsamplesduringtraining,i.e.,thelabelsofthedata
stochastic gradient descent, and            N  v represents the neighbor-                                     in validation and testing set are invisible when learning user
hoods of node          v. fθ1aggr   aggregates the features from neigh-                                       representation.
bors with various operations (e.g., Mean and Pooling) while                                                    C. GEOLOCATIONPREDICTOR
fθ2merge     merges node’s representations from the                       k−1 step                            The  objective  of           geolocation  predictor              is  to  predict  the
and the aggregated features of neighbors. The learned node                                                    highest probability of a location the user belongs to. Here,
embeddings can be used for downstream tasks such as link                                                      we adopt a multilayer perceptron layer (MLP) to make the
prediction,node/graphclassiﬁcation.                                                                           predictionsbasedonthelearneduserrepresentations                          X′:
    There are many variants of GNNs which are used to deal
with graph structure data, cf. [43], [44] for comprehensive                                                                               Y′=softmax (MLP(    X′)),                   (11)
VOLUME 8, 2020                                                                                                                                                                                                                                                        95227

                                                                     T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
where     Y′∈R n×c is predictions for all users. Here we adopt                                                   TABLE2.        Statistics of datasets.
crossentropyasthelossfunction   :
                                            n∑     c∑
                              ℓ=−                       yijlog  y′ij,                    (12)
                                          i=1    j=1
where     yij denotestheprobabilitythatthe                i-thuserbelongsto
the  j-thcluster.Duringtraining,Adam[45]isadoptedasthe
stochasticgradientdescentoptimizer.                                                                              onlyintroducestheextracost           O (md   2)inEq.(3)and       O (md   2)
   Algorithm1         TrainingAlgorithmofMAGNN                                                                   in Eq. (5). Therefore, the complexity for content learning
     Input   :Tweetcontent,Mentionnetwork               G=(V,E).                                                 is O (n(m  2d+md   2)), where      n  is the number of all users.
     Output     :Trainedmodel.                                                                                   Asfornetworklearning,theoperationofcalculatingattention
     /* Content  Learning                                 */                                                     score (cf. Eq. (7) and Eq. (8)) and output features (i.e.,                            frv)
                                                                                                                 of each head can be paralleled across all nodes and the time
 1  for  useridi =1 ton  do                                                                                      complexity of attentional GNN learning with one attention
 2       for  sentenceidj =1 toT   do                                                                            head is   O (ndd′+|E|d′), where |E|is the number of edges
 3              Initialize    j-thsentencetokens’embeddings;                                                     in the mention network. By contrast, the time complexity of
 4             foreach     token    do                                                                           GCN4Geo is       O (LA   0  F+LNF     2), where     L  is the number of
 5                   Calculatenewtokenembedding     ˆe via                                                       layers,   N  isthenumberofusers,            A 0 representsthenumberof
                         Eq.(1),(2)and(3);                                                                       non-zerosintheadjacencymatrixofmentionnetworkand                                  F
 6             end                                                                                               isthedimensionoffeatures.
 7              Computethe         j-thsentenceembedding              sj via
                  Eq.(4);                                                                                        V. EXPERIMENTS
 8       end                                                                                                     In this section, we conduct experiments on three real-world
 9        Computethe         i-thusercontentrepresentation               x i via                                 datasetstoevaluateourmodelagainstbaselines.Speciﬁcally,
           Eq.(5);                                                                                               weaimtoanswerthefollowingresearchquestions      :
10  end
11   Concatenatecontentrepresentationofallusersinto                                                                  •Q1.    How  does  MAGNN  perform  compared  to  the
     matrix    X ;                                                                                                       state-of-the-art  approaches/baselines  on  geolocation
     /* Network  Learning                                 */                                                             prediction?
12  foreach      userv ∈V  do                                                                                        •Q2.   Howdoeseachcomponent(contextembeddingand
13        Get   v’sneighborhoods        N  v in G ;                                                                      network embedding) contributes to the performance of
14        for  headidr =1 toR   do                                                                                       MAGNN?
15              Computeattentionscores     βrv∗among    N  v via                                                     •Q3.   Howdothekeyhyper-parametersaffectMAGNN’s
                  Eq.(7)and(8);                                                                                          performance?
16              Compute       v’srepresentation        frv with βrv∗via                                              •Q4.   Howcanweexplainthepredictionresultsmadeby
                  Eq.(9);                                                                                                ourMAGNN?
17        end
18        Computefusionrepresentation                 x′v ofuser    v via                                        A. DATASET
           Eq.(10);                                                                                              Toevaluatetheperformanceofourmodel,weconductexper-
19  end                                                                                                          iments on three real-world Twitter datasets which have been
20   Concatenatefusionrepresentationofallusersinto                                                               widely used forevaluating the user geolocation models.The
     matrix    X′;                                                                                               datasets are listed below and their statistics are summarized
     /*  Geolocation  Predictor                       */                                                         inTable2.
21   Computepredictions            Y′with   X′viaEq.(11);                                                            •GeoText        [46] is a Twitter dataset consisting of 9.5K
22   CalculatelosswithEq.(12);                                                                                           usersfrom49statesandWashingtonD.C.inU.S.,which
23   UpdatemodelparameterswithAdam.                                                                                      isoriginallycompiledbytheauthorsin[46].Thedataset
                                                                                                                         hasalreadybeendividedintothetraining,development
                                                                                                                         andtestingsetwith5,685,1,895and1,895users,respec-
 D. COMPLEXITY                                                                                                           tively.
Algorithm1outlines the training procedure of MAGNN.                                                                  •Twitter-US        [11] is a larger dataset consisting of 449K
Inordertosimplifytheformula,weassume                        d e=d k=d v=                                                 users from the U.S., which was created by the authors
d e=d  for content learning. Thus, the complexity of singleˆ                                                             in[11].ThisdatasetisalsoreferredtoasUTGeo2011in
self-attention is     O (m  2d ), where      m   is the sequence length.                                                 some  papers  [4],  [11].  Following  previous  works,
Since multiple heads can be computed in parallel, the com-                                                               10Kusersareheldoutforvalidationand10Kusersleft
plexity of calculating the content presentation for one user                                                             fortesting.
95228                                                                                                                                                                                                                                                                                          VOLUME 8, 2020

T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
    •Twitter-World           [1] is a much larger dataset released by                                                   •Mean     predictionerror,measuredinkilometres,givesthe
        the authors of [1] and had been rebuilt by the authors                                                               averagederrorbetweenthepredictedclustercentersand
        of [6]. This dataset consists 1.3M users from different                                                              theground-truthgeolocationsforalltestingsamples.
        countries in the world, of which 10K users are kept as                                                          •Median      predictionerrorreportsthemedianvalueofthe
        modelevaluationwhileanother10Kusersareemployed                                                                       predictederrorsforalltestingsamples.
        fortesting.Theprimarylocationsofusersaremappedto                                                                •Acc@161         measures the accuracy of the classiﬁcation.
        thegeographiccenterofthecityfromwherethemajority                                                                     Namely, if the distance between the predicted cluster
        oftheirtweetsareposted.                                                                                              centerandground-truthiswithin161km(or100miles),
                                                                                                                             theresultwillbeconsideredasacorrectprediction.
1) MENTION NETWORK CONSTRUCTION                                                                                         Note that the distance of coordinates is computed using
Weconstructtheinteractionnetwork               G  ofusersutilizingthe                                               the Haversine formula [48]. The lower values of Mean and
mention information extracted from tweets following pre-                                                            Medianerrorindicateabetterprediction.Conversely,achiev-
vious works. For each pair of users, there is an undirected                                                         inghighervalueofAcc@161isdesirable.
edge between them if one mentions the other, or both of
them mention someone else. Additionally, the user who has                                                           C. BASELINES
too many edges will be considered as ‘‘celebrity’’ and is                                                           We compare MAGNN with the following user geolocation
removed to alleviate the negative factor of social inﬂuence                                                         models:
following [6], [28] – the ‘‘celebrity threshold’’ is 5, 15 and                                                          Text-based:
5forGeoText,Twitter-USandTwitter-World,respectively.
                                                                                                                        •HierLR        [2]   is   a   text-based   geolocation   model,
2) DATA PRE-PROCESSING                                                                                                       which adopts a grid representation of locations and
Weﬁrstcollect50tweets(sentences)foreveryuserrandomly.                                                                        resorts  to  hierarchical  classiﬁcation  using  logistic
For each tweet, we tokenize the content and remove stop                                                                      regression(LR).
words as well as symbols using the natural language toolkit                                                             •MLP4Geo         [20]isatext-basedmodelwhichusesdialec-
nltk  [47].Furthermore,word2vec                      1 willbeutilizedtogenerate                                              taltermstoimprovethepredictionperformance.Asim-
theinitialembeddingforeachtoken.                                                                                             pleMLPnetworkisusedtopredictthelocations.
                                                                                                                        •DocSim       [11] uses a method of matching the similarity
3) LABEL GENERATION                                                                                                          (KLdivergence)ofthesubjectdocumentforprediction.
Weuse      k-dtreetodividethecoordinatesintoclusters,which                                                              •LocWords        [1] is a text-based model which uses several
are then used as labels of user locations. In order to avoid                                                                 methodstoﬁndthelocationindicativewords(LIWs)for
sample imbalance, we set the ‘‘bucket-size’’ – which is the                                                                  prediction.
maximumcapacitylimitofusersinonecluster–to50,2400,                                                                      •MixNet     [27]isatext-basedmodelwhichappliesmixture
2400 for GeoText, Twitter-US and Twitter-World, respec-                                                                      densitynetwork(MDN)forembeddingcoordinatesina
tivelyassuggestedby[6],[28].                                                                                                 continuousvectorspacewithsharedparameters.
                                                                                                                        Network-based:
4) EXPERIMENTAL SETTINGS                                                                                                •MADCEL          [26]isanetwork-basedmodel,whichapplies
All  experiments  are  performed  on  a  machine  with  two                                                                  Modiﬁed Adsorption with celebrity removal. Only the
GeForce GTX 1080Ti graphics cards and 128GB of RAM.                                                                          results of weighted network will be reported since it
All neural networks based models are trained with mini-                                                                      performsbetterthanbinarynetwork.
batch based Adam [45] optimizer with exponential decay.                                                                 •GCN-LP        [6] is a GCN-based model and it is similar to
For MAGNN, we train the model using activation func-                                                                         labelpropagation.Itperformstheconvolutionoperation
tionReLU( ·),LeakyReLU(  ·),Sigmod( ·)forcontentlearning,                                                                    on network for prediction and user’ features are repre-
network learning and the predictor, respectively. Moreover,                                                                  sentedbyone-hotencodingofit’sneighbours.
thelearningrateofourmodelisinitializedwith0.001which                                                                    Multiview-based:
is decayed with a rate of 0.0005. In addition, early stopping
isadoptedintrainingMAGNNifthevalidationlossdoesnot                                                                      •MADCEL-LR            [26]combinesthetextandnetworkinfor-
decrease for 20 consecutive epochs. Furthermore, the num-                                                                    mationandusesLRforlocationprediction.
ber of graph attention heads is determined by grid search                                                               •MENET        [4]concatenatesthefeaturesfromtextualinfor-
on{4,8,16,32,64}fordifferentdatasets.                                                                                        mation (    tf-idf  [49],   doc2vec      [50]), user interaction net-
                                                                                                                             work (    node2vec        [36]) and metadata (           timestamp      ) and
B. METRICS                                                                                                                   use fully connected networks for location prediction.
Weevaluateallapproachesusingthefollowingthreemetrics                                                                         For a fair comparison, we only use text and network
that are commonly used for user geolocation performance                                                                      informationinMENET.
evaluation :                                                                                                            •GeoAtt      [5] models the textual context with RNN and
                                                                                                                             attention mechanism. We remove the location descrip-
   1https://radimrehurek.com/gensim/models/word2vec.html                                                                     tionsinGeoAttforfaircomparisons.
VOLUME 8, 2020                                                                                                                                                                                                                                                        95229

                                                                         T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
TABLE3.        Performance comparison among different algorithms on three datasets.
    •DCCA        [6]  is  a  multiview  geolocation  model  using                                                        usage–e.g.,therearemanyofﬁcialaccountsassociatingwith
         Twitter text and network information and measures the                                                           various companies and NGOs. Also, a large number of per-
         canonicalcorrelationforlocationprediction.                                                                      sonalaccountsuseTwitterforinformationdisseminationand
    •GCN4Geo         [6]isaGCN-basedmodelthatusesbothtext                                                                knowledge sharing instead of building social relationships.
         and network context for geolocation prediction, where                                                           Inbothcases,homophilyassumptionisnotheldanymore.
         layer-wisegatesareemployedforcontrollingtheneigh-                                                                   Second, the performance of deep learning-based multi-
         borhood smoothing to alleviate the noisy propagation                                                            view  models,  including  MENET,  GeoAtt,  DCCA  and
         inGCNs.                                                                                                         GCN4Geo, is very similar if both text and network features
    •KB-emb        [40]  proposed  a  prediction  method  based                                                          areused.Surprisingly,theirperformanceareveryclosetothe
         on  entity  linking  as  well  as  the  embedding  of                                                           models using simple classiﬁcation methods, e.g., the LR in
         knowledge-base.                                                                                                 MADCEL. This result implicates that meaningful features
    •GausMix        [7] is constructed using a series of Gaussian                                                        are more important than complicated models in the user
         mixture models. It exploits both text and network fea-                                                          geolocation prediction task. This observation can be further
         tures and weights the features according to their geo-                                                          proveninpreviouswork[4],[5]thatincorporatesmorestrong
         graphicscope.                                                                                                   indicators such as timezone of users and description in the
                                                                                                                         location ﬁeld – however, improving MAGNN with more
D. Q1:OVERALLPERFORMANCECOMPARISON                                                                                       features,e.g.,self-declaredlocationsandtimezone,isbeyond
Theoverallperformanceofallmethodsacrossthreedatasets                                                                     the scope of this work and is left for our future work. Fur-
arepresentedinTable3,fromwhichwehavefollowingmajor                                                                       thermore, previous multi-view models fail to improve the
observations.                                                                                                            model performance largely due to the ignorance of node
    First,relyingonlyontweetcontent[2],[20]isnotenough                                                                   importancewhenmodelingtheuserinteractionnetwork.For
for  user  geolocation  prediction,  which  usually  exhibits                                                            example,MENETusesnode2vectoembedthenetworkwhile
extremely high prediction bias. This result is intuitive since                                                           GCN4Geo directly leverages GCNs for modeling the user
neither indicative words [1], [11] nor topic-based language                                                              interactions. However, both node2vec and GCNs do not dis-
models[27],[46]canﬁlteroutnoisysignalsfromusertweet-                                                                     criminate the relative inﬂuence of nodes when aggregating
ing content. For example, users usually publish short texts                                                              thelocalstructuralinformation.Forexample,iftwousersare
containing acronyms and misspellings which are difﬁcult to                                                               topologically the same, they would be located to the same
be identiﬁed. Moreover, estimating spatial word distribution                                                             region(withoutconsideringtheirtweetcontent)eventheyare
often confronts sparsity problem, i.e., some location words                                                              residingingeographicallydifferentlocations.
w.r.t.lesspopulatedlocationsareunobservedduringtraining,                                                                     Third, our MAGNN consistently outperforms the base-
whichfurtherobfuscatethegeolocationmodels.Ontheother                                                                     lines on all metrics, which proves the effect of addressing
hand, user interaction network plays a key role in predicting                                                            the user geolocation problem with the proposed multi-head
the home locations. However, we also cannot rely only on                                                                 attention based neural networks. This is mainly because the
user networks for accurate user geolocation. This is mainly                                                              multi-heads attention could capture multiple aspects mean-
because many accounts use Twitter, as well as other OSN                                                                  ing for dynamic features aggregation, while ﬁltering out the
platforms such as Facebook and Instagram, for the purpose                                                                noise of content and structure information to reduce predic-
of propagating information like advertising and commercial                                                               tion bias. Compared to RNN-based attention models in [5],
95230                                                                                                                                                                                                                                                                                          VOLUME 8, 2020

T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
                                                                                                                        TABLE5.        Performance of MAGNN with different number of attention
                                                                                                                        heads in network learning (i.e.,                    R ).
FIGURE2.        The results on Macro-Recall (a) and Macro-F1 (b). We omit
other baselines since GCN4Geo usually performs the best among the
baselines.
                                                                                                                         F. Q3:PARAMETERSENSITIVITY
TABLE4.        Performance of MAGNN variants on GeoText and Twitter-US.                                                 Asthemulti-headattentionmechanismisusedinourmodel,
                                                                                                                        differentheadsaresupposedtocapturemulti-aspectfeatures
                                                                                                                        and make our model more stable. In this section, we ana-
                                                                                                                        lyze the performance of MAGNN w.r.t. the head number.
                                                                                                                        Since network plays more important role, we ﬁxed an opti-
                                                                                                                        mal number of heads in content learning and investigate the
                                                                                                                        performance of MAGNN by varying the head number                                    R  in
                                                                                                                        network learning. In particular, we use 8 attention heads to
MAGNN can capture long-range dependencies in textual                                                                    embedthecontentofeachuserintoa512-dimensionalvector,
informationandadaptivelyadjustinteractionlearning.                                                                      andtheninvestigatethe inﬂuenceofthenumberofattention
    Finally,  the  Macro-Recall  and  Macro-F1  results  of                                                             heads    R . Tables5shows the results carried on GeoText and
MAGNN  and  GCN4Geo  are  shown  in  Figure2 –  we                                                                      Twitter-US. On GeoText, the more heads the better perfor-
omitted other methods because GCN4Geo usually performs                                                                  mance ofour model,when                 R≤32. Thisresult suggeststhat
best among the baselines. Clearly, MAGNN outperforms                                                                    increasing       R  is a direct optimization method for a smaller
GCN4Geo slightly due to its ability of distinguishing the                                                               dataset.However,furtherincreasing                    R  (e.g.,greaterthan32)
importance of neighboring nodes when aggregating features                                                               doesnotimplyhigherperformance,whichmeansoneshould
from social friends. We note that the number of training                                                                carefully tune this hyperparameters to balance the effective-
samples in different clusters are extremely imbalanced, e.g.,                                                           nessvs.efﬁciency–thecomputationalcostsurgesconsider-
peoplegenerallyliveindenselypopularcities(e.g.,NYCity                                                                   ably with the value of            R . This can be further proven by the
andLosAnglesinGeoTextData)whileonlyfewuserslivein                                                                       results on Twitter-US, a signiﬁcantly larger dataset, where a
theruralareas.Therefore,howtoaddresstheclassimbalance                                                                   smaller value of         R  is enough for our model to achieve best
issue inherent in user geolocation is a challenging problem                                                             results.
requiring further examinations, which is left as our future
work.                                                                                                                    G. Q4:QUALITATIVEANALYSIS
                                                                                                                        In this section, we provide qualitative interpretation of the
E. Q2:ABLATIONSTUDY                                                                                                     resultsmadebyMAGNNfromthelatentspace–thelearned
To investigate the effect of different components in our                                                                latentspacewhichreﬂectshowexpressiveanddistinctrepre-
model, we implement two variants of MAGNN, including:                                                                   sentationourmodelcanlearn.
(1)  MAGNN-content              , which only utilizes user content fea-                                                      In this section, we provide qualitative interpretation of
tures for prediction; and (2)             MAGNN-network                which only                                       the results made by MAGNN. We randomly select four
relies on interaction network for user geolocation. The per-                                                            cluster from GeoText and their corresponding users and use
formance of two variants, as well as MAGNN, is shown                                                                    t-SNE [51] to map the learned latent representation into 2D
in Table4. The result suggests that network information                                                                 space. Figure3illustrates the results of MLP4Geo and our
playsmoreimportantrolethancontentfeatures,whichisalso                                                                   MAGNN, from which we can easily observe the clustering
observed in recent experimental comparisons [32]. It also                                                               effect in the latent space learned by MAGNN. MLP4Geo,
points out the promising ways of improving geolocation                                                                  incontrast,isaplainmodelwhichsimplyconcatenatescon-
performance in future studies, i.e., focusing more on users’                                                            tentfeatures      X  andnetworkadjacentmatrix                A  andthenfeed
interactions rather than their publishing contents. Another                                                             themtoMLPsforgeolocationprediction.Therefore,itisdif-
potential way of further improving MAGNN is to explore                                                                  ﬁcultforthissimplemodeltocapturenon-linearinteractions
more auxiliary features of user (e.g., user proﬁle, time-                                                               amongsamplesfromdifferentclasses,and,moreimportantly,
zone)  which  have  been  proven  in  previous  work  [4]  to                                                           todiscriminatetheusersusinguniformlyscatteredrepresen-
be strong indicators for better regularizing the geolocation                                                            tations. This result also explains the performance gain made
results.                                                                                                                byourGNNbasedmodelwhichaggregatesimportantsignals
VOLUME 8, 2020                                                                                                                                                                                                                                                        95231

                                                                                    T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
                                                                                                                                           VI. CONCLUSIONREMARKS
                                                                                                                                           In this work, we presented a new social user geolocation
                                                                                                                                           framework, which is built upon the twitter content and user
                                                                                                                                           social network, without requiring any explicit user proﬁle
                                                                                                                                           information. With the proposed graph neural networks with
                                                                                                                                           multi-headattentionmechanism,ourmodelcanﬁlteroutthe
                                                                                                                                           noise from the content information and confounding user
                                                                                                                                           contacts, so as to focus on the most important information,
                                                                                                                                           both linguistic and structural, to alleviate the problem of
                                                                                                                                           inference bias when geolocating the users. Extensive exper-
                                                                                                                                           iments have been conducted on large scale datasets which
                                                                                                                                           demonstrate the superior performance of our model against
                                                                                                                                           previousstate-of-the-artUGmethods.Wealsoprovideinter-
                                                                                                                                           pretable results regarding our model and its performance.
                                                                                                                                           One of our immediate future work is to further improve the
                                                                                                                                           UG performance by exploiting multi-aspect features, such
                                                                                                                                           as proﬁle and timezone. In addition, how to better distill
                                                                                                                                           spatio-temporalknowledgeandgeographicalsemanticsfrom
                                                                                                                                           user published content – in addition to indicative words – is
                                                                                                                                           anothertopicofourongoingwork.
                                                                                                                                           REFERENCES
                                                                                                                                             [1]B.Han,P.Cook,andT.Baldwin,‘‘Geolocationpredictioninsocialmedia
                                                                                                                                                    data by ﬁnding location indicative words,’’ in                         Proc. Int. Conf. Comput.
                                                                                                                                                    Linguistics(COLING)              ,2012,pp.1045–1062.
                                                                                                                                             [2]B. Wing and J. Baldridge, ‘‘Hierarchical discriminative classiﬁcation for
                                                                                                                                                    text-basedgeolocation,’’in              Proc.Conf.EmpiricalMethodsNaturalLang.
                                                                                                                                                    Process.(EMNLP)             ,2014,pp.336–348.
                                                                                                                                             [3]B. Han, P. Cook, and T. Baldwin, ‘‘Text-based Twitter user geolocation
                                                                                                                                                    prediction,’’      J.Artif.Intell.Res.       ,vol.49,pp.451–500,Mar.2014.
                                                                                                                                             [4]T.  Huu  Do,  D.  Minh  Nguyen,  E.  Tsiligianni,  B.  Cornelis,  and
                                                                                                                                                    N.   Deligiannis,   ‘‘Multiview   deep   learning   for   predicting   Twitter
                                                                                                                                                    Users’    location,’’    2017,                arXiv:1712.08091           .    [Online].    Available:
                                                                                                                                                    http://arxiv.org/abs/1712.08091
FIGURE3.        Visualization of training samples from 4 randomly selected                                                                   [5]Y. Miura, M. Taniguchi, T. Taniguchi, and T. Ohkuma, ‘‘Unifying text,
regions from GeoText using t-SNE.                                                                                                                   metadata, and user network representations with a neural network for
                                                                                                                                                    geolocationprediction,’’in              Proc. Annu. Meeting Assoc. Comput. Linguis-
aggressively and can effectively group the users from same                                                                                          tics(ACL)     ,2017,pp.1260–1272.
                                                                                                                                             [6]A. Rahimi, T. Cohn, and T. Baldwin, ‘‘Semi-supervised user geolocation
regionstogetherinthelatentspace.                                                                                                                    viagraphconvolutionalnetworks,’’in                     Proc.Annu.MeetingAssoc.Comput.
                                                                                                                                                    Linguistics(ACL)          ,2018,pp.2009–2019.
                                                                                                                                             [7]J. Bakerman, K. Pazdernik, A. Wilson, G. Fairchild, and R. Bahran,
H. DISCUSSION                                                                                                                                       ‘‘Twittergeolocation:Ahybridapproach,’’                        ACMTrans.Knowl.Discovery
Fromtheempiricalobservationsonthreereal-worlddatasets,                                                                                              fromData      ,vol.12,no.3,p.34,Apr.2018.
theproposedmethodMAGNNisabletoestimatethegeolo-                                                                                              [8]X.Zheng,J.Han,andA.Sun,‘‘AsurveyoflocationpredictiononTwitter,’’
                                                                                                                                                    IEEETrans.Knowl.DataEng.                   ,vol.30,no.9,pp.1652–1671,Sep.2018.
cation of Twitter users with higher accuracy than previous                                                                                   [9]E.Amitay,N.Har’El,R.Sivan,andA.Soffer,‘‘Web-a-where:Geotagging
methods. MAGNN achieves superior performance due to its                                                                                             Web content,’’ in          Proc. Int. Conf. Res. Develop. Inf. Retr. (SIGIR)                        , 2004,
ability of effectively fusing content features and network                                                                                          pp.273–280.
                                                                                                                                           [10]B. P. Wing and J. Baldridge, ‘‘Simple supervised document geolocation
features within the attentive graph neural networks archi-                                                                                          with geodesic grids,’’ in             Proc. Annu. Meeting Assoc. Comput. Linguistics
tectures. This also demonstrates the power of the proposed                                                                                          (ACL)    ,2011,pp.955–964.
method on mining hidden features regarding Twitter content                                                                                 [11]S. Roller, M. Speriosu, S. Rallapalli, B. Wing, and J. Baldridge, ‘‘Super-
                                                                                                                                                    vised  text-based  geolocation  using  language  models  on  an  adaptive
and user mention network, while ﬁltering out noisy signals                                                                                          grid,’’ in    Proc. Annu. Meeting Assoc. Comput. Linguistics (ACL)                                 , 2012,
that have been ignored in previous methods. Nevertheless,                                                                                           pp.1500–1510.
it is worthwhile to note that the multi-head attention used in                                                                             [12]A. Ahmed, L. Hong, and A. J. Smola, ‘‘Hierarchical geographical mod-
                                                                                                                                                    eling of user locations from social media posts,’’ in                          Proc. 22nd Int. Conf.
MAGNN may require more memory cost, especially when                                                                                                 WorldWideWeb(WWW)                   ,2013,pp.25–36.
the number of attention heads increases, which restricts the                                                                               [13]W.-H. Chong and E.-P. Lim, ‘‘Tweet geolocation: Leveraging location,
application of our model in resource-limited setting. One of                                                                                        user and peer signals,’’ in             Proc. ACM Conf. Inf. Knowl. Manage. CIKM                             ,
                                                                                                                                                    Nov.2017,pp.1279–1288.
the promising ways of improving memory efﬁciency is to                                                                                     [14]K.S.Jones,‘‘Astatisticalinterpretationoftermspeciﬁcityanditsapplica-
replace multi-head attention in MAGNN with multi-linear                                                                                             tioninretrieval,’’       J.Document.        ,vol.28,no.1,pp.11–21,Jan.1972.
attention mechanism as suggested in [52]. However, this is                                                                                 [15]K.Ren,S.Zhang,andH.Lin,‘‘Whereareyousettlingdown:Geo-locating
                                                                                                                                                    Twitterusersbasedontweetsandsocialnetworks,’’in                             Proc.AsiaInf.Retr.
beyondthescopeofthisworkandleftasourfuturework.                                                                                                     Symp.     Springer,2012,pp.150–161.
95232                                                                                                                                                                                                                                                                                          VOLUME 8, 2020

T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
[16]L. Backstrom, E. Sun, and C. Marlow, ‘‘Find me if you can: Improving                                                                                  [39]P.  Zola,  P.  Cortez,  and  M.  Carpita,  ‘‘Twitter  user  geolocation  using
          geographical prediction with social and spatial proximity,’’ in                                Proc. 19th                                                 Web country noun searches,’’                   Decis. Support Syst.          , vol. 120, pp.50–59,
          Int.Conf.WorldWideWeb(WWW)                       ,2010,pp.61–70.                                                                                          May2019.
[17]C. A. Davis, Jr., G. L. Pappa, D. R. R. de Oliveira, and F. de L. Arcanjo,                                                                            [40]T. Miyazaki, A. Rahimi, T. Cohn, and T. Baldwin, ‘‘Twitter geolocation
          ‘‘Inferring the location of Twitter messages based on user relationships,’’                                                                               using knowledge-based methods,’’ in                       Proc. EMNLP Workshop 4th Work-
          Trans.GIS      ,vol.15,no.6,pp.735–751,Dec.2011.                                                                                                          shopNoisyUser-GeneratedText(W-NUT)                           ,2018,pp.7–16.
[18]L. Kong, Z. Liu, and Y. Huang, ‘‘SPOT: Locating social media users                                                                                    [41]P. Li, H. Lu, N. Kanhabua, S. Zhao, and G. Pan, ‘‘Location inference for
          basedonsocialnetworkcontext,’’                   Proc.VLDBEndowment                 ,vol.7,no.13,                                                         non-geotagged tweets in user timelines,’’                      IEEE Trans. Knowl. Data Eng.                   ,
          pp.1681–1684,Aug.2014.                                                                                                                                    vol.31,no.6,pp.1150–1165,Jun.2019.
[19]E. Rodrigues, R. Assunç                    ¯ao, G. L. Pappa, D. Renno, and W. Meira, Jr.,                                                             [42]W.-H.ChongandE.-P.Lim,‘‘Fine-grainedgeolocationoftweetsintempo-
          ‘‘Exploring multiple evidence to infer users’ location in Twitter,’’                                 Neuro-                                               ralproximity,’’        ACMTrans.Inf.Syst.            ,vol.37,no.2,pp.1–33,Mar.2019.
          computing       ,vol.171,pp.30–38,Jan.2016.                                                                                                     [43]Z.Wu,S.Pan,F.Chen,G.Long,C.Zhang,andP.S.Yu,‘‘Acomprehensive
[20]A.Rahimi,T.Cohn,andT.Baldwin,‘‘Aneuralmodelforusergeolocation                                                                                                   survey on graph neural networks,’’ 2019,                        arXiv:1901.00596           . [Online].
          and lexical dialectology,’’ in               Proc. 55th Annu. Meeting Assoc. Comput.                                                                      Available:http://arxiv.org/abs/1901.00596
          Linguistics     ,2017,pp.209–216.                                                                                                               [44]J. Zhou, G. Cui, Z. Zhang, C. Yang, Z. Liu, L. Wang, C. Li, and M. Sun,
[21]T. N. Kipf and M. Welling, ‘‘Semi-supervised classiﬁcation with graph                                                                                           ‘‘Graph neural networks: A review of methods and applications,’’ 2018,
          convolutional networks,’’ in                Proc. Int. Conf. Learn. Represent. (ICLR)                       ,                                             arXiv:1812.08434           .[Online].Available:http://arxiv.org/abs/1812.08434
          2017.                                                                                                                                           [45]D. P. Kingma and J. Ba, ‘‘Adam: A method for stochastic optimization,’’
[22]P.Velickovic,G.Cucurull,A.Casanova,A.Romero,P.Liò,andY.Bengio,                                                                                                  in Proc.Int.Conf.Learn.Represent.(ICLR)                       ,2015.
          ‘‘Graphattentionnetworks,’’in                 Proc.Int.Conf.Learn.Represent.(ICLR)                       ,                                      [46]J.Eisenstein,B.O’Connor,N.A.Smith,andE.P.Xing,‘‘Alatentvariable
          2018.                                                                                                                                                     model for geographic lexical variation,’’ in                       Proc. Annu. Meeting Assoc.
[23]A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,                                                                                         Comput.Linguistics(ACL)                ,2010,pp.1277–1287.
          L.Kaiser, and I. Polosukhin, ‘‘Attention is all you need,’’ in                              Proc. Adv.                                          [47]S.Bird,E.Klein,andE.Loper,                       NaturalLanguageProcessingWithPython:
          NeuralInf.Process.Syst.(NIPS)                 ,2017,pp.5998–6008.                                                                                         Analyzing Text With the Natural Language Toolkit                             . Newton, MA, USA:
[24]J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, ‘‘BERT: Pre-training                                                                                          O’ReillyMedia,2009.
          of deep bidirectional transformers for language understanding,’’ in                                   Proc.                                     [48]R. W. Sinnott, ‘‘Virtues of the haversine,’’                             Sky Telescope        , vol. 68, no. 2,
          North Amer. Chapter Assoc. Comput. Linguistics, Hum. Lang. Technol.                                                                                       p.159,1984.
          (NAACL-HLT)           ,2019,pp.4171–4186.                                                                                                       [49]J.Leskovec,A.Rajaraman,andJ.D.Ullman,                                 MiningofMassiveDatasets                .
[25]Z.Cheng,J.Caverlee,andK.Lee,‘‘Youarewhereyoutweet:Acontent-                                                                                                     Cambridge,U.K.:CambridgeUniv.Press,2014.
          based approach to geo-locating Twitter users,’’ in                           Proc. 19th ACM Int.                                                [50]T.Mikolov,I.Sutskever,K.Chen,G.S.Corrado,andJ.Dean,‘‘Distributed
          Conf.Inf.Knowl.Manage.(CIKM)                     ,2010,pp.759–768.                                                                                        representationsofwordsandphrasesandtheircompositionality,’’in                                    Proc.
[26]A. Rahimi, T. Cohn, and T. Baldwin, ‘‘Twitter user geolocation using a                                                                                          Adv.NeuralInf.Process.Syst.               ,2013,pp.3111–3119.
          uniﬁedtextandnetworkpredictionmodel,’’in                          Proc.Annu.MeetingAssoc.                                                       [51]L.vanderMaatenandG.Hinton,‘‘Visualizingdatausingt-SNE,’’                                            J.Mach.
          Comput.Linguistics(ACL)                ,2015,pp.630–636.                                                                                                  Learn.Res.      ,vol.9,pp.2579–2605,Nov.2008.
[27]A. Rahimi, T. Baldwin, and T. Cohn, ‘‘Continuous representation of                                                                                    [52]X. Ma, P. Zhang, S. Zhang, N. Duan, Y. Hou, M. Zhou, and D. Song, ‘‘A
          location for geolocation and lexical dialectology using mixture density                                                                                   tensorized transformer for language modeling,’’ in                            Proc. Adv. Neural Inf.
          networks,’’ in         Proc. Conf. Empirical Methods Natural Lang. Process.                                ,                                              Process.Syst.       ,2019,pp.2229–2239.
          2017,pp.167–176.
[28]A.Rahimi,D.Vu,T.Cohn,andT.Baldwin,‘‘Exploitingtextandnetwork
          context for geolocation of social media users,’’ in                          Proc. Conf. North
          Amer. Chapter Assoc. Comput. Linguistics Hum. Lang. Technol.                                      , 2015,
          pp.1362–1367.
[29]Q.LeandT.Mikolov,‘‘Distributedrepresentationsofsentencesanddoc-
          uments,’’in       Proc.Int.Conf.Mach.Learn.(ICML)                      ,2014,pp.1188–1196.
[30]J.  Chung,  C.  Gulcehre,  K.  Cho,  and  Y.  Bengio,  ‘‘Empirical  evalua-                                                                                                                           TING ZHONG             received the B.S. degree in com-
          tion of gated recurrent neural networks on sequence modeling,’’ 2014,                                                                                                                           puterapplicationandtheM.S.degreeincomputer
          arXiv:1412.3555          .[Online].Available:http://arxiv.org/abs/1412.3555                                                                                                                     softwareandtheoryfromBeijingNormalUniver-
[31]D. Bahdanau, K. Cho, and Y. Bengio, ‘‘Neural machine translation by                                                                                                                                   sity, Beijing, China, in 1999 and 2002, respec-
          jointlylearningtoalignandtranslate,’’in                   Proc.Int.Conf.Learn.Represent.                                                                                                        tively, and the Ph.D. degree in information and
          (ICLR)    ,2015.                                                                                                                                                                                communication engineering from the University
[32]P. Hamouni, T. Khazaei, and E. Amjadian, ‘‘TF-MF: Improving mul-                                                                                                                                      of Electronic Science and Technology of China,
          tiview representation for Twitter user geolocation prediction,’’ in                                 Proc.                                                                                       Chengdu, China, in 2009. From 2003 to 2009,
          IEEE/ACM  Int.  Conf.  Adv.  Social  Netw.  Anal.  Mining  (ASONAM)                                         ,                                                                                   she was a Lecturer with the University of Elec-
          Aug.2019,pp.543–545.                                                                                                                                                                            tronicScienceandTechnologyofChina,Chengdu,
[33]J.McGee,J.Caverlee,andZ.Cheng,‘‘Locationpredictioninsocialmedia                                                                                       China,whereshehasbeenanAssociateProfessorsince2010.Herresearch
          based on tie strength,’’ in             Proc. ACM Int. Conf. Inf. Knowl. Manage.                                                                interestsincludedeeplearning,socialnetworks,andcloudcomputing.
          (CIKM)      ,2013,pp.459–468.
[34]D.Jurgens,‘‘That’swhatfriendsarefor:Inferringlocationinonlinesocial
          media platforms based on social relationships,’’ in                           Proc. Int. AAAI Conf.
          WeblogsSocialMedia              ,2013.
[35]R. Li, S. Wang,H. Deng, R. Wang, and K.C.-C. Chang, ‘‘Towards social
          user proﬁling: Uniﬁed and discriminative inﬂuence model for inferring
          home locations,’’ in            Proc. SIGKDD Int. Conf. Knowl. Discovery Data
          Mining(KDD)          ,2012,pp.1023–1031.                                                                                                                                                        TIANLIANG  WANG                   received the B.S. degree
[36]A. Grover and J. Leskovec, ‘‘node2vec: Scalable feature learning for                                                                                                                                  from the College of Computer Science and Elec-
          networks,’’in        Proc.22ndACMSIGKDDInt.Conf.Knowl.DiscoveryData                                                                                                                             tronic Engineering, Hunan University, Changsha,
          Mining     ,Aug.2016,pp.855–864.                                                                                                                                                                China, in 2018. He is currently pursuing the M.S.
[37]J.Qiu,Y.Dong,H.Ma,J.Li,K.Wang,andJ.Tang,‘‘Networkembedding                                                                                                                                            degree with the University of Electronic Science
          asmatrixfactorization:UnifyingDeepWalk,LINE,PTE,andnode2vec,’’                                                                                                                                  and Technology of China. His current research
          in  Proc. 11th ACM Int. Conf. Web Search Data Mining (WSDM)                                        , 2018,                                                                                      interests include social network knowledge dis-
          pp.459–467.
[38]W.Hamilton,Z.Ying,andJ.Leskovec,‘‘Inductiverepresentationlearning                                                                                                                                     covery, spatio-temporal data mining, and graph
          on large graphs,’’ in           Proc. Adv. Neural Inf. Process. Syst. (NIPS)                       , 2017,                                                                                      neuralnetworks.
          pp.1024–1034.
VOLUME 8, 2020                                                                                                                                                                                                                                                        95233

                                                                                             T. Zhong       et al.  : Multiple-Aspect Attentional Graph Neural Networks for Online Social Network User Localization
                                                JIAHAO  WANG                received the M.S. and Ph.D.                                                                                                   FAN ZHOU           receivedtheB.S.degreeincomputer
                                                degrees from the University of Electronic Sci-                                                                                                            science from Sichuan University, China, in 2003,
                                                enceandTechnologyofChina,in2004and2007,                                                                                                                   and the M.S. and Ph.D. degrees from the Uni-
                                                respectively.  He  was  a  Postdoctoral  Research                                                                                                         versity of Electronic Science and Technology of
                                                Associate with The Hong Kong University, from                                                                                                             China, in 2006 and 2011, respectively. He is cur-
                                                2007 to 2009. He was a Visiting Scholar with the                                                                                                          rently an Associate Professor with the School of
                                                Illinois Institute of Technology (IIT), USA, from                                                                                                         InformationandSoftwareEngineering,University
                                                2013to2014.HeiscurrentlyanAssociateProfes-                                                                                                                of Electronic Science and Technology of China.
                                                sor with the School of Information and Software                                                                                                           His  research  interests  include  machine  learn-
                                                Engineering,UniversityofElectronicScienceand                                                                                                              ing, spatio-temporal data management, and social
TechnologyofChina.HisresearchinterestsincludetheIoTdatamining,the                                                                                                                                         networkknowledgediscovery.
IoTsecurity,anddeeplearningforIoT.
                                                JIN WU        received the B.S. degree in automatic
                                                control and the M.S. and Ph.D. degrees from the
                                                University of Electronic Science and Technology
                                                of China, in 1993, 1996, and 2004, respectively.
                                                She is currently an Associate Professor with the
                                                University of Electronic Science and Technology
                                                of China. Her research interests include machine
                                                learning, knowledge mapping, software develop-
                                                menttechniques,andprocesstechnology.
95234                                                                                                                                                                                                                                                                                          VOLUME 8, 2020

