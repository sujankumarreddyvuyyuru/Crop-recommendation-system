# Crop-recommendation-system



ABSTRACT:



In order to mitigate the agrarian crisis in the current status quo, there is a need for better recommendation systems to alleviate the crisis by helping the farmers to make an informed decision before starting the cultivation of crops.

One of the oldest nations currently engaged in agriculture is India. However, due to globalisation, agricultural trends have radically changed in recent years. The state of Indian agriculture has been impacted by numerous reasons. In order to regain health, many innovative technologies have been developed. Crop recommendation is one of these methods. In India, precision agriculture is just beginning. The "site-specific" farming technology is called crop suggestion. It has given us the benefit of efficient input, output, and superior farming decision-making. Crop recommendation has made better advances, although it still has certain problems. There are numerous systems that offer the inputs for various farming fields.


Systems make suggestions for crops, fertilisers, and even farming methods. One important area of precision agriculture is crop recommendation. Crop recommendations are based on a number of factors. To tackle problems with crop selection, precision agriculture tries to determine these factors in a site-specific manner. Although the "site-specific" approach has improved the outcomes, such systems still require oversight. Not all precision agriculture techniques yield reliable outcomes. However, it is crucial that the advice offered in agriculture are correct and precise because mistakes could result in material loss and financial loss. Numerous studies are being conducted in an effort to develop a reliable and effective model for crop prediction. Ensembling is one of these methods that is comprised of such research projects. This study suggests a system that uses the voting method to create an effective and accurate model among the numerous machine learning techniques currently being employed in this industry. To predict the crop yield, selected Machine Learning algorithms such as Logistic Regression, Random Forest (RF), Decision Trees, Gaussian Naïve Bayes , Ensemble learning,Voting classifier are used.

   
KEY WORDS:
              Logistic Regression
              Random Forest (RF)
              Decision Trees
              Gaussian Naïve Bayes 
              Ensemble learning
              Voting classifier






INTRODUCTION:

                Recommendation of crops is one major domain in precision agriculture. Recommendation of crops is dependent on various parameters. Precision agriculture aims in identifying these parameters in a site-specific manner in order to resolve issues regarding crop selection. The “site-specific” technique has improved the results yet there is a need to supervise the results of such systems. Not all precision agriculture systems provide accurate results. But in agriculture it is important that the recommendations made are accurate and precise because incase of errors it may lead to heavy material and capital loss. Many research works is being carried out, in order to attain an accurate and efficient model for crop prediction. Ensembling is one such technique that is included in such research works. Among these various machine learning techniques that are being used in this field; this paper proposes a system that uses the voting method to build an efficient and accurate model.


RELATED WORK:
 
   [1]The system used is a subscription-based system containing personal information from each of the registered farmers. The system contains modules that manage previously planted crop information collected from different sources and display matching crops that can be planted[2]. Finally,a feedback system is provided so that developers can make necessary changes if farmers encounter problems using the system.RSF is a recommended system for farmers, as mentioned in article [3], considering location collection module, data analysis and storage module, crop production database and geography database. The similar location detection module identifies locations that are similar to the user's location and looks for similar crops planted at those locations.Recommendations for users are therefore generated based on the similarity matrix. The location module uses the Google API service to determine the user's current location and identify similar locations. However, the system does not receive user feedback to improve the process.
The system of the paper [4], proposed by author S. Pudumalar and related co-authors, uses an ensemble technique called the Majority Voting Technique. It combines the power of multiple models to achieve higher predictive accuracy. The methods used are Random Trees, KNN and Naive Bayes for Ensemble, so if one method makes a wrong prediction, the other models are likely to make a correct prediction, and the majority voting method is used.
In this paper [5], we propose a method called Crop Selection Method (CSM) to solve the crop selection problem and maximize the crop net yield rate for the whole season and subsequently achieve the maximum economic growth of the country. Did. The proposed method can improve the net yield rate of crops.
 
   In this article [6], a smart crop recommendation system that can be easily used by farmers across India is proposed and implemented. The system helps farmers make informed decisions about which crops to plant based on a variety of environmental and geographic factors. We also implemented a second-order system called Rainfall Predictor that forecasts precipitation for the next 12 months.
The development of an accurate agricultural production forecast system based on monthly real-time weather is discussed in this study[7]. The output of agricultural goods is difficult to forecast because to yearly irregular weather patterns and rapid regional climate change brought on by human-caused global warming. A technique for forecasting agricultural yields based on current weather conditions is urgently needed. This research explains how to set up forecasting systems and analyse big volumes of weather data (monthly, daily). Using data from 33 years of agrometeorological observations, construct a nonparametric statistical model. The constructed model will forecast ultimate production using monthly weather data. Results from simulations are included in this publication.
[8] In this study, we present a system that uses data mining techniques to predict categories of analyzed soil datasets. These predicted categories indicate yield. The crop yield prediction problem is formalized as a classification rule using Naive Bayes and K nearest neighbors.
This study [9] addresses the problem of crop selection and We proposed a crop selection method (CSM) to maximize the net production rate of crops throughout the season and maximize the economic growth of the country. The strategy proposed by has the potential to increase the net yield of crops.
 
   This study [10] aims to develop a smart crop recommendation system that can be used by farmers across India. Proposed and implemented. The system helps farmers make informed choices about which crops to grow, based on a variety of environmental and geographic parameters. I also set up a backup mechanism called Rainfall Predictor that predicts precipitation over the next 12 months. The system proposed by implements various machine learning techniques such as decision trees, K Nearest Neighbors (K- NN), random forests, neural networks, and performs multi-label classification. This system proposed by achieved an accuracy of 71% using a rainfall forecasting model and an accuracy of 91.00% using neural network techniques to generate the appropriate his forecasting system
This proposed[11] technique is based on a soil database and used to identify specific crops. Farmers will benefit from this research by increasing agriculturalproductivity, minimizing soil degradation in arable land, reducing chemical use in crop production, and maximizing water use efficiency. 4,444 farmers will benefit from this study by increasing agricultural productivity, minimizing soil degradation on 4,444 acres, reducing chemical use in crop production, and improving water efficiency
This proposed [12] approach is used to accurately recommend optimal crops based on soil type and characteristics such as average rainfall and surface temperature. Random Forest, Naive Bayes, and Linear SVM are among the machine learning algorithms used in this proposed system. The input soil dataset was classified by this crop recommendation system into two his crop types: Kharif and Rabi


 
   The authors of [13] stated that nationally planned guiding principles in the field of agricultural development require crop yield forecasting to help farmers reduce chemical use in crop production and prevent soil degradation. concludes. Productivity and efficient use of water resources. Data mining techniques help farmers select suitable seeds for sowing based on soil requirements and ensure increased productivity to benefit from such techniques [14]. Majority voting methods such as random number trees , K-Nearest Neighbor, and Naive Bayes algorithms are used to build ensemble recommendation models to accurately and efficiently suggest crops based on site- specific parameters.

   
This paper [15] describes the requirements and plans for the development of a software model for precision agriculture. An in- depth analysis of the basics of precision agriculture. The authors take the fundamentals of precision agriculture and develop models to support them. This white paper describes a model that applies precision agriculture (PA) principles to small open farms at the individual farm and crop levels to control variability to some extent. The overarching objective of this model is to provide direct advisory services to even the smallest farmers at the smallest acreage level using the most accessible technologies such as SMS and email. This model was developed for the Kerala scenario where the average farm size is much smaller than most parts of India. Therefore, this model can only be used with minor modifications elsewhere in India. This paper [16] conducts a comparative study of classification algorithms and their performance in predicting yields in precision agriculture. These algorithms were implemented on datasets collected over several years in soybean yield forecasting. The yield prediction algorithms used in this paper are support vector machines, random forests, Neural Networks, REPTree, Bagging, Bayes. The final conclusion drawn is that among the above
 algorithms, slumping has the smallest error deviation with a mean absolute error of 18985.7864, so slumping is the best yield prediction algorithm. This paper [17] describes the need for crop yield forecasting and its support in national strategic policy-making in agriculture. A framework eXtensible Crop Yield Prediction Framework (XCYPF) has been developed. You can flexibly incorporate various yield prediction methods. Tools have been developed to predict the yield of various crops using subordinate variables and independent variables.

 
The paper [18] describes the use of agricultural data using data mining and visual data mining techniques. In this paper, we can reduce highdimensional agricultural data to a smaller size to gain useful knowledge related to yield and inputs (such as fertilizer). The techniques used are selforganizing maps and multidimensional scaling techniques (Sammon mapping) to reduce the data. The conclusion that can be drawn from this is that selforganizing maps are better for large datasets, and Sammon's mapping is better for smaller datasets. A paper [19] demonstrates the importance of crop selection and discusses the factors that determine crop selection, such as production rates, market prices, and government policies. In this paper, we propose a crop selection method (CSM) to solve the problem of crop selection and improve the net yield rate of crops. Considering factors such as weather, soil type, water density and plant type, we propose different plants to be selected for each season. Predicted values of influential parameters determine the accuracy of CSM. Therefore, forecasting methods with improved accuracy and performance should be included. Paperbased data mining techniques [20] are used to estimate cereal yields in key regions of Bangladesh. This methodology consists of his two parts: clustering (to create district clusters) and classification using k-NN (k- nearest neighbors), linear regression, and (ANN) artificial neural networks in rapid miner tools. will be The prediction accuracy is in
 
   the range of 90-95. The dataset contained 5 environmental variables, 3 biological variables and 2 regional variables to determine yields in different districts. This paper suggests future work on geospatial analysis to improve accuracy This paper[21] even aims to solve the problems for the ensemble learning. A method is proposed to select the optimal set of classifiers from a pool of classifiers. This proposal aims to achieve higher accuracy and performance. Based on its accuracy and classification performance, a method called SAD has been proposed.
The paper [22] proposes various classification methods to classify liver disease datasets. This paper emphasizes the need for accuracy as it depends on the dataset and learning algorithm. Classification algorithms such as J48, Naive Bayes, ANN, ZeroR, 1BK, and VFI are used to classify these diseases and compare their efficacy and correction rates. It was concluded that all classifiers except Naive Bayes showed improved predictive performance. Among the proposed algorithms multi layer perceptrons showed the highest accuracy. The soil data set from the paper [23] is analyzed and the categories are predicted. From the predicted soil categories, yields are determined as classification rules. Naïve Bayes and k-Nearest Neighbor algorithms are used for yield prediction. Future work indicated is to build efficient models using various classification techniques such as support vector machines, principal component analysis, etc.
Several types of soil are available in India. These include alluvial soil (cotton,rice),blacksoil(sugarcane, sunflower),redsoil(corn,ragi), lateric soil(legu mes, tea, coffee). A lot of research has been done to improve agricultural planning. Machine learning techniques can be used to recommend harvests. The research focus is on predicting agricultural yields using various machine learning approaches. Predict agricultural yields based on historical data such as precipitation, temperature,yield, and pesticides. Each of these data properties is evaluated and multiple machine learning methods are used to train the data and create a model[24].

Machine learning starts with pieces of data such as financial transactions, people, and photos. Information is collected and processed to be used as training data for machine learning systems. The software shows better results when the data exceeds. The developer then chooses which ML model to use, he inputs the data and trains the system to find patterns or make predictions on his own. Research on machine learning algorithms was done in research papers by Rashi Agarwal [25].

The system helps farmers make informed decisions about which crops to plant based on a variety of environmental and geographic factors. They used decision trees, ANNs, randomforests, and neural networks.The highest accuracy has been found in the neural network. Priyadharshini A [26] conducted a survey ofmachine learning algorithms in a research paper.
The technology helps farmers select the right crops and provides data that regular farmers don't have, reducing crop failures and lowering productivity. Various machine learning algorithms were applied. The neural network was the most accurate of all. Mayank Champaneri [27] conducted a study to predict yield using data mining techniques. We used theRandom Forest Classifier because it can perform classification and regression tasks. A user-friendly website has been created that anyone can use to predict his yield for selected crops by providing climate data for the region.
Machine learning prediction algorithms [28][29] require highly accurate estimation based on previously learned data. Historical information in predictive analytics is the use of data, statistical methods, and machine learning approaches to predict future outcomes.
 
A high level of production in the agricultural sector is paramount to the economic well-being of most countries and the well-being of its population. A major challenge for most farmers, especially subsistence farmers who make significant contributions to the agricultural sector, is being able to decide what and where to grow [29] [30] [31].
This Crop Recommendation system should be able to recommend crops based on soil properties alone. This makes such data easily accessible, making it more accessible to most farmers. Using the literature, machine learning algorithms is used including SVM and RF to develop a highly accurate CR system considering only the three mostimportant soil nutrients: nitrogen, potassium and phosphors. and soil pH. This means that farmers can use CR systems with minimal resource investment as this information is more readily available. The main contributions to the literature are :a) to develop highly accurate CR system that is more accessible to farmers and uses only soil attributes such as the three most important soil nutrients : nitrogen, potassium and phosphorus. and soil pH. b) Compare different machine learning algorithms, including SVM and RF algorithms, to see which algorithm is best suited for developing CR systems using only ground attribute data. Developing crop recommendation systems at all levels requires information and knowledge about plants and the soil in which they are grown, as well as other environmental factors that affect plant growth, such as temperature, rainfall, and humidity [ 29][33][32].
Soil attributes that can be considered are soil nutrients, soil pH, and soil color [37][35][34], to name a few. When choosing which crop to use, you can choose the crop that is most prevalent in the surveyed locations [36] [31] [38]. Machine learning-based models are suitable for developing CR systems because they provide the best (most accurate) soil classification, independent of the performance measures used to evaluate the performance of the algorithm [ 36]. [37] Use Naive Bayesian (NB) and k-Nearest Neighbor (KNN) for soil classification to allow crop yield prediction using available data. It also suggests that classification algorithms such as support vector machines can be used to create more efficient models. This is confirmed by his [38] who developed his CR system using a kernel- based support vector machine (SVM) and reports an accuracy of 94.95%., however, [31] found that random forest (RF)- based models perform better than their SVM counterparts when it comes to soil classification. They report accuracies of 75.73% and 86.35% for SVM and RF, respectively.





METHODOLGY:
                1. Training the model by different types of algorithms 1.Logistic regression
                2.Decision Tree
                3.Random Forest Classifier
                4. Naive Bayes Classifier
                5. Creating an ensemble model with voting classifier and predicting
                  the highest probable output.



                  
Logistic regression:



It is a process of modeling the probability of a discrete outcome given an input variable. The most common logistic regression models a binary outcome; something that can take two values such as true/false, yes/no, and so on.



Decision Trees:
 These are a type of Supervised Machine Learning (that is you explain what the input is and what the corresponding output is in the training data) where the data is continuously split according to a certain parameter. 
 The tree can be explained by two entities, namely decision nodes and leaves. The leaves are the decisions or the final outcomes. And the decision nodes are where the data is
 split.
 
 Random Forest:
 
 It is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the
 performance of the model.

 
 Gaussian Naïve bayes:

 
When working with continuous data, an assumption often taken is that the continuous values associated with each class are distributed according to a normal (or Gaussian) distribution. Gaussian Naive Bayes supports continuous valued features and models each as conforming
 to a Gaussian (normal) distribution.
An approach to create a simple model is to assume that the data is described by a Gaussian distribution with no co-variance (independent dimensions) between dimensions. This model can be fit by simply finding the mean and standard deviation of the points within each label, which is all what is needed to define such a distribution.
 
Voting Classifier:


It is a machine learning model that trains on an ensemble of numerous models and predicts an output (class) based on their highest probability of chosen class as the output.
It simply aggregates the findings of each classifier passed into Voting Classifier and predicts the output class based on the highest majority of voting. The idea is instead of creating separate dedicated models and finding the accuracy for each them, we create a single model which trains by these models and predicts output based on their combined majority of voting for each output class.


Ensemble Method:


Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would. This has been the case in a number of machine learning competitions, where the winning solutions used ensemble methods.


IMPLEMENTATION: 
DATASET:
Attribute Information:
                          1. N(value of nitrogen in Kgs)
                          2. P(value of phosporous in Kgs)
                          3. K(value of pottasium in Kgs) 4. Temperature(in Celsius)
                          5. Humidity(in percentage)
                          6. pH value of soil
                          7. Rainfall(in mm)




The library’s used are:
                        • Flask,request,pandas,pickle,
                        • from sklearn import model_selection
                        • from sklearn.linear_model import LogisticRegression
                        • from sklearn.tree import DecisionTreeClassifier
                        • from sklearn.svm import SVC
                        • from sklearn.ensemble import RandomForestClassifier
                        • from sklearn.ensemble import VotingClassifier
                        • from sklearn.naive_bayes import GaussianNB




By Entering the values of nitrogen(N),Potassium(K),Phosphorus(P),Temperature(T),Humidity(H ),Ph value of soil ,Rainfall and by clicking on the submit button . We are going to get respond from website this crop is recommended .



 Conclusion:


 
 India is a nation in which agriculture plays a prime role. In prosperity of the farmers, prospers the nation. Thus our work would help farmers in so wing the right seed based on soil requirements to increase productivity and acquire profit out of such a technique. Thus the farmer’s can plant the right crop increasing his yield and also increasing the overall productivity of the nation. Our future work is aimed at an improved data set with large number of attributes and also implements yield prediction.
